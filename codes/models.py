import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
import random


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.coe_timediff = 0.2

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.W_h_s = nn.Linear(in_dim, out_dim, bias=False)

        self.tau_emb = nn.Embedding(366, in_dim)
        self.Wtau_attn = nn.Linear(in_dim, attn_dim, bias=False)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, in_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, in_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, in_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, in_dim))

    def forward(self, q_sub, q_rel, q_tau, hidden, edges, n_node, old_nodes_new_idx, training=False, add_sampling=True):
        # edges:  [batch_idx, head, rela, tail, tau, old_idx, new_idx]
        sub = edges[:, 5]
        rel = edges[:, 2]
        obj = edges[:, 6]
        tau = edges[:, 4]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        tau = torch.where(tau >= 0, tau, q_tau)
        delta_tau = tau - q_tau

        h_tau1 = self.weight_t1 * delta_tau.unsqueeze(1) + self.bias_t1
        h_tau2 = torch.sin(self.weight_t2 * delta_tau.unsqueeze(1) + self.bias_t2)
        h_hau = h_tau1 + h_tau2

        message = hs + hr + h_hau
        alpha = self.w_alpha(
            nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr) + self.Wtau_attn(h_hau)))
        add_sampling = False
        if add_sampling:
            alpha = self.sampling(alpha, training=training)
            alpha = torch.sigmoid(alpha)
            info_loss = self.get_info_loss(alpha)
        else:
            alpha = torch.sigmoid(alpha)
            info_loss = 0.
        alpha_s = 1 - alpha


        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))

        message_s = message
        message_s = alpha_s * message_s
        message_agg_s = scatter(message_s, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new_s = self.act(self.W_h_s(message_agg_s))

        return hidden_new, hidden_new_s

    def get_info_loss(self, att):
        r = 0.8
        info_loss = att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)
        info_loss = info_loss.mean()
        return info_loss

    def sampling(self, att_log_logits, training=True):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = (att_log_logit + random_noise) / temp
        else:
            att_bern = att_log_logit
        # att_bern = att_bern.sigmoid()
        return att_bern


class CSI_trans(torch.nn.Module):
    def __init__(self, params, loader):
        super(CSI_trans, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.W_final_s = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.gate_s = nn.GRU(self.hidden_dim, self.hidden_dim)

        self.cat_or_add = 'add'
        if self.cat_or_add == "cat":
            self.W_final_caus = nn.Linear(self.hidden_dim * 2, 1, bias=False)  # get score
        elif self.cat_or_add == "add":
            self.W_final_caus = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        else:
            assert False

    def forward(self, subs, rels, taus, dynamic_KG, dynamic_M_sub, mode='train'):
        n = len(subs)

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        q_tau = taus[0]

        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        h0_s = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()
        scores_all = []
        info_losses = 0.
        training = True if mode == 'train' else False
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), dynamic_KG,
                                                                        dynamic_M_sub, mode=mode)
            sampling = True if i == self.n_layer - 1 else False
            hidden, hidden_s = self.gnn_layers[i](q_sub, q_rel, q_tau, hidden, edges, nodes.size(0), old_nodes_new_idx,
                                                  training=training, add_sampling=sampling)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            h0_s = torch.zeros(1, nodes.size(0), hidden_s.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0_s)
            hidden_s = self.dropout(hidden_s)
            hidden_s, h0_s = self.gate_s(hidden_s.unsqueeze(0), h0_s)
            hidden_s = hidden_s.squeeze(0)

            # info_losses += info_loss
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()  # non_visited entities have 0 scores
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores

        scores_s = self.W_final_s(hidden_s).squeeze(-1)
        # scores_s = torch.sigmoid(scores_s)
        scores_all_s = torch.zeros((n, self.loader.n_ent)).cuda()  # non_visited entities have 0 scores
        scores_all_s[[nodes[:, 0], nodes[:, 1]]] = scores_s
        scores_all_s = F.log_softmax(scores_all_s, dim=-1)
        # scores_all_s = torch.softmax(scores_all_s, dim=-1)
        # uniform_target = torch.ones_like(scores_all_s, dtype=torch.float).cuda() / self.loader.n_ent
        # c_loss = F.kl_div(scores_all_s, uniform_target, reduction='batchmean')

        num = hidden.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.cat_or_add == "cat":
            x = torch.cat((hidden_s[random_idx], hidden), dim=1)
        else:
            x = hidden_s[random_idx] + hidden

        scores_caus = self.W_final_caus(x).squeeze(-1)
        scores_all_caus = torch.zeros((n, self.loader.n_ent)).cuda()  # non_visited entities have 0 scores
        scores_all_caus[[nodes[:, 0], nodes[:, 1]]] = scores_caus

        if training:
            return scores_all, scores_all_s, scores_all_caus
        else:
            return scores_all



