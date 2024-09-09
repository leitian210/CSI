import random

import torch
import numpy as np
import time
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import CSI_trans
from utils import cal_ranks, cal_performance
import random


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = CSI_trans(args, loader)
        self.model.cuda()

        self.loader = loader
        self.KG = loader.KG
        self.M_sub = loader.M_sub

        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.t_time = 0
        self.train_data_dealt = self.loader.to_train(self.loader.train_data_split, self.n_batch)
        self.valid_q_dealt, self.valid_a_dealt = self.loader.to_evaluate(self.n_tbatch, 'valid')
        self.test_q_dealt, self.test_a_dealt = self.loader.to_evaluate(self.n_tbatch, 'test')

    def get_dynamic(self, tau_):
        # atleast = max(tau_ - 250, 0)
        m1 = self.KG[:, 3] < tau_
        # m2 = self.KG[:, 3] >= atleast
        # m3 = self.KG[:, 3] == -1
        d_mask = m1
        # d_mask = m1 * m2 + m3
        dynamic_KG = self.KG[d_mask]
        dynamic_M_sub = self.M_sub[d_mask]
        return dynamic_KG, dynamic_M_sub

    def train_batch(self, ):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        coinf = 0.5
        coinf_caus = 0.5
        t_time = time.time()
        self.model.train()

        for i, triple in enumerate(self.train_data_dealt):
            dynamic_KG, dynamic_M_sub = self.get_dynamic(triple[0, 3])

            self.model.zero_grad()
            scores, scores_s, scores_caus = self.model(triple[:, 0], triple[:, 1], triple[:, 3], dynamic_KG,
                                                       dynamic_M_sub)

            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]

            # pos_scores_s = scores_s[[torch.arange(len(scores_s)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]
            uniform_target = torch.ones_like(scores_s, dtype=torch.float).cuda() / self.n_ent
            c_loss = F.kl_div(scores_s, uniform_target, reduction='batchmean')

            pos_scores_caus = scores_caus[
                [torch.arange(len(scores_caus)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]
            max_n_caus = torch.max(scores_caus, 1, keepdim=True)[0]
            loss_caus = torch.mean(
                - pos_scores_caus + max_n_caus + torch.log(torch.sum(torch.exp(scores_caus - max_n_caus), 1)))

            loss = torch.mean(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
            loss = loss + c_loss * coinf + loss_caus * coinf_caus
            loss.backward()
            self.optimizer.step()
            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()
        self.t_time += time.time() - t_time

        valid_mrr, out_str = self.evaluate()
        # self.loader.shuffle_train()
        # random.shuffle(self.train_data_dealt)
        return valid_mrr, out_str

    def evaluate(self, ):
        batch_size = self.n_tbatch
        i_time = time.time()

        ranking = []
        self.model.eval()

        for i, query in enumerate(self.test_q_dealt):
            answer = self.test_a_dealt[i]
            subs = query[:, 0]
            rels = query[:, 1]
            taus = query[:, 2]
            objs = np.zeros((len(query), self.loader.n_ent))
            for j in range(len(query)):
                objs[j][answer[j]] = 1
            dynamic_KG, dynamic_M_sub = self.get_dynamic(taus[0])
            scores = self.model(subs, rels, taus, dynamic_KG, dynamic_M_sub, mode='test').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.filters[(subs[i], rels[i], taus[i])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h3, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time
        v_mrr, v_h1, v_h3, v_h10 = t_mrr, t_h1, t_h3, t_h10
        out_str = '[VALID] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n' % (
        v_mrr, v_h1, v_h3, v_h10, t_mrr, t_h1, t_h3, t_h10, self.t_time, i_time)
        return v_mrr, out_str

