import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import time

class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir

        with open(os.path.join(task_dir, 'entity2id.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip().split('\t')[0]
                self.entity2id[entity] = n_ent
                n_ent += 1

        with open(os.path.join(task_dir, 'relation2id.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation, _ = line.strip().split('\t')
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        self.filters = defaultdict(lambda:set())

        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
        self.all_triple = self.fact_triple + self.train_triple + self.valid_triple + self.test_triple
    
        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)
        self.all_data = self.double_triple(self.all_triple)

        self.train_data_split = self.split_by_time(self.train_data)
        # self.valid_data_split = self.split_by_time(self.valid_data)
        # self.test_data_split = self.split_by_time(self.test_data)
        self.load_graph(self.all_data)
        # self.load_graph_fact(self.fact_data)

        self.valid_q, self.valid_a, self.valid_q_split, self.valid_a_split = self.load_query(self.valid_data)
        self.test_q,  self.test_a, self.test_q_split, self.test_a_split  = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        gran = 1
        if '18' in self.task_dir:
            gran = 24
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                line = line.strip().split('\t')
                s= int(line[0])
                r = int(line[1])
                o = int(line[2])
                tau = int(int(line[3])/gran)
                triples.append([s, r, o, tau])
                self.filters[(s, r, tau)].add(o)
                self.filters[(o, r + self.n_rel, tau)].add(s)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            s, r, o, tau = triple
            new_triples.append([o, r+self.n_rel, s, tau])
        return triples + new_triples

    def split_by_time(self, data):
        snapshot_list = []
        snapshot = []
        snapshots_num = 0
        latest_t = 0
        for i in range(len(data)):
            t = data[i][3]
            train = data[i]

            if latest_t != t:
                # show snapshot
                latest_t = t
                if len(snapshot):
                    snapshot_list.append(np.array(snapshot).copy())
                    snapshots_num += 1
                snapshot = []
            snapshot.append(train)

        if len(snapshot) > 0:
            snapshot_list.append(np.array(snapshot).copy())
            snapshots_num += 1

        return snapshot_list

    def load_query(self, triples):
        triples.sort(key=lambda x: (x[3], x[0], x[1]))
        trip_srtau = defaultdict(lambda: list())
        query_split = defaultdict(lambda: list())
        answers_split = defaultdict(lambda: list())
        for trip in triples:
            s, r, o, tau = trip
            trip_srtau[(s, r, tau)].append(o)

        queries = []
        answers = []
        for key in trip_srtau:
            queries.append(key)
            answers.append(np.array(trip_srtau[key]))
            query_split[key[2]].append(np.array(key))
            answers_split[key[2]].append(np.array(trip_srtau[key]))

        return queries, answers, query_split, answers_split

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1), -np.ones((self.n_ent, 1))], 1)
        self.KG = np.concatenate([idd, np.array(triples)], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))

    def get_neighbors(self, nodes, KG, M_sub, mode='train'):
        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)

        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail, tau)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def to_train(self, snapshot_list, batch_size):
        datas = []
        for i, snapshot in enumerate(snapshot_list):
            data = []
            n_snap = len(snapshot) // batch_size + (len(snapshot) % batch_size > 0)
            for idx in range(n_snap):
                start = idx * batch_size
                end = min((idx + 1) * batch_size, len(snapshot))
                data.append(snapshot[start:end])
            datas += data
        datas = np.array(datas)
        return datas

    def to_evaluate(self, batch_size, data):
        if data=='valid':
            queries, answers = self.valid_q_split, self.valid_a_split
        if data=='test':
            queries, answers = self.test_q_split, self.test_a_split

        q_split, a_split = [], []
        for tau in queries.keys():
            query = queries[tau]
            answer = answers[tau]
            q, a = [], []
            assert len(query) == len(answer)
            n_snap = len(query) // batch_size + (len(query) % batch_size > 0)
            for idx in range(n_snap):
                start = idx * batch_size
                end = min((idx + 1) * batch_size, len(query))
                q.append(np.array(query[start:end]))
                a.append(answer[start:end])
            q_split += q
            a_split += a
        return q_split, a_split

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        objs = []
        taus = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        taus = query[batch_idx, 2]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs, taus

    def shuffle_train(self,):
        fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        self.fact_data = self.double_triple(all_triple[:n_all*3//4].tolist())
        self.train_data = np.array(self.double_triple(all_triple[n_all*3//4:].tolist()))
        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)

