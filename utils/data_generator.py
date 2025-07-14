import os
import pickle as pkl
import time
import  pdb

from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from utils.helper import sparse_mx_to_torch_sparse_tensor, normalize
import torch
import multiprocessing as mp


class Data(object):
    def __init__(self, dataset, norm_adj):
        # pkl_path = os.path.join('./data/' + dataset)
        # self.pkl_path = pkl_path
        self.dataset = dataset
        self.norm_adj = norm_adj
        self._dataset_config()
        self._load_data()
        self._create_user_item_adj()
        self._create_user_item_SVD()
        self.uu_mat = self.generate_user_adj()

    def _dataset_config(self):
        if self.dataset == 'flickr':
            self.num_users, self.num_items = 8358, 82120
        elif self.dataset == 'epinions':
            self.num_users, self.num_items = 18202, 47449
        elif self.dataset == 'ciao':
            self.num_users, self.num_items = 7375, 91091
        elif self.dataset == 'dianping':
            self.num_users, self.num_items = 59426, 10224


    def _load_data(self):
        data_path = './data/' + self.dataset + '_data/'
        self.user_users = np.load(data_path + 'user_users_d.npy', allow_pickle=True).tolist()
        self.train_dict = np.load(data_path + 'traindata.npy', allow_pickle=True).tolist()
        self.test_dict = np.load(data_path + 'testdata.npy', allow_pickle=True).tolist()
        self.user_trust = self.user_users
        for key in self.train_dict.keys():
            self.train_dict[key] = list(self.train_dict[key])
        for key in self.test_dict.keys():
            self.test_dict[key] = list(self.test_dict[key])
        for key in self.user_trust.keys():
            self.user_trust[key] = list(self.user_trust[key])


    def _create_user_item_adj(self):
        self.adj_train, user_item = self.generate_adj()
        if eval(self.norm_adj):
            self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))
            self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)
        print('num_users %d, num_items %d' % (self.num_users, self.num_items))
        print('adjacency matrix shape: ', self.adj_train.shape)
        self.user_item_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_users, self.num_items)

    def _create_user_item_SVD(self):
        user_item = np.zeros((self.num_users, self.num_items)).astype(int)
        for i, v in self.train_dict.items():
            user_item[i][v] = 1
        coo_user_item = sp.coo_matrix(user_item).tocsr().astype(np.float32)
        coo_user_item = normalize(coo_user_item)
        coo_user_item = sparse_mx_to_torch_sparse_tensor(coo_user_item)
        svd_users, s, svd_items = torch.svd_lowrank(coo_user_item, q=8)
        self.user_mul_s = svd_users @ (torch.diag(s))
        self.item_mul_s = svd_items @ (torch.diag(s))
        self.svd_users = svd_users
        self.svd_items = svd_items
        del s

    def generate_adj(self):
        user_item = np.zeros((self.num_users, self.num_items)).astype(int)
        for i, v in self.train_dict.items():
            user_item[i][v] = 1
        coo_user_item = sp.coo_matrix(user_item)
        start = time.time()
        print('generating adj csr... ')
        start = time.time()
        rows = np.concatenate((coo_user_item.row, coo_user_item.transpose().row + self.num_users))
        cols = np.concatenate((coo_user_item.col + self.num_users, coo_user_item.transpose().col))
        data = np.ones((coo_user_item.nnz * 2,))
        adj_csr = sp.coo_matrix((data, (rows, cols))).tocsr().astype(np.float32)
        print('time elapsed: {:.3f}'.format(time.time() - start))
        # print('saving adj_csr to ' + self.pkl_path + '/adj_csr.npz')
        # sp.save_npz(self.pkl_path + '/adj_csr.npz', adj_csr)
        # print("time elapsed {:.4f}s".format(time.time() - start))
        return adj_csr, user_item                                   # adj_csr只包含了U-I、I-U部分，user_item为原始的U-I图

    def generate_user_adj(self):
        user_user = np.zeros((self.num_users, self.num_users)).astype(int)
        for i, v in self.user_trust.items():
            user_user[i][v] = 1
        coo_user_user = sp.coo_matrix(user_user)
        start = time.time()
        print('generating user_user adj csr... ')
        start = time.time()
        rows = np.array(coo_user_user.row)
        cols = np.array(coo_user_user.col)
        data = np.ones((coo_user_user.nnz,))
        adj_user_csr = sp.coo_matrix((data, (rows, cols))).tocsr().astype(np.float32)
        print('time elapsed: {:.3f}'.format(time.time() - start))
        # print('saving adj_csr to ' + self.pkl_path + '/adj_csr.npz')
        # sp.save_npz(self.pkl_path + '/adj_csr.npz', adj_csr)
        # print("time elapsed {:.4f}s".format(time.time() - start))
        return adj_user_csr

    def generate_rating_matrix(self, train_set, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
        return rating_matrix


    def lightgcn_adj_matrix(self):
        self.num_node = self.num_users + self.num_items
        self.training_user, self.training_item = [], []
        for u, items in self.train_dict.items():
            for i in items:
                self.training_user.append(u)
                self.training_item.append(i + self.num_users)
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        coo = adj_matrix.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), coo.shape)


    def agcn_adj_matrix(self):
        '''
        refer to SIGIR'20: Joint item recommendation and attribute inference
        '''
        self.num_node = self.num_users + self.num_items
        item_users = defaultdict(list)
        for u, items in self.train_dict.items():
            for i in items:
                item_users[i].append(u)
        adj_indices, adj_values = [], []
        for u, items in self.train_dict.items():
            len_u = len(self.train_dict[u])
            for i in items:
                adj_indices.append([u, i+self.num_users])
                adj_values.append(1.0 / len_u)
        for i, users in item_users.items():
            len_i = len(users)
            for u in users:
                adj_indices.append([i + self.num_users, u])
                adj_values.append(1.0 / len_i)
        adj_indices = np.asarray(adj_indices).T
        adj_values = np.asarray(adj_values)
        graph = torch.sparse.FloatTensor(torch.LongTensor(adj_indices), torch.FloatTensor(adj_values),\
                                         [self.num_node, self.num_node])
        return graph                                             # graph只包含了U-I、I-U部分

    def generate_adj_trust_matrix(self):
        self.num_node = self.num_users
        adj_indices, adj_values = [], []
        for u, users in self.user_trust.items():
            len_u = len(self.user_trust[u])
            for i in users:
                adj_indices.append([u, i])
                adj_values.append(1.0 / len_u)
        adj_indices = np.asarray(adj_indices).T
        adj_values = np.asarray(adj_values)
        graph = torch.sparse.FloatTensor(torch.LongTensor(adj_indices), torch.FloatTensor(adj_values),\
                                         [self.num_node, self.num_node])
        return graph

    def hetero_graph(self):
        num_nodes = self.num_users + self.num_items
        item_users = defaultdict(list)                           # key表示物品编号，value表示与该物品交互过的用户编号
        social_degree = np.zeros((self.num_users,), dtype=int)
        inter_degree = np.zeros((self.num_users,), dtype=int)
        item_degree = np.zeros((self.num_items,), dtype=int)
        for u, items in self.train_dict.items():
            inter_degree[u] = len(items)                         # 用户u交互的物品数量
            for i in items:
                item_users[i].append(u)

        adj_indices, adj_values = [], []
        for u, users in self.user_users.items():
            social_degree[u] = len(users)                        # 用户u交互的用户数量
            len_u = len(users) + 1
            adj_indices.append([u, u])
            adj_values.append(1.0 / len_u)
            for v in users:
                adj_indices.append([u, v])
                adj_values.append(1.0 / len_u)

        for i, users in item_users.items():
            item_degree[i] = len(users)                          # 物品i交互的用户数量
            len_i = len(users) + 1
            adj_indices.append([i + self.num_users, i + self.num_users])
            adj_values.append(1.0 / len_i)
            for u in users:
                adj_indices.append([i + self.num_users, u])
                adj_values.append(1.0 / len_i)
        adj_indices = np.asarray(adj_indices).T
        adj_values = np.asarray(adj_values)
        graph = torch.sparse.FloatTensor(torch.LongTensor(adj_indices), torch.FloatTensor(adj_values),
                                         [num_nodes, num_nodes])
        return graph, social_degree, inter_degree, item_degree   # graph只包含了U-U、I-U以及对角线部分

    def generate_mat(self):
        num_nodes = self.num_users + self.num_items
        UI_distance_mat = self.adj_train
        User_distance_mat = self.uu_mat
        return UI_distance_mat, User_distance_mat, self.user_item_csr

    def split_user_group(self, thredthod_list):
        u1, u2, u3, u4 = [], [], [], []
        for u in self.test_dict.keys():
            if u in self.train_dict.keys():
                items = self.train_dict[u]
            else:
                u1.append(u)
            if len(items) < thredthod_list[0]:
                u1.append(u)
            elif len(items) < thredthod_list[1]:
                u2.append(u)
            elif len(items) < thredthod_list[2]:
                u3.append(u)
            else:
                u4.append(u)
        print('u1 size:', len(u1))
        print('u2 size:', len(u2))
        print('u3 size:', len(u3))
        print('u4 size:', len(u4))
        return u1, u2, u3, u4