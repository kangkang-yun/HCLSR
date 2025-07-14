import pdb
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import manifolds
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import models.encoders as encoders
from utils.helper import default_device
from collections import defaultdict
import pdb

'''
Hyperbolic Contrastive Learning for Social Recommendation
(1) hyperbolic graph learning
(2) personalized transfer
(3) contrastive learning in hyperbolic geometry for social recommendation
'''

######################################   Hyperbolic Model  #########################################
def normal_distribution(x):
    mean = torch.mean(x, dim=1)
    std = torch.std(x, dim=1)
    return (x - mean.unsqueeze(1)) * 0.1 / std.unsqueeze(1)

def normal_dis(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) * 0.1 / std

class HCLSRModel(nn.Module):
    def __init__(self, users_items, args):
        super(HCLSRModel, self).__init__()
        self.args = args
        self.c = torch.tensor([args.c]).to('cuda')
        # self.c = nn.Parameter(torch.tensor([args.c], dtype=torch.float), requires_grad=True).to('cuda') # learnable curve
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.nnodes = args.n_nodes
        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.interest_weight = args.interest_weight
        self.pretrain_type = args.pretrain_type
        self.embedding_dim = args.embedding_dim

        ###  parameters in hyperbolic space  ###
        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=self.embedding_dim).to('cuda')
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)


        ### load pre-trained social embeddings ###
        self._load_pretrained_social_features()

        # user self_gating unit #
        self.gating_weight_ubias = nn.Parameter(
            torch.FloatTensor(1, args.embedding_dim))
        nn.init.xavier_normal_(self.gating_weight_ubias.data)
        self.gating_weight_u = nn.Parameter(
            torch.FloatTensor(args.embedding_dim, args.embedding_dim))
        nn.init.xavier_normal_(self.gating_weight_u.data)

        # GCN
        self.alpha = 0.8
        self.k = 6
        self.mlp  = MLP(self.embedding_dim, self.embedding_dim*self.k, self.embedding_dim//2, self.embedding_dim*self.k)
        self.mlp1 = MLP(self.embedding_dim, self.embedding_dim*self.k, self.embedding_dim//2, self.embedding_dim*self.k)
        self.meta_netu = nn.Linear(self.embedding_dim*3, self.embedding_dim, bias=False)

        # # contrastive negative sampling
        # self.user_social = 5000
        # self.user_SVD = 5000
        # self.item_SVD = 5000



    def _load_pretrained_social_features(self):
        if self.pretrain_type == 'hyperbolic':
            feature_path = './pretrained/' + self.args.dataset + '/hypergnn/' + str(self.args.embedding_dim) + '_dim/'
            user_social_feature = np.load(feature_path + 'H_user_embeddings.npy', allow_pickle=True)
        elif self.pretrain_type == 'euclidean':
            feature_path = './pretrained/' + self.args.dataset + '/euclignn/' + self.args.embedding_dim + '_dim/'
            user_social_feature = np.load(feature_path + 'E_user_embeddings.npy', allow_pickle=True)
        user_social_feature = normal_dis(user_social_feature)
        self.user_social_feature = torch.from_numpy(user_social_feature).to('cuda')


    def manifold_parameters(self, x):
        x.weight = nn.Parameter(self.manifold.expmap0(x.weight, self.c))
        x.weight = manifolds.ManifoldParameter(x.weight, True, self.manifold, self.c)

    def tangent_parameters(self, x):
        y = self.manifold.proj(x.weight, self.c)
        y_tangent = self.manifold.logmap0(y, self.c)
        return y_tangent

    def gating_user(self,em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weight_u)))

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def metafortansform(self, auxiembedu, targetembedu, targetembedi, user_item_mat):

        # Neighbor information of the target node
        user_item_mat_coo = user_item_mat.tocoo()
        user_item_tensor = self.sparse_mx_to_torch_sparse_tensor(user_item_mat_coo).to('cuda')
        uneighbor = torch.matmul(user_item_tensor, targetembedi)

        # Meta-knowlege extraction 公式（5）
        tembedu = (self.meta_netu(torch.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach()))

        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1 = self.mlp(tembedu).reshape(-1, self.embedding_dim, self.k)  # d*k
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, self.embedding_dim)  # k*d
        meta_biasu1 = (torch.mean(metau1, dim=0))
        meta_biasu2 = (torch.mean(metau2, dim=0))
        low_weightu1 = F.softmax(metau1 + meta_biasu1, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu2, dim=1)
        # low_weightu1 = metau1 + meta_biasu1
        # low_weightu2 = metau2 + meta_biasu2

        # The learned matrix as the weights of the transformed network
        tembedus = (torch.sum(torch.multiply((auxiembedu).unsqueeze(-1), low_weightu1), dim=1))
        tembedus = torch.sum(torch.multiply((tembedus).unsqueeze(-1), low_weightu2), dim=1)

        transfuEmbed = tembedus

        return transfuEmbed

    def metaregular(self, em0, em, adj):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:,torch.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[torch.randperm(embedding.shape[0])]
            return corrupted_embedding
        def score(x1, x2):
            x1 = F.normalize(x1, p=2, dim=-1)
            x2 = F.normalize(x2, p=2, dim=-1)
            return torch.sum(torch.multiply(x1, x2), 1)
        user_embeddings = em
        adj = (adj + sp.eye(self.num_users)).tocsr()
        Adj_Norm = torch.from_numpy(np.sum(adj, axis=1)).float().cuda()
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_embeddings = torch.spmm(adj.cuda(), user_embeddings)/Adj_Norm
        user_embeddings = em0
        graph = torch.mean(edge_embeddings, 0)
        pos = score(user_embeddings, graph)
        neg1 = score(row_column_shuffle(user_embeddings), graph)
        # pos = self.manifold.sqdist(user_embeddings, graph, self.c)
        # neg1 = self.manifold.sqdist(row_column_shuffle(user_embeddings), graph, self.c)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos-neg1)))
        # global_loss = torch.sum(-torch.log(torch.sigmoid(neg1 - pos)))
        return global_loss*0.05

    def random_select(self, tensor, k):
        """
        如果张量长度大于 k，则随机筛选到长度为 k；
        否则不做处理，直接返回原张量。

        参数：
        tensor (torch.Tensor): 一维张量
        k (int): 最大长度阈值

        返回：
        torch.Tensor: 处理后的张量
        """
        if tensor.size(0) > k:
            # 随机选择 k 个索引
            indices = torch.randperm(tensor.size(0))[:k]
            return tensor[indices]
        else:
            return tensor

    def hyperbolic_ssl_loss(self, data1, data2, triples):
        ssl_temp = 0.5
        index = torch.unique(torch.tensor(triples[:, 0]))
        # index = self.random_select(index, self.user_social)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        dist_matrix = self.manifold.dist(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), self.c)
        sim_matrix = torch.squeeze(dist_matrix, dim=2)
        sim_matrix = torch.exp(-sim_matrix / ssl_temp)
        pos_sim = sim_matrix[range(len(index)), range(len(index))]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = -torch.log(loss_0)
        loss_1 = -torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = torch.sum(loss) / (len(index))
        del dist_matrix
        return loss*2

    def hyperbolic_ssl_loss_user_SVD(self, emb, emb_SVD, triples):
        ssl_temp = 0.5
        index = torch.unique(torch.tensor(triples[:, 0]))
        # index = self.random_select(index, self.user_SVD)
        embeddings1 = emb[index]
        embeddings2 = emb_SVD[index]
        dist_matrix = self.manifold.dist(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), self.c)
        sim_matrix = torch.squeeze(dist_matrix, dim=2)
        sim_matrix = torch.exp(-sim_matrix / ssl_temp)
        pos_sim = sim_matrix[range(len(index)), range(len(index))]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = -torch.log(loss_0)
        loss_1 = -torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = torch.sum(loss) / (len(index))
        del dist_matrix
        return loss*2

    def hyperbolic_ssl_loss_item_SVD(self, emb, emb_SVD, triples):
        ssl_temp = 0.5
        index = torch.unique(torch.tensor(triples[:, 1]))
        # index = self.random_select(index, self.item_SVD)
        embeddings1 = emb[index]
        embeddings2 = emb_SVD[index]
        dist_matrix = self.manifold.dist(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), self.c)
        sim_matrix = torch.squeeze(dist_matrix, dim=2)
        sim_matrix = torch.exp(-sim_matrix / ssl_temp)
        pos_sim = sim_matrix[range(len(index)), range(len(index))]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = -torch.log(loss_0)
        loss_1 = -torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = torch.sum(loss) / (len(index))
        del dist_matrix
        return loss*2


    def encode(self, adj_uv, adj_uu, adj_trust, user_item_mat, u_mul_s, i_mul_s, svd_users, svd_items):
        adj_uu = adj_uu.to('cuda')
        adj_uv = adj_uv.to('cuda')
        adj_trust = adj_trust.to('cuda')
        u_mul_s = u_mul_s.to('cuda')
        i_mul_s = i_mul_s.to('cuda')
        svd_users = svd_users.to('cuda')
        svd_items = svd_items.to('cuda')

        # 步骤2
        x = self.manifold.proj(self.embedding.weight, self.c)
        x_tangent = self.manifold.logmap0(x, self.c)

        # 步骤3
        user_emb, item_emb = torch.split(x_tangent, [self.num_users, self.num_items])
        user_uuo_emb = self.gating_user(user_emb)
        uio_emb_temp = x_tangent
        user_uuo_emb_temp = user_uuo_emb
        all_user_embeddings = [user_uuo_emb]
        all_ui_embeddings = [x_tangent]

        uio_emb_temp_SVD = x_tangent
        all_ui_embeddings_SVD = [x_tangent]

        # 步骤4
        for i in range(self.num_layers):
            ui_emb0 = torch.spmm(adj_uv, uio_emb_temp)
            uu_emb0 = torch.spmm(adj_trust, user_uuo_emb_temp)

            # 用于SVD增强卷积
            u_emb_temp_SVD, i_emb_temp_SVD = torch.split(uio_emb_temp_SVD, [self.num_users, self.num_items])
            vt_ei = svd_items @ i_emb_temp_SVD
            ut_eu = svd_users @ u_emb_temp_SVD
            u_emb_temp_SVD = 0.8*u_mul_s @ vt_ei + 0.2*uu_emb0
            i_emb_temp_SVD = i_mul_s @ ut_eu
            uio_emb_temp_SVD = torch.cat([u_emb_temp_SVD, i_emb_temp_SVD], 0)

            user_emb0, item_emb0 = torch.split(ui_emb0, [self.num_users, self.num_items])
            user_ed = 0.8*user_emb0 + 0.2*uu_emb0
            user_uuo_emb_temp = 0.8*uu_emb0 + 0.2*user_emb0
            uio_emb_temp = torch.cat([user_ed, item_emb0], 0)

            all_ui_embeddings.append(uio_emb_temp)
            all_user_embeddings.append(user_uuo_emb_temp)
            all_ui_embeddings_SVD.append(uio_emb_temp_SVD)

        # 步骤5
        user_Embedding_aux = sum(all_user_embeddings[1:])
        ui_Embedding = sum(all_ui_embeddings[1:])
        user_Embedding, item_Embedding = torch.split(ui_Embedding, [self.num_users, self.num_items])

        ui_Embedding_SVD = sum(all_ui_embeddings_SVD[1:])

        # 个性化knowledge transfer
        meta_user_emb = self.metafortansform(user_Embedding_aux, user_Embedding, item_Embedding, user_item_mat)
        user_Embedding = self.alpha * user_Embedding + (1 - self.alpha) * (meta_user_emb + user_Embedding_aux)
        user_item_embedding = torch.cat([user_Embedding, item_Embedding], 0)

        # 步骤6
        user_item_embedding = self.manifold.expmap0(user_item_embedding, self.c)
        user_item_embedding = self.manifold.proj(user_item_embedding, self.c)
        uu_social_embedding = self.manifold.expmap0(meta_user_emb + user_Embedding_aux, self.c)
        uu_social_embedding = self.manifold.proj(uu_social_embedding, self.c)
        ui_Embedding_SVD = self.manifold.expmap0(ui_Embedding_SVD, self.c)
        ui_Embedding_SVD = self.manifold.proj(ui_Embedding_SVD, self.c)
        return user_item_embedding, uu_social_embedding, user_Embedding, meta_user_emb + user_Embedding_aux, ui_Embedding_SVD


    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return sqdist


    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]
        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)
        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss


    def compute_loss_adaptive_margin(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]
        false_edges = triples[:, [0, 2]]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores = self.decode(embeddings, false_edges)
        e_u = embeddings[triples[:, 0]]
        e_i = embeddings[triples[:, 1]]
        e_o = torch.zeros(e_u.shape, dtype=torch.float32).to('cuda')
        e_o[:, 0] = 1
        theta = self.manifold.sqdist(e_u, e_o, self.c) + self.manifold.sqdist(e_i, e_o, self.c) \
                - self.manifold.sqdist(e_u, e_i, self.c)
        # theta = torch.clamp(theta, min=1e-9, max=1e9)
        # scale = (e_u[:, 0] * e_i[:, 0]).view(-1, 1)
        # scale = torch.exp(-d_u/(d_u+d_i)).view(-1, 1)
        # theta = theta * scale
        margin = torch.sigmoid(theta)
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss.mean()


    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix


class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim, bias=False)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out =  nn.Linear(feature_dim, output_dim, bias=False)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU().cuda()
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x