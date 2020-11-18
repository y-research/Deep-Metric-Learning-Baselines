import torch
import torch.nn.functional as F
from y.util import get_pairwise_stds, get_pairwise_similarity, dist
from .l2r_global import global_gpu as gpu, tensor
ANCHOR_ID = ['Anchor', 'Class']

class ListNet(torch.nn.Module):

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(ListNet, self).__init__()

        self.name = 'listnet'
        # assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        if 'Class' == anchor_id:
            assert 0 == self.opt.bs % self.opt.samples_per_class
            self.num_distinct_cls = int(self.opt.bs / self.opt.samples_per_class)


    def get_para_str(self):
        para_str = '_'.join(
            [self.name, self.anchor_id, 'Batch', str(self.opt.bs), 'Scls', str(self.opt.samples_per_class)])

        if self.use_similarity:
            para_str = '_'.join([para_str, 'Sim'])

        # else:
        #    if self.squared_dist:
        #        para_str = '_'.join([para_str, 'SqEuDist'])
        #    else:
        #        para_str = '_'.join([para_str, 'EuDist'])

        return para_str


    def forward(self, batch, labels, **kwargs):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch)
        else:
            dist_mat = dist(batch_reprs=batch, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            #cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            #sim_mat = sim_mat.view(self.num_distinct_cls, -1)
            raise NotImplementedError


        # convert to one-dimension vector
        batch_size = batch.size(0)
        index_mat = torch.triu(torch.ones(batch_size, batch_size), diagonal=1) == 1
        sim_vec = sim_mat[index_mat]
        cls_vec = cls_match_mat[index_mat]

        # cross-entropy between two softmaxed vectors
        # batch_loss = -torch.sum(F.softmax(sim_vec) * F.log_softmax(cls_vec))
        batch_loss = -torch.sum(F.softmax(cls_vec) * F.log_softmax(sim_vec))

        return batch_loss


def torch_ideal_dcg(batch_sorted_labels, gpu=False):
    '''
    :param sorted_labels: [batch, ranking_size]
    :return: [batch, 1]
    '''
    batch_gains = torch.pow(2.0, batch_sorted_labels) - 1.0
    batch_ranks = torch.arange(batch_sorted_labels.size(1))

    batch_discounts = torch.log2(2.0 + batch_ranks.type(torch.cuda.FloatTensor)) if gpu else torch.log2(2.0 + batch_ranks.type(torch.FloatTensor))
    batch_ideal_dcg = torch.sum(batch_gains / batch_discounts, dim=1, keepdim=True)

    return batch_ideal_dcg


def get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_stds: the standard labels sorted in a descending order
    :param batch_stds_sorted_via_preds: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    batch_idcgs = torch_ideal_dcg(batch_sorted_labels=batch_stds, gpu=gpu)                      # ideal discount cumulative gains

    batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_stds_sorted_via_preds.size(1)).type(tensor)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg


def lambdarank_loss(batch_preds=None, batch_stds=None, sigma=1.0):
    '''
    This method will impose explicit bias to highly ranked documents that are essentially ties
    :param batch_preds:
    :param batch_stds:
    :return:
    '''

    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)   # sort documents according to the predicted relevance
    batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

    batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
    batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0) # ensuring S_{ij} \in {-1, 0, 1}

    batch_pred_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j

    batch_delta_ndcg = get_delta_ndcg(batch_stds, batch_stds_sorted_via_preds)
    # print('batch_delta_ndcg', batch_delta_ndcg)

    batch_loss_1st = 0.5 * sigma * batch_pred_s_ij * (1.0 - batch_std_Sij) # cf. the 1st equation in page-3
    batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_s_ij) + 1.0)  # cf. the 1st equation in page-3
    # print('batch_loss_1st', batch_loss_1st)
    # print('batch_loss_2nd', batch_loss_2nd)

    # the coefficient of 0.5 is added due to all pairs are used
    batch_loss = torch.sum(0.5 * (batch_loss_1st + batch_loss_2nd) * batch_delta_ndcg)    # weighting with delta-nDCG
    # print('batch loss', batch_loss)

    return batch_loss



class Lambdarank(torch.nn.Module):

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(Lambdarank, self).__init__()

        self.name = 'lambdarank'
        # assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        if 'Class' == anchor_id:
            assert 0 == self.opt.bs % self.opt.samples_per_class
            self.num_distinct_cls = int(self.opt.bs / self.opt.samples_per_class)

    def get_para_str(self):
        para_str = '_'.join(
            [self.name, self.anchor_id, 'Batch', str(self.opt.bs), 'Scls', str(self.opt.samples_per_class)])

        if self.use_similarity:
            para_str = '_'.join([para_str, 'Sim'])

        # else:
        #    if self.squared_dist:
        #        para_str = '_'.join([para_str, 'SqEuDist'])
        #    else:
        #        para_str = '_'.join([para_str, 'EuDist'])

        return para_str


    def forward(self, batch, labels, **kwargs):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''
        # print('batch', batch.grad_fn)
        # print('label', labels)

        cls_match_mat = get_pairwise_stds(batch_labels=labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch)
        else:
            dist_mat = dist(batch_reprs=batch, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        # print('sim mat', sim_mat)
        # print('cls_match mat', cls_match_mat)
        batch_loss = lambdarank_loss(batch_preds=sim_mat, batch_stds=cls_match_mat)

        return batch_loss


def ranknet_loss(batch_preds=None, batch_stds=None, sigma=1.0):
    '''
    This method will impose explicit bias to highly ranked documents that are essentially ties
    :param batch_preds:
    :param batch_stds:
    :return:
    '''

    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)   # sort documents according to the predicted relevance
    batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)    # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

    batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
    batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0) # ensuring S_{ij} \in {-1, 0, 1}

    batch_pred_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j

    batch_loss_1st = 0.5 * sigma * batch_pred_s_ij * (1.0 - batch_std_Sij) # cf. the 1st equation in page-3
    batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_s_ij) + 1.0)  # cf. the 1st equation in page-3
    # print('batch_loss_1st', batch_loss_1st)
    # print('batch_loss_2nd', batch_loss_2nd)

    # the coefficient of 0.5 is added due to all pairs are used
    batch_loss = torch.sum(0.5 * (batch_loss_1st + batch_loss_2nd))    # weighting with delta-nDCG
    # print('batch loss', batch_loss)

    return batch_loss


class RankNet(torch.nn.Module):

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(RankNet, self).__init__()

        self.name = 'RankNet'
        # assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        if 'Class' == anchor_id:
            assert 0 == self.opt.bs % self.opt.samples_per_class
            self.num_distinct_cls = int(self.opt.bs / self.opt.samples_per_class)

    def get_para_str(self):
        para_str = '_'.join(
            [self.name, self.anchor_id, 'Batch', str(self.opt.bs), 'Scls', str(self.opt.samples_per_class)])

        if self.use_similarity:
            para_str = '_'.join([para_str, 'Sim'])

        # else:
        #    if self.squared_dist:
        #        para_str = '_'.join([para_str, 'SqEuDist'])
        #    else:
        #        para_str = '_'.join([para_str, 'EuDist'])

        return para_str


    def forward(self, batch, labels, **kwargs):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch)
        else:
            dist_mat = dist(batch_reprs=batch, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = ranknet_loss(batch_preds=sim_mat, batch_stds=cls_match_mat)

        return batch_loss