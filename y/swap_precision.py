#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/02/13 | https://y-research.github.io

"""Description

"""

import torch

from y.util import get_pairwise_stds, get_pairwise_similarity, dist

ANCHOR_ID = ['Anchor', 'Class']

def swap_precision(simi_mat, cls_match_mat, k=None, margin=None):
    '''
    assuming no-existence of classes with a single instance == samples_per_class > 1
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value
    :param margin:
    :return:
    '''

    simi_mat_hat = simi_mat + (1.0 - cls_match_mat)*margin # impose margin

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1. # todo using desc_indice directly without (+1) to improve efficiency
    #print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)

    ''' get proper K rather than a rigid predefined K
    torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
    It works but is a little confusing at first.
    '''
    #batch_ks = torch.clamp(batch_pos_nums, max=k)
    '''
    due to no explicit self-similarity filtering.
    implicit assumption: a common L2-normalization leading to self-similarity of the maximum one! 
    '''
    batch_ks = torch.clamp(batch_pos_nums, max=k+1)
    k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
    #print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k
    step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k
    
    step-2: determine the size of N, i.e., |N| which determines the size of P
    
    step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    batch_false_pos = (cls_match_mat < 1) & (rank_mat<=k_mat) # torch.uint8 -> used as indice
    #print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1) # used as one/zero
    #print('batch_fp_nums', batch_fp_nums)

    # P
    batch_false_negs = cls_match_mat.bool() & (rank_mat>k_mat) # all false negative

    ''' just for check '''
    #batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    #print('batch_fn_nums', batch_fn_nums)

    # batch_loss = 0
    batch_loss = torch.tensor(0., requires_grad=True).cuda()
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        if fp_num > 0: # error exists, in other words, skip correct case
            #print('fp_num', fp_num)
            all_false_neg = simi_mat_hat[i,:][batch_false_negs[i,:]]
            #print('all_false_neg', all_false_neg)
            top_false_neg, _ = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            #print('top_false_neg', top_false_neg)

            false_pos = simi_mat_hat[i,:][batch_false_pos[i,:]]

            loss = torch.sum(false_pos - top_false_neg)
            batch_loss += loss

    return batch_loss


class SwapPrecision(torch.nn.Module):
    """
    Sampling Wisely: Deep Image Embedding by Top-K Precision Optimization
    Jing Lu, Chaofan Xu, Wei Zhang, Ling-Yu Duan, Tao Mei; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 7961-7970
    """
    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(SwapPrecision, self).__init__()

        self.name = 'SwapPrecision'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin

        if 'Class' == anchor_id:
            assert 0 == self.opt.bs % self.opt.samples_per_class
            self.num_distinct_cls = int(self.opt.bs / self.opt.samples_per_class)

    def get_para_str(self):
        para_str = '_'.join([self.name, self.anchor_id, 'Batch', str(self.opt.bs), 'Scls', str(self.opt.samples_per_class)])

        if self.use_similarity:
            para_str = '_'.join([para_str, 'Sim'])

        #else:
        #    if self.squared_dist:
        #        para_str = '_'.join([para_str, 'SqEuDist'])
        #    else:
        #        para_str = '_'.join([para_str, 'EuDist'])

        para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id: # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = swap_precision(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k, margin=self.margin)

        return batch_loss

#----------------------------------------------------------------------------------------------------------------------#

def rank_aware_swap_precision(simi_mat, cls_match_mat, k=None, margin=None, use_margin=True):
    '''
    assuming no-existence of classes with a single instance
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value, which is an explicit value in order to be consistent with precision
    :param margin:
    :return:
    '''

    if use_margin:
        simi_mat_hat = simi_mat + (1.0 - cls_match_mat)*margin # impose margin
    else:
        simi_mat_hat = simi_mat

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1. # todo using desc_indice directly to improve efficiency
    #print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)

    '''
    get proper K: torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin. It works but is a little confusing at first.
    '''
    #batch_ks = torch.clamp(batch_pos_nums, max=k)
    '''
    due to no explicit self-similarity filtering.
    implicit assumption: a common L2-normalization leading to self-similarity of the maximum one! 
    '''
    batch_ks = torch.clamp(batch_pos_nums, max=k+1)

    k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
    # print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k given the explicit k, where n_{+}>k is due to the explicit k is smaller than the real number of positive neighbors
    
    step-1: determine set of false positive neighbors, i.e., N 
    # cls_match_std is zero && rank<=k
    
    step-2: determine the size of N, i.e., |N| which determines the size of P
    
    step-3: determine set of false negative neighbors, i.e., P, where topk() retrieval is a must when {the explicit k is smaller than the real number of positive neighbors} 
    # cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    batch_false_pos = (cls_match_mat < 1) & (rank_mat<=k_mat) # torch.uint8 -> used as indice
    #print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1) # used as one/zero
    #print('batch_fp_nums', batch_fp_nums)

    # P
    batch_false_negs = cls_match_mat.bool() & (rank_mat>k_mat) # all false negative

    # just for check
    #batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    #print('batch_fn_nums', batch_fn_nums)

    batch_weighted_loss = 0
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        if fp_num > 0: # error exists, in other words, skip correct case
            #print('fp_num', fp_num)
            all_false_neg = simi_mat_hat[i,:][batch_false_negs[i,:]]
            all_false_neg_ranks = rank_mat[i,:][batch_false_negs[i,:]]
            #print('all_false_neg_ranks', all_false_neg_ranks)

            #print('all_false_neg', all_false_neg)
            top_false_neg, top_false_neg_inds = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            #print('top_false_neg', top_false_neg)
            #print('top_false_neg_inds', top_false_neg_inds)
            top_false_neg_ranks = all_false_neg_ranks[top_false_neg_inds]
            #print('top_false_neg_ranks', top_false_neg_ranks)

            false_pos = simi_mat_hat[i,:][batch_false_pos[i,:]]
            false_pos_ranks = rank_mat[i,:][batch_false_pos[i,:]]
            #print('false_pos_ranks', false_pos_ranks)

            #equal# loss = torch.sum(false_pos - top_false_neg)
            #weighted_loss = torch.sum(false_pos*(1.0/false_pos_ranks) - top_false_neg*(1.0/top_false_neg_ranks))

            '''
            (1) requires margin
            (2) needs positive 1.0 coefficient            
            '''

            weighted_loss = torch.sum(false_pos * (1.0 / torch.log2(false_pos_ranks+1.0) + 1.0) - top_false_neg * (1.0 / torch.log2(top_false_neg_ranks+1.0) + 1.0))

            batch_weighted_loss += weighted_loss

    return batch_weighted_loss


class RankAwareSwapPrecision(torch.nn.Module):
    """ """
    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None, use_margin=True):
        super(RankAwareSwapPrecision, self).__init__()

        self.name = 'RankAwareSwapPrecision'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin
        self.use_margin = use_margin

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

        para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id: # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = rank_aware_swap_precision(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k, margin=self.margin, use_margin=self.use_margin)

        return batch_loss


# ----------------------------------------------------------------------------------------------------------------------#

def rank_aware_margin_loss(simi_mat, cls_match_mat, margin=None, use_margin=True):
    '''
    assuming no-existence of classes with a single instance
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value
    :param margin:
    :return:
    '''

    ''' Possible advantages: no explicit k '''

    if use_margin:
        simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * margin  # impose margin
    else:
        simi_mat_hat = simi_mat

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1.  # todo using desc_indice directly to improve efficiency
    # print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)
    k_mat = batch_pos_nums.view(-1, 1).repeat(1, rank_mat.size(1))
    # print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k.

    step-1: determine set of false positive neighbors, i.e., N 
    # cls_match_std is zero && rank<=k

    step-2: determine the size of N, i.e., |N| which determines the size of P

    step-3: determine set of false negative neighbors, i.e., P
    # cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
    # print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero
    # print('batch_fp_nums', batch_fp_nums)

    # P
    batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

    # just for check
    # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    # print('batch_fn_nums', batch_fn_nums)

    batch_weighted_loss = 0
    list_size = cls_match_mat.size(1)
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        k = int(batch_pos_nums.data[i].item())
        if fp_num > 0:  # error exists, in other words, skip correct case
            # print('fp_num', fp_num)
            all_false_neg = simi_mat_hat[i, :][batch_false_negs[i, :]]
            all_false_neg_ranks = rank_mat[i, :][batch_false_negs[i, :]]
            # print('all_false_neg_ranks', all_false_neg_ranks)

            # print('all_false_neg', all_false_neg)
            top_false_neg, top_false_neg_inds = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            # print('top_false_neg', top_false_neg)
            # print('top_false_neg_inds', top_false_neg_inds)
            top_false_neg_ranks = all_false_neg_ranks[top_false_neg_inds]
            # print('top_false_neg_ranks', top_false_neg_ranks)

            false_pos = simi_mat_hat[i, :][batch_false_pos[i, :]]
            false_pos_ranks = rank_mat[i, :][batch_false_pos[i, :]]
            # print('false_pos_ranks', false_pos_ranks)

            # equal# loss = torch.sum(false_pos - top_false_neg)
            # weighted_loss = torch.sum(false_pos*(1.0/false_pos_ranks) - top_false_neg*(1.0/top_false_neg_ranks))

            '''
            (1) requires margin
            (2) needs positive 1.0 coefficient
            (3) distance to the discrimination boundary: the higher the distance is, the larger the weight is            
            '''

            # weighted_loss = torch.sum(false_pos * (1.0 / torch.log2(false_pos_ranks + 1.0) + 1.0) - top_false_neg * (1.0 / torch.log2(top_false_neg_ranks + 1.0) + 1.0))

            #weighted_loss = torch.sum(false_pos * (1.0 + (1.0 - (1.0 / torch.log2(k - false_pos_ranks + 2.0)))) - top_false_neg * (1.0 + (1.0 - 1.0 / torch.log2(top_false_neg_ranks - k + 1.0))))

            #weighted_loss = torch.sum(false_pos * (1.0 + (1.0 - (1.0 / (k - false_pos_ranks + 1.0)))) - top_false_neg * (1.0 + (1.0 - 1.0 / (top_false_neg_ranks - k))))

            #print('false_pos_ranks', false_pos_ranks)
            #print('k - false_pos_ranks', k - false_pos_ranks)
            #print('list_size', list_size)
            #print('top_false_neg_ranks', top_false_neg_ranks)
            #print('top_false_neg_ranks - k', top_false_neg_ranks - k)

            #weighted_loss = torch.sum(false_pos * (0.9 + ((k - false_pos_ranks + 1)/k*.1)) - top_false_neg * (0.9 + ((top_false_neg_ranks - k)/(list_size-k)*0.1)))

            weighted_loss = torch.sum(false_pos * (0.5 + ((k - false_pos_ranks + 1)/k*0.5)) - top_false_neg * (0.5 + ((top_false_neg_ranks - k)/(list_size-k)*0.5)))



            #weighted_loss = torch.sum(false_pos - top_false_neg)

            batch_weighted_loss += weighted_loss

    return batch_weighted_loss


class RankAwareMargin(torch.nn.Module):
    """ """
    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None, use_margin=True):
        super(RankAwareMargin, self).__init__()

        self.name = 'RankAwareMargin'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin
        self.use_margin = use_margin

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

        para_str = '_'.join([para_str, 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id: # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = rank_aware_margin_loss(simi_mat=sim_mat, cls_match_mat=cls_match_mat, margin=self.margin, use_margin=self.use_margin)

        return batch_loss


# ----------------------------------------------------------------------------------------------------------------------#


def beta_minus_origin_swap_precision(simi_mat, cls_match_mat, k=None, margin=None):
    '''
    assuming no-existence of classes with a single instance == samples_per_class > 1
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value
    :param margin:
    :return:
    '''

    simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * margin  # impose margin

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1.  # todo using desc_indice directly without (+1) to improve efficiency
    # print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)

    ''' get proper K rather than a rigid predefined K
    torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
    It works but is a little confusing at first.
    '''
    # batch_ks = torch.clamp(batch_pos_nums, max=k)
    '''
    due to no explicit self-similarity filtering.
    implicit assumption: a common L2-normalization leading to self-similarity of the maximum one! 
    '''
    batch_ks = torch.clamp(batch_pos_nums, max=k + 1)
    k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
    # print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k
    step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k

    step-2: determine the size of N, i.e., |N| which determines the size of P

    step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    #rank_mat = rank_mat.to(dtype=torch.float32)
    batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
    # print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero
    # print('batch_fp_nums', batch_fp_nums)

    # P
    batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

    ''' just for check '''
    # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    # print('batch_fn_nums', batch_fn_nums)

    # batch_loss = 0
    batch_loss = torch.tensor(0., requires_grad=True).cuda()
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        if fp_num > 0:  # error exists, in other words, skip correct case
            all_false_neg = simi_mat_hat[i, :][batch_false_negs[i, :]]
            rank_neg = rank_mat[i, :][batch_false_negs[i, :]]
            top_false_neg, neg_idx = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            rank_top_neg = torch.gather(rank_neg, -1, neg_idx)
            ks = torch.zeros(fp_num).cuda()
            batch_ones = torch.ones_like(ks)
            ks0 = ks.add(k)
            # ks1 = ks.add(k+1)
            ks3 = ks.add(k + 3)
            beta1 = torch.add(batch_ones, -1 / (rank_top_neg - ks0))
            # print('rank top neg', rank_top_neg)
            # print('ks0', ks0)
            # print('beta1', beta1)
            # print('top_false_neg', top_false_neg)
            loss_neg = 3 * torch.dot(beta1, top_false_neg) / fp_num
            # print("loss_neg", loss_neg)

            rank_pos = rank_mat[i, :][batch_false_pos[i, :]]
            false_pos = simi_mat_hat[i, :][batch_false_pos[i, :]]
            # print('fp_num', fp_num)
            # print('ks3', ks3)
            # print('rank_pos', rank_pos)
            # print('batch_ones', batch_ones)
            beta2 = torch.add(batch_ones, -1 / (ks3 - rank_pos))
            # print('beta2', beta2)
            # print('false pos', false_pos)
            # print('false_pos', false_pos)
            loss_pos = 3 * torch.dot(beta2, false_pos) / fp_num
            # print("loss_pos", loss_pos)

            loss = torch.sum(loss_pos - loss_neg)  # /fp_num
            batch_loss += loss

    return batch_loss


class BetaMinusOriginSwapPrecision(torch.nn.Module):
    """
    Sampling Wisely: Deep Image Embedding by Top-K Precision Optimization
    Jing Lu, Chaofan Xu, Wei Zhang, Ling-Yu Duan, Tao Mei; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 7961-7970
    """

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(BetaMinusOriginSwapPrecision, self).__init__()

        self.name = 'beta_minus_origin_SwapPrecision'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin

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

        para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(
            batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = beta_minus_origin_swap_precision(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k, margin=self.margin)

        return batch_loss


# ----------------------------------------------------------------------------------------------------------------------#


def beta_minus_mean_swap_precision(simi_mat, cls_match_mat, k=None, margin=None):
    '''
    assuming no-existence of classes with a single instance == samples_per_class > 1
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value
    :param margin:
    :return:
    '''

    simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * margin  # impose margin

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1.  # todo using desc_indice directly without (+1) to improve efficiency
    # print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)

    ''' get proper K rather than a rigid predefined K
    torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
    It works but is a little confusing at first.
    '''
    # batch_ks = torch.clamp(batch_pos_nums, max=k)
    '''
    due to no explicit self-similarity filtering.
    implicit assumption: a common L2-normalization leading to self-similarity of the maximum one! 
    '''
    batch_ks = torch.clamp(batch_pos_nums, max=k + 1)
    k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
    # print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k
    step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k

    step-2: determine the size of N, i.e., |N| which determines the size of P

    step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    #rank_mat = rank_mat.to(dtype=torch.float32)
    batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
    # print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero
    # print('batch_fp_nums', batch_fp_nums)

    # P
    batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

    ''' just for check '''
    # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    # print('batch_fn_nums', batch_fn_nums)

    # batch_loss = 0
    batch_loss = torch.tensor(0., requires_grad=True).cuda()
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        if fp_num > 0:  # error exists, in other words, skip correct case
            all_false_neg = simi_mat_hat[i, :][batch_false_negs[i, :]]
            rank_neg = rank_mat[i, :][batch_false_negs[i, :]]
            top_false_neg, neg_idx = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            rank_top_neg = torch.gather(rank_neg, -1, neg_idx)
            ks = torch.zeros(fp_num).cuda()
            batch_ones = torch.ones_like(ks)
            ks0 = ks.add(k)
            # ks1 = ks.add(k+1)
            ks3 = ks.add(k + 3)
            beta1 = torch.add(batch_ones, -1 / (rank_top_neg - ks0))
            # print('rank top neg', rank_top_neg)
            # print('ks0', ks0)
            # print('beta1', beta1)
            # print('top_false_neg', top_false_neg)
            loss_neg = torch.dot(beta1, top_false_neg)/fp_num #3 *
            # print("loss_neg", loss_neg)

            rank_pos = rank_mat[i, :][batch_false_pos[i, :]]
            false_pos = simi_mat_hat[i, :][batch_false_pos[i, :]]
            # print('fp_num', fp_num)
            # print('ks3', ks3)
            # print('rank_pos', rank_pos)
            # print('batch_ones', batch_ones)
            beta2 = torch.add(batch_ones, -1 / (ks3 - rank_pos))
            # print('beta2', beta2)
            # print('false pos', false_pos)
            # print('false_pos', false_pos)
            loss_pos = torch.dot(beta2, false_pos)/fp_num # 3 *
            # print("loss_pos", loss_pos)

            loss = torch.sum(loss_pos - loss_neg)#/fp_num
            batch_loss += loss

    return batch_loss


class BetaMinusMeanSwapPrecision(torch.nn.Module):
    """
    Sampling Wisely: Deep Image Embedding by Top-K Precision Optimization
    Jing Lu, Chaofan Xu, Wei Zhang, Ling-Yu Duan, Tao Mei; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 7961-7970
    """

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(BetaMinusMeanSwapPrecision, self).__init__()

        self.name = 'beta_minus_mean_SwapPrecision'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin

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

        para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(
            batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = beta_minus_mean_swap_precision(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k, margin=self.margin)

        return batch_loss



def beta_K_Mean_Sep_swap_precision(simi_mat, cls_match_mat, k=None, margin=None):
    '''
    assuming no-existence of classes with a single instance == samples_per_class > 1
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value
    :param margin:
    :return:
    '''

    simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * margin  # impose margin

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1.  # todo using desc_indice directly without (+1) to improve efficiency
    # print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)

    ''' get proper K rather than a rigid predefined K
    torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
    It works but is a little confusing at first.
    '''
    # batch_ks = torch.clamp(batch_pos_nums, max=k)
    '''
    due to no explicit self-similarity filtering.
    implicit assumption: a common L2-normalization leading to self-similarity of the maximum one! 
    '''
    batch_ks = torch.clamp(batch_pos_nums, max=k + 1)
    k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
    # print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k
    step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k

    step-2: determine the size of N, i.e., |N| which determines the size of P

    step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    #rank_mat = rank_mat.to(dtype=torch.float32) # for pytorch=1.0
    batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
    # print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero
    # print('batch_fp_nums', batch_fp_nums)

    # P
    #batch_false_negs = (cls_match_mat > 0) & (rank_mat > k_mat) # for pytorch=1.0
    batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

    ''' just for check '''
    # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    # print('batch_fn_nums', batch_fn_nums)

    # batch_loss = 0
    batch_loss = torch.tensor(0., requires_grad=True).cuda()
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        if fp_num > 0:  # error exists, in other words, skip correct case
            all_false_neg = rank_mat[i, :][batch_false_negs[i, :]]
            rank_neg = rank_mat[i, :][batch_false_negs[i, :]]
            top_false_neg, neg_idx = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            rank_top_neg = torch.gather(rank_neg, -1, neg_idx)
            ks = torch.zeros(fp_num).cuda()
            # ks0 = ks.add(k)
            ks1 = ks.add(k + 1)
            ks2 = ks.add(k + 2)
            batch_size_mat = ks.add(rank_mat.size(1))
            batch_ones = torch.ones_like(ks)
            beta1 = batch_ones.add(
                (rank_top_neg - ks1) / (batch_size_mat - ks1))  # (rank_top_neg-ks1)/(batch_size_mat-ks0)
            # print('beta 1', beta1)
            beta1sum = beta1.sum()
            loss_neg = torch.dot(top_false_neg, beta1) / beta1sum

            rank_pos = rank_mat[i, :][batch_false_pos[i, :]]
            false_pos = rank_mat[i, :][batch_false_pos[i, :]]
            beta2 = batch_ones.add((ks2 - rank_pos) / ks1)
            beta2sum = beta2.sum()

            loss_pos = torch.dot(false_pos, beta2) / beta2sum
            loss = torch.sum(loss_pos - loss_neg)

            batch_loss += loss

        return batch_loss

    return batch_loss


class BetaKMeanSepSwapPrecision(torch.nn.Module):
    """
    Sampling Wisely: Deep Image Embedding by Top-K Precision Optimization
    Jing Lu, Chaofan Xu, Wei Zhang, Ling-Yu Duan, Tao Mei; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 7961-7970
    """

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(BetaKMeanSepSwapPrecision, self).__init__()

        self.name = 'BetaKMeanSepSwapPrecision'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin

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

        para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(
            batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = beta_K_Mean_Sep_swap_precision(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k,
                                          margin=self.margin)

        return batch_loss



# ----------------------------------------------------------------------------------------------------------------------
def swap_precision_paper(simi_mat, cls_match_mat, k=None, margin=None):
    '''
    assuming no-existence of classes with a single instance == samples_per_class > 1
    :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
    :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
    :param k: cutoff value
    :param margin:
    :return:
    '''

    simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * margin  # impose margin

    ''' get rank positions '''
    _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
    _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
    rank_mat = desc_indice + 1.  # todo using desc_indice directly without (+1) to improve efficiency
    # print('rank_mat', rank_mat)

    # number of true neighbours within the batch
    batch_pos_nums = torch.sum(cls_match_mat, dim=1)

    ''' get proper K rather than a rigid predefined K
    torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
    It works but is a little confusing at first.
    '''
    # batch_ks = torch.clamp(batch_pos_nums, max=k)
    '''
    due to no explicit self-similarity filtering.
    implicit assumption: a common L2-normalization leading to self-similarity of the maximum one! 
    '''
    batch_ks = torch.clamp(batch_pos_nums, max=k + 1)
    k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
    # print('k_mat', k_mat.size())

    '''
    Only deal with a single case: n_{+}>=k
    step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k

    step-2: determine the size of N, i.e., |N| which determines the size of P

    step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
    '''
    # N
    rank_mat = rank_mat.to(dtype=torch.float32) # for pytorch=1.0
    batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
    # print('batch_false_pos', batch_false_pos) bool
    batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero
    # print('batch_fp_nums', batch_fp_nums)

    # P
    batch_false_negs = (cls_match_mat > 0) & (rank_mat > k_mat) # for pytorch=1.0
    #batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

    ''' just for check '''
    # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
    # print('batch_fn_nums', batch_fn_nums)

    batch_loss = 0
    for i in range(cls_match_mat.size(0)):
        fp_num = int(batch_fp_nums.data[i].item())
        if fp_num > 0:  # error exists, in other words, skip correct case
            # print('fp_num', fp_num)
            all_false_neg = simi_mat_hat[i, :][batch_false_negs[i, :]]
            # print('all_false_neg', all_false_neg)
            top_false_neg, _ = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
            top_false_neg = top_false_neg.sum() +2.0/40 * (1 + (40 * (top_false_neg - 0.5)).exp().sum()).log()
            # print('top_false_neg', top_false_neg)

            false_pos = simi_mat_hat[i, :][batch_false_pos[i, :]]
            false_pos = false_pos.sum() +2.0/40 * (1 + (40 * (false_pos - 0.5)).exp().sum()).log()

            loss = torch.sum(false_pos - top_false_neg)
            batch_loss += loss

    return batch_loss


class SwapPrecisionPaper(torch.nn.Module):
    """
    Sampling Wisely: Deep Image Embedding by Top-K Precision Optimization
    Jing Lu, Chaofan Xu, Wei Zhang, Ling-Yu Duan, Tao Mei; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 7961-7970
    """

    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(SwapPrecisionPaper, self).__init__()

        self.name = 'SwapPrecision'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        self.k = self.opt.pk
        self.margin = self.opt.margin

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

        para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(
            batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
            sim_mat = -dist_mat

        if 'Class' == self.anchor_id:  # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)

        batch_loss = swap_precision_paper(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k, margin=self.margin)

        return batch_loss