#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/04/08 | https://y-research.github.io

"""Description

"""

import torch

from y.util import get_pairwise_stds, get_pairwise_similarity, dist, ANCHOR_ID
from y.l2r_global import torch_one, tensor


def linlog_energy(dist_mat, cls_match_bina_mat, cls_unmatch_bina_mat, margin=None):
    '''
    '''

    if margin is not None: dist_mat = dist_mat + cls_unmatch_bina_mat*margin # impose margin

    #print('dist_mat', dist_mat)
    dist_mat_triu = torch.triu(dist_mat, diagonal=1) # unique pairwise distances

    attraction_mat = dist_mat_triu * cls_match_bina_mat
    #print('attraction_mat', attraction_mat)

    one_mat = torch.ones_like(dist_mat).type(tensor)

    '''
    + torch.tril(one_mat) : ensuring tril() part being zero after log()
    + cls_match_bina_mat  : ensuring pairs of the same class being zero after log()
    '''
    repulsion_mat = torch.log(dist_mat_triu * cls_unmatch_bina_mat + torch.tril(one_mat) + cls_match_bina_mat)
    #print('repulsion_mat', repulsion_mat)

    batch_energy = torch.sum(attraction_mat) - torch.sum(repulsion_mat)
    #print('batch_energy', batch_energy)

    return batch_energy


class LinLogEnergy(torch.nn.Module):
    """

    """
    def __init__(self, anchor_id='Anchor', use_similarity=False, opt=None):
        super(LinLogEnergy, self).__init__()

        self.name = 'LinLogEnergy'
        assert anchor_id in ANCHOR_ID

        self.opt = opt
        self.anchor_id = anchor_id
        self.use_similarity = use_similarity

        #self.k = self.opt.pk
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

        #para_str = '_'.join([para_str, 'K', str(self.k), 'Margin', '{:,g}'.format(self.margin)])
        para_str = '_'.join([para_str, 'Margin', '{:,g}'.format(self.margin)])

        return para_str

    def forward(self, batch_reprs, batch_labels):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_bina_mat = get_pairwise_stds(batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise
        cls_unmatch_bina_mat = torch_one - cls_match_bina_mat

        if self.use_similarity:
            sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
            dist_mat = torch_one - sim_mat
        else:
            dist_mat = dist(batch_reprs=batch_reprs, squared=True)  # [batch_size, batch_size], pairwise distances

        '''
        if 'Class' == self.anchor_id: # vs. anchor wise sorting
            cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
            sim_mat = sim_mat.view(self.num_distinct_cls, -1)
        '''

        batch_loss = linlog_energy(dist_mat=dist_mat, cls_match_bina_mat=cls_match_bina_mat, cls_unmatch_bina_mat=cls_unmatch_bina_mat, margin=self.margin)

        return batch_loss