import torch
import numpy as np
import datetime
import copy

from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.logger import Logger
from torch.autograd import Variable
from sklearn import metrics

#timer = CUDATimer()
log = Logger('evaluation{0}.py.txt'.format(datetime.datetime.now()))

def ranking_and_hits(model, dev_rank_batcher, vocab, name):
    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    mrr_left = []
    mrr_right = []

    hits_left_raw = []
    hits_right_raw = []
    hits_raw = []
    ranks_raw = []
    ranks_left_raw=[]
    ranks_right_raw=[]
    mrr_left_raw=[]
    mrr_right_raw=[]
    rel2ranks = {}
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i in range(10):
        hits_left_raw.append([])
        hits_right_raw.append([])
        hits_raw.append([])
    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()
        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data

        #MY CODE#########
        pred1_raw = pred1.clone()
        pred2_raw = pred2.clone()
        #################
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            # save the prediction that is relevant
            target_value1 = pred1[i,e2[i, 0]]
            target_value2 = pred2[i,e1[i, 0]]
            #MY CODE#################
            target_value1_raw = pred1_raw[i, e2[i, 0]]
            target_value2_raw = pred2_raw[i, e1[i, 0]]
            pred1_raw[i][e2[i]] = target_value1_raw
            pred2_raw[i][e1[i]] = target_value2_raw
            #############################



            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2


        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()

        # MY CODE###########
        max_values_raw, argsort1_raw = torch.sort(pred1_raw, 1, descending=True)
        max_values_raw, argsort2_raw = torch.sort(pred2_raw, 1, descending=True)
        argsort1_raw = argsort1_raw.cpu().numpy()
        argsort2_raw = argsort2_raw.cpu().numpy()
        #################

        for i in range(Config.batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i]==e2[i, 0])[0][0]
            rank2 = np.where(argsort2[i]==e1[i, 0])[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

            ##MY CODE###%%%&&&*************
            rank1_raw = np.where(argsort1_raw[i] == e2[i, 0])[0][0]
            rank2_raw = np.where(argsort2_raw[i] == e1[i, 0])[0][0]
            ranks_raw.append(rank1_raw + 1)
            ranks_left_raw.append(rank1_raw + 1)
            ranks_raw.append(rank2_raw + 1)
            ranks_right_raw.append(rank2_raw + 1)

            for hits_level in range(10):
                if rank1_raw <= hits_level:
                    hits_raw[hits_level].append(1.0)
                    hits_left_raw[hits_level].append(1.0)
                else:
                    hits_raw[hits_level].append(0.0)
                    hits_left_raw[hits_level].append(0.0)

                if rank2_raw <= hits_level:
                    hits_raw[hits_level].append(1.0)
                    hits_right_raw[hits_level].append(1.0)
                else:
                    hits_raw[hits_level].append(0.0)
                    hits_right_raw[hits_level].append(0.0)
            ####******************************

        dev_rank_batcher.state.loss = [0]

    for i in range(9,10):
        #log.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
        #log.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
        log.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    #log.info('Mean rank left: {0}', np.mean(ranks_left))
    #log.info('Mean rank right: {0}', np.mean(ranks_right))
    log.info('MR: {0}', np.mean(ranks))
    #log.info('Mean reciprocal rank left: {0}', np.mean(1./np.array(ranks_left)))
    #log.info('Mean reciprocal rank right: {0}', np.mean(1./np.array(ranks_right)))
    log.info('MRR: {0}', np.mean(1./np.array(ranks)))

    #MY CODE
    #log.info('Raw Hits left @{0}: {1}'.format(10, np.mean(hits_left_raw[9])))
    #log.info('Raw Hits right @{0}: {1}'.format(10, np.mean(hits_right_raw[9])))
    log.info('Raw Hits @{0}: {1}'.format(10, np.mean(hits_raw[9])))
    #log.info('Mean rank left raw: {0}', np.mean(ranks_left_raw))
    #log.info('Mean rank right raw: {0}', np.mean(ranks_right_raw))
    log.info('Raw MR: {0}', np.mean(ranks_raw))
    #log.info('Mean reciprocal rank left raw: {0}', np.mean(1. / np.array(ranks_left_raw)))
    #log.info('Mean reciprocal rank right raw: {0}', np.mean(1. / np.array(ranks_right_raw)))
    log.info('Raw MRR: {0}', np.mean(1. / np.array(ranks_raw)))




