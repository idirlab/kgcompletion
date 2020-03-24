import cPickle
import numpy as np
import scipy
import scipy.sparse as sp
from random import shuffle
import itertools


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))


path = '../data/FB15k/'

with open(path + 'FB15k_entity2idx.pkl', 'rb') as f:
    entity2idx = cPickle.load(f)
with open(path + 'FB15k_idx2entity.pkl', 'rb') as f:
    idx2entity = cPickle.load(f)
# train
inpl = load_file(path + 'FB15k-train-lhs.pkl')
inpr = load_file(path + 'FB15k-train-rhs.pkl')
inpo = load_file(path + 'FB15k-train-rel.pkl')

# test
inpl_test = load_file(path + 'FB15k-test-lhs.pkl')
inpr_test = load_file(path + 'FB15k-test-rhs.pkl')
inpo_test = load_file(path + 'FB15k-test-rel.pkl')

# valid
inpl_valid = load_file(path + 'FB15k-valid-lhs.pkl')
inpr_valid = load_file(path + 'FB15k-valid-rhs.pkl')
inpo_valid = load_file(path + 'FB15k-valid-rel.pkl')


def find_all_cartesian_rels():
    all_cartesian = []
    concat = 0
    for rid in range(14951, 16296):
        # Test
        rel_row, rel_col = np.nonzero(inpo_test[rid])
        row, col = np.nonzero(inpl_test[:, rel_col])
        test_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr_test[:, rel_col])
        test_tails = row[np.argsort(col)]

        # valid
        rel_row, rel_col = np.nonzero(inpo_valid[rid])
        row, col = np.nonzero(inpl_valid[:, rel_col])
        valid_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr_valid[:, rel_col])
        valid_tails = row[np.argsort(col)]

        # Train
        rel_row, rel_col = np.nonzero(inpo[rid])
        row, col = np.nonzero(inpl[:, rel_col])
        train_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr[:, rel_col])
        train_tails = row[np.argsort(col)]

        #
        res_tail = np.concatenate([train_tails, test_tails, valid_tails])
        res_head = np.concatenate([train_heads, test_heads, valid_heads])

        res_tail = list(set(res_tail))
        res_head = list(set(res_head))
        a = len(res_tail) * len(res_head)
        b = len(np.nonzero(inpo[rid])[1])
        b1 = len(np.nonzero(inpo_test[rid])[1])
        b2 = len(np.nonzero(inpo_valid[rid])[1])
        b += b1
        b += b2

        if round(float(b) / a, 2) >= 0.8 and len(np.nonzero(inpo[rid])[1]) > 1:
            all_cartesian.append(rid)
            print idx2entity[rid]
            if './' in idx2entity[rid]:
                concat += 1

    print len(all_cartesian)
    print concat
    return all_cartesian


def cartesian_performance(rid, gt):
    all_entities = range(14951)
    file_gt = './ground_truth/{}.txt'.format(gt)  # ground truth
    rel = idx2entity[rid]
    if np.nonzero(inpo_test[rid])[1].any():

        # Test
        rel_row, rel_col = np.nonzero(inpo_test[rid])
        row, col = np.nonzero(inpl_test[:, rel_col])
        test_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr_test[:, rel_col])
        test_tails = row[np.argsort(col)]
        test_pairs = [i for i in zip(test_heads, test_tails)]

        # Train
        rel_row, rel_col = np.nonzero(inpo[rid])
        row, col = np.nonzero(inpl[:, rel_col])
        train_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr[:, rel_col])
        train_tails = row[np.argsort(col)]
        train_pairs = [i for i in zip(train_heads, train_tails)]

        # valid
        rel_row, rel_col = np.nonzero(inpo_valid[rid])
        row, col = np.nonzero(inpl_valid[:, rel_col])
        valid_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr_valid[:, rel_col])
        valid_tails = row[np.argsort(col)]
        valid_pairs = [i for i in zip(valid_heads, valid_tails)]

        # GROUND TRUTH FROM MAY FREEBASE
        res_gt = []
        hlist, tlist = [], []
        res_head2, res_tail2 = [], []
        with open(file_gt, 'r') as f:
            for lines in f:
                h, r, t = lines[:-1].split('\t')
                hlist.append(h)
                tlist.append(t)
        hlist = list(set(hlist))
        tlist = list(set(tlist))

        for i in hlist:
            if i in entity2idx:
                res_head2.append(entity2idx[i])
        for i in tlist:
            if i in entity2idx:
                res_tail2.append(entity2idx[i])

        for element in itertools.product(res_head2, res_tail2):
            res_gt.append(element)  # ground truth

        rank, frank, frank_gt = 0, 0, 0
        hit10, fhit10, fhit10_gt = 0, 0, 0
        hit1, fhit1, fhit1_gt = 0, 0, 0
        mrr, fmrr, fmrr_gt = 0, 0, 0
        for i in test_pairs:
            for j in range(10):
                res_tail = list(set(train_tails))
                res_head = list(set(train_heads))
                remaining_heads = [x for x in all_entities if x not in res_head]
                shuffle(remaining_heads)
                shuffle(res_head)
                res_head += remaining_heads
                remaining_tails = [x for x in all_entities if x not in res_tail]
                shuffle(remaining_tails)
                shuffle(res_tail)
                res_tail += remaining_tails

                rank_h = res_head.index(i[0]) + 1
                mrr += float(1) / rank_h
                rank_t = res_tail.index(i[1]) + 1
                mrr += float(1) / rank_t
                rank += rank_h + rank_t
                if rank_h <= 10:
                    hit10 += 1
                if rank_t <= 10:
                    hit10 += 1
                if rank_h <= 1:
                    hit1 += 1
                if rank_t <= 1:
                    hit1 += 1

                fres_tail = list(res_tail)
                for j in range(rank_t - 1):
                    if (i[0], res_tail[j]) in test_pairs:
                        fres_tail.remove(res_tail[j])
                    elif (i[0], res_tail[j]) in train_pairs:
                        fres_tail.remove(res_tail[j])
                    elif (i[0], res_tail[j]) in valid_pairs:
                        fres_tail.remove(res_tail[j])

                fres_head = list(res_head)
                for j in range(rank_h - 1):
                    if (res_head[j], i[1]) in test_pairs:
                        fres_head.remove(res_head[j])
                    elif (res_head[j], i[1]) in train_pairs:
                        fres_head.remove(res_head[j])
                    elif (res_head[j], i[1]) in valid_pairs:
                        fres_head.remove(res_head[j])

                frank_h = fres_head.index(i[0]) + 1
                fmrr += float(1) / frank_h
                frank_t = fres_tail.index(i[1]) + 1
                fmrr += float(1) / frank_t
                frank += frank_h + frank_t
                if frank_h <= 10:
                    fhit10 += 1
                if frank_t <= 10:
                    fhit10 += 1
                if frank_h <= 1:
                    fhit1 += 1
                if frank_t <= 1:
                    fhit1 += 1

                for j in range(frank_h - 1):
                    if (res_head[j], i[1]) in res_gt and res_head[j] in fres_head:
                        fres_head.remove(res_head[j])
                for j in range(frank_t - 1):
                    if (i[0], res_tail[j]) in res_gt and res_tail[j] in fres_tail:  #
                        fres_tail.remove(res_tail[j])

                frank_h_gt = fres_head.index(i[0]) + 1
                fmrr_gt += float(1) / frank_h_gt
                frank_t_gt = fres_tail.index(i[1]) + 1
                fmrr_gt += float(1) / frank_t_gt

                frank_gt += frank_h_gt + frank_t_gt
                if frank_h_gt <= 10:
                    fhit10_gt += 1
                if frank_t_gt <= 10:
                    fhit10_gt += 1
                if frank_h_gt <= 1:
                    fhit1_gt += 1
                if frank_t_gt <= 1:
                    fhit1_gt += 1

        print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(rel,
                                                                            round(float(rank) / (20 * len(test_pairs)),
                                                                                  2),
                                                                            round(float(hit10) / (20 * len(test_pairs)),
                                                                                  2) * 100,
                                                                            round(float(hit1) / (20 * len(test_pairs)),
                                                                                  2) * 100,
                                                                            round(float(mrr) / (20 * len(test_pairs)),
                                                                                  2),
                                                                            round(float(frank) / (20 * len(test_pairs)),
                                                                                  2),
                                                                            round(
                                                                                float(fhit10) / (20 * len(test_pairs)),
                                                                                2) * 100,
                                                                            round(float(fhit1) / (20 * len(test_pairs)),
                                                                                  2) * 100,
                                                                            round(float(fmrr) / (20 * len(test_pairs)),
                                                                                  2),
                                                                            round(
                                                                                float(frank_gt) / (
                                                                                        20 * len(test_pairs)),
                                                                                2),
                                                                            round(float(fhit10_gt) / (
                                                                                    20 * len(test_pairs)), 2) * 100,
                                                                            round(
                                                                                float(fhit1_gt) / (
                                                                                        20 * len(test_pairs)),
                                                                                2) * 100,
                                                                            round(
                                                                                float(fmrr_gt) / (20 * len(test_pairs)),
                                                                                2))


all_cartesian_rels = find_all_cartesian_rels()
gt_files = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # ground truth files from Freebase
list_rels = [16181, 15401, 15415, 15436, 15908, 15218, 15601, 15606, 15902]
for rels, gt in zip(list_rels, gt_files):
    cartesian_performance(rels, gt)
