import sys
import cPickle
from scipy import stats
import numpy as np
import scipy
import scipy.sparse as sp
from itertools import groupby
from collections import defaultdict


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))


def applyrule(test_rel, rules_dir):
    rulesdic = defaultdict(list)
    with open(rules_dir, 'r') as k:
        for rule in k:
            if '?' not in rule.split('\t')[0].split(' ')[10]:
                rel = rule.split('\t')[0].split(' ')[10]
                r = entity2idx[rel]
                rulesdic[r].append(rule)
            elif '/' in rule.split('\t')[0].split(' ')[8] and '/m/' not in rule.split('\t')[0].split(' ')[8]:
                rel = rule.split('\t')[0].split(' ')[16]
                r = entity2idx[rel]
                rulesdic[r].append(rule)

    rel = idx2entity[test_rel]
    if np.nonzero(inpo_test[test_rel])[1].any():
        rid = entity2idx[rel]
        # Test
        rel_row, rel_col = np.nonzero(inpo_test[rid])  # indexes of test cases that contain 'test_rel'
        row, col = np.nonzero(inpl_test[:, rel_col])
        test_head = row[np.argsort(col)]
        heads_name = [idx2entity[e] for e in test_head]
        row, col = np.nonzero(inpr_test[:, rel_col])
        test_tails = row[np.argsort(col)]
        tails_name = [idx2entity[e] for e in test_tails]
        test_pairs = [i for i in zip(heads_name, tails_name)]
        # Train
        rel_row, rel_col = np.nonzero(inpo[rid])  # indexes of train cases that contain 'test_rel'
        row, col = np.nonzero(inpl[:, rel_col])
        train_head = row[np.argsort(col)]
        heads_name = [idx2entity[e] for e in train_head]
        row, col = np.nonzero(inpr[:, rel_col])
        train_tails = row[np.argsort(col)]
        tails_name = [idx2entity[e] for e in train_tails]
        train_pairs = [i for i in zip(heads_name, tails_name)]
        # valid
        rel_row, rel_col = np.nonzero(inpo_valid[rid])  # indexes of valid cases that contain 'test_rel'
        row, col = np.nonzero(inpl_valid[:, rel_col])
        valid_head = row[np.argsort(col)]
        heads_name = [idx2entity[e] for e in valid_head]
        row, col = np.nonzero(inpr_valid[:, rel_col])
        valid_tails = row[np.argsort(col)]
        tails_name = [idx2entity[e] for e in valid_tails]
        valid_pairs = [i for i in zip(heads_name, tails_name)]

        res_pca = []
        res_std = []

        for itr, rules in enumerate(rulesdic[test_rel]):
            ###########################################################################
            ###### check if the len of rule is 2 :?x1 R1  ?x2   => ?x3  R2  ?x4########
            ###########################################################################
            if rel == rules.split('\t')[0].split(' ')[10]:  # ?x1 R1  ?x2   => ?x3  R2  ?x4
                x1 = rules.split('\t')[0].split(' ')[0]
                x2 = rules.split('\t')[0].split(' ')[4]
                x3 = rules.split('\t')[0].split(' ')[8]
                x4 = rules.split('\t')[0].split(' ')[12]
                rel1 = rules.split('\t')[0].split(' ')[2]
                pca_conf = rules.split('\t')[3]  # pca conf
                std_conf = rules.split('\t')[2]  # std conf
                if '/m/' in x2 and '/m/' in x4 and x1 == x3:  # a R1 c1==> a R2 c2
                    rel_idx = entity2idx[rel1]
                    rel_row, rel_col = np.nonzero(inpo[rel_idx])  # indexes of train cases that contain rel1
                    row, head_col = np.nonzero(inpl[:, rel_col])
                    heads1 = row[np.argsort(head_col)]
                    row, tail_col = np.nonzero(inpr[:, rel_col])
                    tails1 = row[np.argsort(tail_col)]
                    x2_id = entity2idx[x2]
                    x1_idx = [i for i, item in enumerate(tails1) if item == x2_id]
                    heads1 = heads1[x1_idx]  # heads1 connected to c1
                    x1_idx = [i for i, item in enumerate(heads1) if
                              item in test_head]  # heads1 that are equal to test head
                    heads1 = heads1[x1_idx]
                    aname = [idx2entity[i] for i in heads1]
                    bname = [x4 for i in range(len(aname))]

                    response_pairs = [i for i in zip(aname, bname)]
                    response_pairs = list(set(response_pairs))
                    for i in response_pairs:
                        res_pca.append((i[0], i[1], pca_conf, itr))
                        res_std.append((i[0], i[1], std_conf, itr))
                if '/m/' in x1 and '/m/' in x4 and x2 == x3:  # c a==> a c
                    rel_idx = entity2idx[rel1]
                    rel_row, rel_col = np.nonzero(inpo[rel_idx])
                    row, head_col = np.nonzero(inpl[:, rel_col])
                    heads1 = row[np.argsort(head_col)]
                    row, tail_col = np.nonzero(inpr[:, rel_col])
                    tails1 = row[np.argsort(tail_col)]
                    x1_id = entity2idx[x1]
                    x1_idx = [i for i, item in enumerate(heads1) if item == x1_id]
                    tails1 = tails1[x1_idx]
                    x1_idx = [i for i, item in enumerate(tails1) if item in test_head]
                    tails1 = tails1[x1_idx]
                    aname = [idx2entity[i] for i in tails1]
                    bname = [x4 for i in range(len(aname))]

                    response_pairs = [i for i in zip(aname, bname)]
                    response_pairs = list(set(response_pairs))
                    for i in response_pairs:
                        res_pca.append((i[0], i[1], pca_conf, itr))
                        res_std.append((i[0], i[1], std_conf, itr))
                if '/m/' in x2 and '/m/' in x3 and x1 == x4:  # b c==> c b
                    x3_id = entity2idx[x3]
                    if x3_id in test_head:
                        rel_idx = entity2idx[rel1]
                        rel_row, rel_col = np.nonzero(inpo[rel_idx])
                        row, head_col = np.nonzero(inpl[:, rel_col])
                        heads1 = row[np.argsort(head_col)]
                        row, tail_col = np.nonzero(inpr[:, rel_col])
                        tails1 = row[np.argsort(tail_col)]
                        x2_id = entity2idx[x2]
                        x1_idx = [i for i, item in enumerate(tails1) if item == x2_id]
                        heads1 = heads1[x1_idx]
                        bname = [idx2entity[i] for i in heads1]
                        aname = [x3 for i in range(len(bname))]

                        response_pairs = [i for i in zip(aname, bname)]
                        response_pairs = list(set(response_pairs))
                        for i in response_pairs:
                            res_pca.append((i[0], i[1], pca_conf, itr))
                            res_std.append((i[0], i[1], std_conf, itr))
                if '/m/' in x1 and '/m/' in x3 and x2 == x4:  # c b==> c b
                    x3_id = entity2idx[x3]
                    if x3_id in test_head:
                        rel_idx = entity2idx[rel1]
                        rel_row, rel_col = np.nonzero(inpo[rel_idx])
                        row, head_col = np.nonzero(inpl[:, rel_col])
                        heads1 = row[np.argsort(head_col)]
                        row, tail_col = np.nonzero(inpr[:, rel_col])
                        tails1 = row[np.argsort(tail_col)]
                        x1_id = entity2idx[x1]
                        x1_idx = [i for i, item in enumerate(heads1) if item == x1_id]
                        tails1 = tails1[x1_idx]
                        bname = [idx2entity[i] for i in tails1]
                        aname = [x3 for i in range(len(bname))]

                        response_pairs = [i for i in zip(aname, bname)]
                        response_pairs = list(set(response_pairs))
                        for i in response_pairs:
                            res_pca.append((i[0], i[1], pca_conf, itr))
                            res_std.append((i[0], i[1], std_conf, itr))
                if x4 == x2 == '?b' and '/m' not in x3:  # ?a R1  ?b   => ?a  R2  ?b
                    rel_idx = entity2idx[rel1]
                    rel_row, rel_col = np.nonzero(inpo[rel_idx])
                    row, head_col = np.nonzero(inpl[:, rel_col])
                    heads1 = row[np.argsort(head_col)]
                    row, tail_col = np.nonzero(inpr[:, rel_col])
                    tails1 = row[np.argsort(tail_col)]
                    x1_idx = [i for i, item in enumerate(heads1) if item in test_head]
                    tails1 = tails1[x1_idx]
                    heads1 = heads1[x1_idx]
                    aname = [idx2entity[i] for i in heads1]
                    bname = [idx2entity[i] for i in tails1]
                    response_pairs = [i for i in zip(aname, bname)]
                    response_pairs = list(set(response_pairs))
                    for i in response_pairs:
                        res_pca.append((i[0], i[1], pca_conf, itr))
                        res_std.append((i[0], i[1], std_conf, itr))
                if x4 == x1 == '?b' and '/m' not in x3:  # ?b R1  ?a   => ?a  R2  ?b
                    rel_idx = entity2idx[rel1]
                    rel_row, rel_col = np.nonzero(inpo[rel_idx])
                    row, head_col = np.nonzero(inpl[:, rel_col])
                    heads1 = row[np.argsort(head_col)]
                    row, tail_col = np.nonzero(inpr[:, rel_col])
                    tails1 = row[np.argsort(tail_col)]
                    x2_idx = [i for i, item in enumerate(tails1) if item in test_head]
                    tails1 = tails1[x2_idx]
                    heads1 = heads1[x2_idx]
                    aname = [idx2entity[i] for i in tails1]
                    bname = [idx2entity[i] for i in heads1]
                    response_pairs = [i for i in zip(aname, bname)]
                    response_pairs = list(set(response_pairs))
                    for i in response_pairs:
                        res_pca.append((i[0], i[1], pca_conf, itr))
                        res_std.append((i[0], i[1], std_conf, itr))
            ##########################################################################################
            if '/' in rules.split('\t')[0].split(' ')[8] and '/m/' not in rules.split('\t')[0].split(' ')[8]:
                if rel == rules.split('\t')[0].split(' ')[16]:
                    x1 = rules.split('\t')[0].split(' ')[0]
                    x2 = rules.split('\t')[0].split(' ')[4]
                    x3 = rules.split('\t')[0].split(' ')[6]
                    x4 = rules.split('\t')[0].split(' ')[10]
                    vars = [x1, x2, x3, x4]
                    x5 = rules.split('\t')[0].split(' ')[14]
                    x6 = rules.split('\t')[0].split(' ')[18]
                    ###############******CONFIDENCE******###################
                    pca_conf = rules.split('\t')[3]
                    std_conf = rules.split('\t')[2]

                    rel1 = rules.split('\t')[0].split(' ')[2]
                    rel2 = rules.split('\t')[0].split(' ')[8]

                    idx_head = [x1, x2, x3, x4].index(x5)
                    idx_tail = [x1, x2, x3, x4].index(x6)

                    ###### ?b R1  ?a , ?b R1  ?a   => ?a  R2  ?b ##########
                    if len({x1, x2, x3, x4}) == 2:
                        if x6 == x1 == x3 and x5 == x2 == x4:
                            rel_idx = entity2idx[rel1]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads1 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails1 = row[np.argsort(tail_col)]
                            x1_idx = [i for i, item in enumerate(tails1) if item in test_head]
                            tails1 = tails1[x1_idx]
                            heads1 = heads1[x1_idx]
                            bname = [idx2entity[i] for i in heads1]
                            aname = [idx2entity[i] for i in tails1]
                            response_pairs1 = [i for i in zip(aname, bname)]
                            rel_idx = entity2idx[rel2]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads2 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails2 = row[np.argsort(tail_col)]
                            x3_idx = [i for i, item in enumerate(tails2) if item in test_head]
                            tails2 = tails2[x3_idx]
                            heads2 = heads2[x3_idx]
                            bname = [idx2entity[i] for i in heads2]
                            aname = [idx2entity[i] for i in tails2]
                            response_pairs2 = [i for i in zip(aname, bname)]
                            response_pairs = list(set(response_pairs1) & set(response_pairs2))
                            # response_pairs = list((Counter(response_pairs1) & Counter(response_pairs2)).elements())
                            for i in response_pairs:
                                res_pca.append((i[0], i[1], pca_conf, itr))
                                res_std.append((i[0], i[1], std_conf, itr))
                        ###### ?b R1  ?a , ?a R1  ?b   => ?a  R2  ?b ##########
                        elif x6 == x1 == x4 and x5 == x2 == x3:
                            rel_idx = entity2idx[rel1]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads1 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails1 = row[np.argsort(tail_col)]
                            x1_idx = [i for i, item in enumerate(tails1) if item in test_head]
                            tails1 = tails1[x1_idx]
                            heads1 = heads1[x1_idx]
                            bname = [idx2entity[i] for i in heads1]
                            aname = [idx2entity[i] for i in tails1]
                            response_pairs1 = [i for i in zip(aname, bname)]
                            rel_idx = entity2idx[rel2]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads2 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails2 = row[np.argsort(tail_col)]
                            x4_idx = [i for i, item in enumerate(heads2) if item in test_head]
                            tails2 = tails2[x4_idx]
                            heads2 = heads2[x4_idx]
                            bname = [idx2entity[i] for i in tails2]
                            aname = [idx2entity[i] for i in heads2]
                            response_pairs2 = [i for i in zip(aname, bname)]
                            response_pairs = list(set(response_pairs1) & set(response_pairs2))
                            # response_pairs = list((Counter(response_pairs1) & Counter(response_pairs2)).elements())
                            for i in response_pairs:
                                res_pca.append((i[0], i[1], pca_conf, itr))
                                res_std.append((i[0], i[1], std_conf, itr))
                        ###### ?a R1  ?b , ?b R1  ?a   => ?a  R2  ?b ##########
                        elif x6 == x2 == x3 and x5 == x1 == x4:
                            rel_idx = entity2idx[rel1]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads1 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails1 = row[np.argsort(tail_col)]
                            x2_idx = [i for i, item in enumerate(heads1) if item in test_head]
                            tails1 = tails1[x2_idx]
                            heads1 = heads1[x2_idx]
                            bname = [idx2entity[i] for i in tails1]
                            aname = [idx2entity[i] for i in heads1]
                            response_pairs1 = [i for i in zip(aname, bname)]
                            rel_idx = entity2idx[rel2]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads2 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails2 = row[np.argsort(tail_col)]
                            x3_idx = [i for i, item in enumerate(tails2) if item in test_head]
                            tails2 = tails2[x3_idx]
                            heads2 = heads2[x3_idx]
                            bname = [idx2entity[i] for i in heads2]
                            aname = [idx2entity[i] for i in tails2]
                            response_pairs2 = [i for i in zip(aname, bname)]
                            response_pairs = list(set(response_pairs1) & set(response_pairs2))
                            # response_pairs = list((Counter(response_pairs1) & Counter(response_pairs2)).elements())
                            for i in response_pairs:
                                res_pca.append((i[0], i[1], pca_conf, itr))
                                res_std.append((i[0], i[1], std_conf, itr))
                        ###### ?a R1  ?b , ?a R1  ?b   => ?a  R2  ?b ##########
                        elif x6 == x2 == x4 and x5 == x1 == x3:
                            rel_idx = entity2idx[rel1]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads1 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails1 = row[np.argsort(tail_col)]
                            x2_idx = [i for i, item in enumerate(heads1) if item in test_head]
                            tails1 = tails1[x2_idx]
                            heads1 = heads1[x2_idx]
                            bname = [idx2entity[i] for i in tails1]
                            aname = [idx2entity[i] for i in heads1]
                            response_pairs1 = [i for i in zip(aname, bname)]
                            rel_idx = entity2idx[rel2]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads2 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails2 = row[np.argsort(tail_col)]
                            x4_idx = [i for i, item in enumerate(heads2) if item in test_head]
                            tails2 = tails2[x4_idx]
                            heads2 = heads2[x4_idx]
                            bname = [idx2entity[i] for i in tails2]
                            aname = [idx2entity[i] for i in heads2]
                            response_pairs2 = [i for i in zip(aname, bname)]
                            response_pairs = list(set(response_pairs1) & set(response_pairs2))
                            # response_pairs = list((Counter(response_pairs1) & Counter(response_pairs2)).elements())
                            for i in response_pairs:
                                res_pca.append((i[0], i[1], pca_conf, itr))
                                res_std.append((i[0], i[1], std_conf, itr))
                        else:
                            print rules

                    if len({x1, x2, x3, x4}) != 2:
                        if idx_head == 0:
                            rel_idx = entity2idx[rel1]
                            rel_row, rel_col = np.nonzero(inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads1 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails1 = row[np.argsort(tail_col)]
                            x1_idx = [i for i, item in enumerate(heads1) if item in test_head]
                            tails1 = tails1[x1_idx]
                            heads1 = heads1[x1_idx]
                            ###### ?a R1  ?f , ?f R1  ?b   => ?a  R2  ?b ##########
                            if x2 == x3 and x4 == x6:
                                rel_idx = entity2idx[rel2]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads2 = row[np.argsort(head_col)]

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails2 = row[np.argsort(tail_col)]

                                x3_id = [i for i, item in enumerate(heads2) if item in tails1]
                                heads2 = heads2[x3_id]
                                tails2 = tails2[x3_id]

                                # intersect
                                intersect = list((set(tails1) & set(heads2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(heads2) if x == c]
                                    b = tails2[ind1]
                                    ind2 = [i for i, x in enumerate(tails1) if x == c]
                                    a = heads1[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                            ###### ?a R1  ?f , ?b R1  ?f   => ?a  R2  ?b ##########
                            if x2 == x4 and x3 == x6:
                                rel_idx = entity2idx[rel2]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads2 = row[np.argsort(head_col)]

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails2 = row[np.argsort(tail_col)]

                                x4_id = [i for i, item in enumerate(tails2) if item in tails1]
                                heads2 = heads2[x4_id]
                                tails2 = tails2[x4_id]

                                intersect = list((set(tails1) & set(tails2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(tails2) if x == c]
                                    b = heads2[ind1]
                                    ind2 = [i for i, x in enumerate(tails1) if x == c]
                                    a = heads1[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                        if idx_head == 1:
                            rel_idx = entity2idx[rel1]
                            rel_row, rel_col = np.nonzero(
                                inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads1 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails1 = row[np.argsort(tail_col)]
                            tails_idx = [i for i, item in enumerate(tails1) if item in test_head]
                            tails1 = tails1[tails_idx]
                            heads1 = heads1[tails_idx]
                            ###### ?f R1  ?a , ?f R1  ?b   => ?a  R2  ?b ##########
                            if x1 == x3 and x4 == x6:
                                rel_idx = entity2idx[rel2]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads2 = row[np.argsort(head_col)]

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails2 = row[np.argsort(tail_col)]

                                x3_id = [i for i, item in enumerate(heads2) if item in heads1]
                                heads2 = heads2[x3_id]
                                tails2 = tails2[x3_id]
                                # intersect
                                intersect = list((set(heads1) & set(heads2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(heads2) if x == c]
                                    b = tails2[ind1]
                                    ind2 = [i for i, x in enumerate(heads1) if x == c]
                                    a = tails1[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                            ###### ?f R1  ?a , ?b R1  ?f   => ?a  R2  ?b ##########
                            if x1 == x4 and x3 == x6:
                                rel_idx = entity2idx[rel2]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads2 = row[np.argsort(head_col)]

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails2 = row[np.argsort(tail_col)]

                                x4_id = [i for i, item in enumerate(tails2) if item in heads1]
                                heads2 = heads2[x4_id]
                                tails2 = tails2[x4_id]

                                # intersect
                                intersect = list((set(heads1) & set(tails2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(tails2) if x == c]
                                    b = heads2[ind1]
                                    ind2 = [i for i, x in enumerate(heads1) if x == c]
                                    a = tails1[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))

                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                        if idx_head == 2:  # X1(heads1)--rel1--X2(tails1), X3(heads2)--rel2--X4(tails2)  h_test--rel3--t_test  head_test==X3

                            rel_idx = entity2idx[rel2]
                            rel_row, rel_col = np.nonzero(
                                inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads2 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails2 = row[np.argsort(tail_col)]
                            heads_idx = [i for i, item in enumerate(heads2) if item in test_head]
                            tails2 = tails2[heads_idx]
                            heads2 = heads2[heads_idx]
                            ###### ?f R1  ?b , ?a R1  ?f   => ?a  R2  ?b ##########
                            if x4 == x1 and x2 == x6:
                                rel_idx = entity2idx[rel1]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails1 = row[np.argsort(tail_col)]

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads1 = row[np.argsort(head_col)]

                                x1_id = [i for i, item in enumerate(heads1) if item in tails2]
                                heads1 = heads1[x1_id]
                                tails1 = tails1[x1_id]

                                intersect = list((set(tails2) & set(heads1)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(heads1) if x == c]
                                    b = tails1[ind1]
                                    ind2 = [i for i, x in enumerate(tails2) if x == c]
                                    a = heads2[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                            ###### ?b R1  ?f , ?a R1  ?f   => ?a  R2  ?b ##########
                            if x4 == x2 and x6 == x1:
                                rel_idx = entity2idx[rel1]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails1 = row[np.argsort(tail_col)]

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads1 = row[np.argsort(head_col)]

                                x2_id = [i for i, item in enumerate(tails1) if item in tails2]
                                heads1 = heads1[x2_id]
                                tails1 = tails1[x2_id]

                                intersect = list((set(tails1) & set(tails2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(tails1) if x == c]
                                    b = heads1[ind1]
                                    ind2 = [i for i, x in enumerate(tails2) if x == c]
                                    a = heads2[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                        if idx_head == 3:
                            rel_idx = entity2idx[rel2]
                            rel_row, rel_col = np.nonzero(
                                inpo[rel_idx])
                            row, head_col = np.nonzero(inpl[:, rel_col])
                            heads2 = row[np.argsort(head_col)]
                            row, tail_col = np.nonzero(inpr[:, rel_col])
                            tails2 = row[np.argsort(tail_col)]
                            tails_idx = [i for i, item in enumerate(tails2) if item in test_head]
                            tails2 = tails2[tails_idx]
                            heads2 = heads2[tails_idx]
                            ###### ?f R1  ?b , ?f R1  ?a   => ?a  R2  ?b ##########
                            if x3 == x1 and x2 == x6:
                                rel_idx = entity2idx[rel1]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails1 = row[np.argsort(tail_col)]

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads1 = row[np.argsort(head_col)]

                                x1_id = [i for i, item in enumerate(heads1) if item in heads2]
                                heads1 = heads1[x1_id]
                                tails1 = tails1[x1_id]

                                intersect = list((set(heads1) & set(heads2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(heads1) if x == c]
                                    b = tails1[ind1]
                                    ind2 = [i for i, x in enumerate(heads2) if x == c]
                                    a = tails2[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))

                            ###### ?b R1  ?f , ?f R1  ?a   => ?a  R2  ?b ##########
                            if x3 == x2 and x6 == x1:
                                rel_idx = entity2idx[rel1]
                                rel_row, rel_col = np.nonzero(inpo[rel_idx])

                                row, tail_col = np.nonzero(inpr[:, rel_col])
                                tails1 = row[np.argsort(tail_col)]

                                row, head_col = np.nonzero(inpl[:, rel_col])
                                heads1 = row[np.argsort(head_col)]

                                x2_id = [i for i, item in enumerate(tails1) if item in heads2]
                                heads1 = heads1[x2_id]
                                tails1 = tails1[x2_id]

                                intersect = list((set(tails1) & set(heads2)))

                                for c in intersect:
                                    ind1 = [i for i, x in enumerate(tails1) if x == c]
                                    b = heads1[ind1]
                                    ind2 = [i for i, x in enumerate(heads2) if x == c]
                                    a = tails2[ind2]
                                    for a0 in a:
                                        for b0 in b:
                                            res_pca.append((idx2entity[a0], idx2entity[b0], pca_conf, itr))
                                            res_std.append((idx2entity[a0], idx2entity[b0], std_conf, itr))
                                res_pca = list(set(res_pca))
                                res_std = list(set(res_std))


        print 'std confidence count'
        evaluate_count(test_rel, rel, res_std, test_pairs, train_pairs, valid_pairs)


def evaluate_count(r, rel, res, test_pairs, train_pairs, valid_pairs):  # predict tail(right)
    res_a = sorted(res, key=lambda x: x[0])
    res_groups = {}
    for key, g in groupby(res_a, lambda x: x[0]):
        res_groups[key] = list(g)

    for tst in test_pairs:
        key_res = tst[0]
        if key_res in res_groups:
            res_test = res_groups[key_res]

            res_count = sorted(res_test, key=lambda x: x[1])
            res_c = {}
            final = []
            for key, g in groupby(res_count, lambda x: x[1]):
                res_c[key] = list(g)
            for ii in res_c:
                count = len(res_c[ii])
                max_conf = max(res_c[ii], key=lambda item: item[2])
                final.append(max_conf + (count,))
            final_res = sorted(final, key=lambda x: (x[2], x[4]), reverse=True)
            final_pca = [(i[2], i[4]) for i in final_res]
            rr = [final_pca.index(i) for i in final_pca]
            ranklist = stats.rankdata(rr, method='dense')
            found_res = [(i[0], i[1]) for i in final_res]
            if tst in found_res:
                res_ind = found_res.index(tst)
                raw_rank = ranklist[res_ind]
                if raw_rank <= 10:
                    rhit10 = 1  # raw hit10
                else:
                    rhit10 = 0
                if raw_rank <= 1:
                    rhit1 = 1
                else:
                    rhit1 = 0
            else:
                raw_rank = 0
            if tst in found_res:
                idx_remove = []
                for i, item in enumerate(found_res):
                    if item != tst:
                        if item in test_pairs:
                            idx_remove.append(i)
                        elif item in train_pairs:
                            idx_remove.append(i)
                        elif item in valid_pairs:
                            idx_remove.append(i)
                final_res_filtered = list(final_res)
                for index in sorted(idx_remove, reverse=True):
                    del final_res_filtered[index]

                final_pca = [(i[2], i[4]) for i in final_res_filtered]
                rr = [final_pca.index(i) for i in final_pca]
                ranklist = stats.rankdata(rr, method='dense')
                found_res = [(i[0], i[1]) for i in final_res_filtered]
                res_ind = found_res.index(tst)
                fres_rank = ranklist[res_ind]
                if fres_rank <= 10:
                    filtered_hit10 = 1  # filtered hit10
                else:
                    filtered_hit10 = 0
                if fres_rank <= 1:
                    fhit1 = 1
                else:
                    fhit1 = 0

                print '{0} {1} {2} {3} {4} {5} {6}'.format(tst[0], rel, tst[1], raw_rank, fres_rank, rhit10, filtered_hit10, rhit1, fhit1)
            else:
                print '{0} {1} {2} {3} {4} {5} {6}'.format(tst[0], rel, tst[1], 0, 0, 0, 0)
        else:
            print '{0} {1} {2} {3} {4} {5} {6}'.format(tst[0], rel, tst[1], 0, 0, 0, 0)




def initialize():
    global entity2idx, idx2entity
    global inpr_test, inpl_test, inpo_test
    global inpr, inpl, inpo
    global inpl_valid, inpr_valid, inpo_valid

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


initialize()
if __name__ == '__main__':
    rules_dir = 'AMIEs-rules/amie-FB15k.txt'
    # FB15k:14951 to 16295  FB15k-237: 14505 to 14741
    # WN18: 40943, 40960    WN18RR: 40943, 40953
    # YAGO3-10; 123182, 123218
    applyrule(int(sys.argv[1]), rules_dir)
