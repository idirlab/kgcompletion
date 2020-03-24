import numpy as np
import scipy
import scipy.sparse as sp
import cPickle

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))

def fb15k_actual_rev_rels():
    path = '../data/FB15k/'
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

    rel2rev = {}
    entitypairs = []
    with open('Freebase13_reverse_rels.out', 'r') as f:
        for lines in f:
            try:
                h, r, t = lines[:-2].split('\t')
                if 'ns:m.' not in lines:
                    h = '/' + h[3:].replace('.', '/')
                    t = '/' + t[3:].replace('.', '/')
                    rel2rev[h] = t
                    rel2rev[t] = h
            except:
                continue

    for rid in range(14951, 16296):
        # Test
        rel_row, rel_col = np.nonzero(inpo_test[rid])
        row, col = np.nonzero(inpl_test[:, rel_col])
        test_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr_test[:, rel_col])
        test_tails = row[np.argsort(col)]
        entitypairs += [i for i in zip(test_heads, test_tails)]

        # Train
        rel_row, rel_col = np.nonzero(inpo[rid])
        row, col = np.nonzero(inpl[:, rel_col])
        train_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr[:, rel_col])
        train_tails = row[np.argsort(col)]
        entitypairs += [i for i in zip(train_heads, train_tails)]

        # valid
        rel_row, rel_col = np.nonzero(inpo_valid[rid])
        row, col = np.nonzero(inpl_valid[:, rel_col])
        valid_heads = row[np.argsort(col)]
        row, col = np.nonzero(inpr_valid[:, rel_col])
        valid_tails = row[np.argsort(col)]
        entitypairs += [i for i in zip(valid_heads, valid_tails)]

    rel2revfb15k_13 = {}

    for rid in range(14951, 16296):
        print rid
        rev_FB = []  # Freebase reverse
        if np.nonzero(inpo[rid])[1].any() == True:
            rel = idx2entity[rid]
            # Train
            rev_rel = []
            rev_pairs = []
            rel_row, rel_col = np.nonzero(inpo[rid])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            rev_train_pairs = [i for i in zip(train_tails, train_heads)]
            rev_pairs += rev_train_pairs

            rel_row, rel_col = np.nonzero(inpo_test[rid])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            rev_test_pairs = [i for i in zip(test_tails, test_heads)]
            rev_pairs += rev_test_pairs

            rel_row, rel_col = np.nonzero(inpo_valid[rid])
            row, col = np.nonzero(inpl_valid[:, rel_col])
            valid_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_valid[:, rel_col])
            valid_tails = row[np.argsort(col)]
            rev_valid_pairs = [i for i in zip(valid_tails, valid_heads)]
            rev_pairs += rev_valid_pairs

            rev = list(set(rev_pairs) & set(entitypairs))
            if len(rev) != 0:
                for j in rev:
                    row, col1 = np.nonzero(inpl[j[0]])
                    row, col2 = np.nonzero(inpr[j[1]])
                    rel_cols = np.intersect1d(col1, col2)
                    row, col = np.nonzero(inpo[:, rel_cols])
                    rev_rel += list(row[np.argsort(col)])

                    row, col1 = np.nonzero(inpl_test[j[0]])
                    row, col2 = np.nonzero(inpr_test[j[1]])
                    rel_cols = np.intersect1d(col1, col2)
                    row, col = np.nonzero(inpo_test[:, rel_cols])
                    rev_rel += list(row[np.argsort(col)])

                    row, col1 = np.nonzero(inpl_valid[j[0]])
                    row, col2 = np.nonzero(inpr_valid[j[1]])
                    rel_cols = np.intersect1d(col1, col2)
                    row, col = np.nonzero(inpo_valid[:, rel_cols])
                    rev_rel += list(row[np.argsort(col)])
                rev_rel = list(set(rev_rel))  # reverse rels found in FB15k for rid
                if '.' in rel:
                    rel1, rel2 = rel.split('.')
                    if rel1 in rel2rev and rel2 in rel2rev:
                        rev_rel1 = rel2rev[rel1]
                        rev_rel2 = rel2rev[rel2]
                        rev_FB.append(rev_rel2 + '.' + rev_rel1)

                elif rel in rel2rev:  # not '.' in rel
                    rev_rel3 = rel2rev[rel]
                    rev_FB.append(rev_rel3)

                for i in rev_rel:  # reverse rels found in FB15k
                    rev_name = idx2entity[i]
                    if rev_name in rev_FB:
                        rel2revfb15k_13[rel] = rev_name
                        rel2revfb15k_13[rev_name] = rel

    with open('./results/FB15k/rel2revfb15k_13.pkl', 'wb') as g:
        cPickle.dump(rel2revfb15k_13, g)

def fb15k_duplicate_rels():
    path = '../data/FB15k/'
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

    fb15k_actual_rev_rels()
    with open('./results/FB15k/rel2revfb15k_13.pkl', 'rb') as f:
        rel2rev = cPickle.load(f)

    doublerev, doublesame = 0, 0
    samepairs, revpairs = 0, 0
    fb15k_same, fb15k_rev = {}, {}
    for rid1 in range(14951, 16296):
        print rid1
        for rid2 in range(rid1 + 1, 16296):
            rel1 = idx2entity[rid1]
            rel2 = idx2entity[rid2]
            if rel1 in rel2rev and rel2rev[rel1] == rel2:
                continue

            # Train1
            rel_row, rel_col = np.nonzero(inpo[rid1])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs1 = [i for i in zip(train_heads, train_tails)]
            # Test1
            rel_row, rel_col = np.nonzero(inpo_test[rid1])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            test_pairs1 = [i for i in zip(test_heads, test_tails)]
            # Valid1
            rel_row, rel_col = np.nonzero(inpo_valid[rid1])
            row, col = np.nonzero(inpl_valid[:, rel_col])
            valid_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_valid[:, rel_col])
            valid_tails = row[np.argsort(col)]
            valid_pairs1 = [i for i in zip(valid_heads, valid_tails)]
            pairs1 = test_pairs1 + train_pairs1 + valid_pairs1

            # Train2
            rel_row, rel_col = np.nonzero(inpo[rid2])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs2 = [i for i in zip(train_heads, train_tails)]
            train_pairs2_rev = [i for i in zip(train_tails, train_heads)]
            # Test2
            rel_row, rel_col = np.nonzero(inpo_test[rid2])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            test_pairs2 = [i for i in zip(test_heads, test_tails)]
            test_pairs2_rev = [i for i in zip(test_tails, test_heads)]
            # Valid2
            rel_row, rel_col = np.nonzero(inpo_valid[rid2])
            row, col = np.nonzero(inpl_valid[:, rel_col])
            valid_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_valid[:, rel_col])
            valid_tails = row[np.argsort(col)]
            valid_pairs2 = [i for i in zip(valid_heads, valid_tails)]
            valid_pairs2_rev = [i for i in zip(valid_tails, valid_heads)]
            pairs2 = test_pairs2 + train_pairs2 + valid_pairs2
            pairs2_rev = test_pairs2_rev + train_pairs2_rev + valid_pairs2_rev

            same = list(set(pairs1) & set(pairs2))  # duplicate pairs
            theta1 = float(len(same)) / len(pairs1)
            theta2 = float(len(same)) / len(pairs2)

            if theta2 > 0.8 and theta1 > 0.8:
                doublesame += 1
                samepairs += len(same)
                fb15k_same[rid1] = rid2
                fb15k_same[rid2] = rid1

            reverse = list(set(pairs1) & set(pairs2_rev))  # reverse duplicate pairs
            theta1 = float(len(reverse)) / len(pairs1)
            theta2 = float(len(reverse)) / len(pairs2)
            if theta2 > 0.8 and theta1 > 0.8:
                doublerev += 1
                revpairs += len(reverse)
                fb15k_rev[rid1] = rid2
                fb15k_rev[rid2] = rid1

    with open('./results/FB15k/FB15k_rev.pkl', 'wb') as g:
        cPickle.dump(fb15k_rev, g)
    with open('./results/FB15k/FB15k_same.pkl', 'wb') as k:
        cPickle.dump(fb15k_same, k)
    print 'pairs of reverse duplicate rels:{}'.format(doublerev)
    print 'pairs of duplicate rels:{}'.format(doublesame)
    print 'pairs of reverse duplicate triples:{}'.format(revpairs)
    print 'pairs of duplicate triples:{}'.format(samepairs)

#*******************************WN18 and YAGO3-10*****************************************
def duplicate_rels(dataset,thresh,relid1,relid2):
    path='../data/{}/'.format(dataset)
    with open(path+'%s_idx2entity.pkl'%dataset, 'rb') as f:
        idx2entity = cPickle.load(f)

    # train
    inpl = load_file(path + '%s-train-lhs.pkl'%dataset)
    inpr = load_file(path + '%s-train-rhs.pkl'%dataset)
    inpo = load_file(path + '%s-train-rel.pkl'%dataset)
    # test
    inpl_test = load_file(path + '%s-test-lhs.pkl'%dataset)
    inpr_test = load_file(path + '%s-test-rhs.pkl'%dataset)
    inpo_test = load_file(path + '%s-test-rel.pkl'%dataset)
    # valid
    inpl_valid = load_file(path + '%s-valid-lhs.pkl'%dataset)
    inpr_valid = load_file(path + '%s-valid-rhs.pkl'%dataset)
    inpo_valid = load_file(path + '%s-valid-rel.pkl'%dataset)
    duplicate_rels = {}
    rev_duplicate_rels = {}
    doublerev, doublesame = 0, 0
    samepairs, revpairs = 0, 0
    for rid1 in range(relid1 ,relid2):#range(40943 ,40961)
        print rid1
        for rid2 in range(rid1, relid2):

            # Train1
            rel_row, rel_col = np.nonzero(inpo[rid1])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs1 = [i for i in zip(train_heads, train_tails)]
            #Test1
            rel_row, rel_col = np.nonzero(inpo_test[rid1])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            test_pairs1 = [i for i in zip(test_heads, test_tails)]
            # Valid1
            rel_row, rel_col = np.nonzero(inpo_valid[rid1])
            row, col = np.nonzero(inpl_valid[:, rel_col])
            valid_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_valid[:, rel_col])
            valid_tails = row[np.argsort(col)]
            valid_pairs1 = [i for i in zip(valid_heads, valid_tails)]

            pairs1 = test_pairs1 + train_pairs1 + valid_pairs1

            # Train2
            rel_row, rel_col = np.nonzero(inpo[rid2])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs2 = [i for i in zip(train_heads, train_tails)]
            train_pairs2_rev = [i for i in zip(train_tails, train_heads)]

            # Test2
            rel_row, rel_col = np.nonzero(inpo_test[rid2])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            test_pairs2 = [i for i in zip(test_heads, test_tails)]
            test_pairs2_rev = [i for i in zip(test_tails, test_heads)]

            # valid2
            rel_row, rel_col = np.nonzero(inpo_valid[rid2])  # indexes of test cases that contain 'test_rel'
            row, col = np.nonzero(inpl_valid[:, rel_col])
            valid_heads = row[np.argsort(col)]  # row numbers of heads
            row, col = np.nonzero(inpr_valid[:, rel_col])
            valid_tails = row[np.argsort(col)]  # row number of cols
            valid_pairs2 = [i for i in zip(valid_heads, valid_tails)]
            valid_pairs2_rev = [i for i in zip(valid_tails, valid_heads)]

            pairs2 = test_pairs2 + train_pairs2 + valid_pairs2
            pairs2_rev = test_pairs2_rev + train_pairs2_rev + valid_pairs2_rev

            same = list(set(pairs1) & set(pairs2))
            theta1 = float(len(same)) / len(pairs1)
            theta2 = float(len(same)) / len(pairs2)

            if theta2 > thresh and theta1 > thresh and rid1 != rid2:
                doublesame += 1
                samepairs += len(same)
                duplicate_rels[rid1] = rid2
                duplicate_rels[rid2] = rid1
                print 'duplicate: {}\t{}\t{}\t{}'.format(idx2entity[rid1], idx2entity[rid2], theta1, theta2)

            reverse = list(set(pairs1) & set(pairs2_rev))
            theta1 = float(len(reverse)) / len(pairs1)
            theta2 = float(len(reverse)) / len(pairs2)

            if theta2 > thresh and theta1 > thresh:
                doublerev += 1
                revpairs += len(reverse)
                rev_duplicate_rels[rid1] = rid2
                rev_duplicate_rels[rid2] = rid1
                print 'reverse duplicate: {}\t{}\t{}\t{}'.format(idx2entity[rid1], idx2entity[rid2], theta1, theta2)

    with open('./results/%s/%s_rev_duplicate.pkl'%(dataset,dataset), 'wb') as g:
        cPickle.dump(rev_duplicate_rels, g)
    with open('./results/%s/%s_duplicate.pkl'%(dataset,dataset), 'wb') as k:
        cPickle.dump(duplicate_rels, k)
    print 'pairs of reverse duplicate rels:{}'.format(doublerev)
    print 'pairs of duplicate rels:{}'.format(doublesame)
    print 'pairs of reverse duplicate triples:{}'.format(revpairs)
    print 'pairs of duplicate triples:{}'.format(samepairs)





#fb15k_duplicate_rels()
duplicate_rels('WN18',0.8, 40943 ,40961)
#duplicate_rels('YAGO3-10',0.7, 123182, 123219)

