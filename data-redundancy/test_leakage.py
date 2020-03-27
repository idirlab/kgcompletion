import numpy as np
import scipy
import scipy.sparse as sp
import cPickle

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))

def test_redundancy(dataset, rel_range):
    path='../data/{}/'.format(dataset)
    with open(path+'%s_entity2idx.pkl'%dataset, 'rb') as f:
        entity2idx = cPickle.load(f)
    with open(path+'%s_idx2entity.pkl'%dataset, 'rb') as f:
        idx2entity = cPickle.load(f)

    with open('./results/%s/%s_duplicate.pkl'%(dataset,dataset), 'rb') as f:
        rel2same = cPickle.load(f)
    with open('./results/%s/%s_rev_duplicate.pkl'%(dataset,dataset), 'rb') as f:
        rel2rev = cPickle.load(f)


    # train
    inpl = load_file(path + '%s-train-lhs.pkl'%dataset)
    inpr = load_file(path + '%s-train-rhs.pkl'%dataset)
    inpo = load_file(path + '%s-train-rel.pkl'%dataset)

    # test
    inpl_test = load_file(path + '%s-test-lhs.pkl'%dataset)
    inpr_test = load_file(path + '%s-test-rhs.pkl'%dataset)
    inpo_test = load_file(path + '%s-test-rel.pkl'%dataset)


    g=open('./results/%s/test-redundancies.txt'% dataset,'w')
    for rid in rel_range:#range(40943, 40961)
        if np.nonzero(inpo_test[rid])[1].any() == True:
            redundant_pairs = []
            rel = idx2entity[rid]
            # Test
            rel_row, rel_col = np.nonzero(inpo_test[rid])  # indexes of test cases that contain rid
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]  # row numbers of heads
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]  # row number of cols
            test_pairs = [i for i in zip(test_heads, test_tails)]

            if rid in rel2same:  #both actual rev and same
                same_id = rel2same[rid]
                # Train
                rel_row, rel_col = np.nonzero(inpo[same_id])  # indexes of test cases that contain 'test_rel'
                row, col = np.nonzero(inpl[:, rel_col])
                train_heads = row[np.argsort(col)]  # row numbers of heads
                row, col = np.nonzero(inpr[:, rel_col])
                train_tails = row[np.argsort(col)]  # row number of cols
                train_pairs = [i for i in zip(train_heads, train_tails)]
                samepairs = list(set(test_pairs) & set(train_pairs))
                if len(samepairs) != 0:
                    redundant_pairs += samepairs
            if rid in rel2rev:  # actual and rev redundant
                rev_id = rel2rev[rid]
                # Train
                rel_row, rel_col = np.nonzero(inpo[rev_id])  # indexes of test cases that contain 'test_rel'
                row, col = np.nonzero(inpl[:, rel_col])
                train_heads = row[np.argsort(col)]  # row numbers of heads
                row, col = np.nonzero(inpr[:, rel_col])
                train_tails = row[np.argsort(col)]  # row number of cols
                train_pairs_rev = [i for i in zip(train_tails, train_heads)]
                revpairs = list(set(test_pairs) & set(train_pairs_rev))
                if len(revpairs) != 0:
                    redundant_pairs += revpairs

            redundant_pairs=list(set(redundant_pairs))
            for i in redundant_pairs:
                h=idx2entity[i[0]]
                t = idx2entity[i[1]]
                g.write('{}\t{}\t{}\t{}\n'.format(h,rel,t,1))

            remaining_pairs=[x for x in test_pairs if x not in redundant_pairs]
            for i in remaining_pairs:
                h = idx2entity[i[0]]
                t = idx2entity[i[1]]
                g.write('{}\t{}\t{}\t{}\n'.format(h, rel, t, 0))


    g.close()
test_redundancy('WN18',range(40943 ,40961))
test_redundancy('YAGO3-10',range(123182, 123219))