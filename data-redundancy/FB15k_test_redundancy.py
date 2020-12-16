from PIL import Image
import numpy as np
import cPickle
import scipy
import scipy.sparse as sp


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))


"""
path = '../data/FB15k/'
with open(path + 'FB15k_entity2idx.pkl', 'rb') as f:
    entity2idx = cPickle.load(f)
with open(path + 'FB15k_idx2entity.pkl', 'rb') as f:
    idx2entity = cPickle.load(f)
with open('./results/rel2revfb15k_13.pkl', 'rb') as f:
    rel2actualrev = cPickle.load(f)  # reverse
with open('./results/fb15k_same.pkl', 'rb') as f:
    rel2same = cPickle.load(f)  # duplicate
with open('./results/fb15k_rev.pkl', 'rb') as f:
    rel2rev = cPickle.load(f)  # reverse duplicate
# train
inpl = load_file(path + 'FB15k-train-lhs.pkl')
inpr = load_file(path + 'FB15k-train-rhs.pkl')
inpo = load_file(path + 'FB15k-train-rel.pkl')

# test
inpl_test = load_file(path + 'FB15k-test-lhs.pkl')
inpr_test = load_file(path + 'FB15k-test-rhs.pkl')
inpo_test = load_file(path + 'FB15k-test-rel.pkl')

"""
def test_redundancy():
    h, w = 59071, 6  # 59071 test triples exist
    data = np.zeros((h, w))
    g = open('./results/test_redundancies_fb15k.txt', 'w')
    for testid in range(h):
        redundant_pairs, remaining_pairs = [], []
        print testid
        rid, b = np.nonzero(inpo_test[:, testid])
        rid = rid[0]
        head, b = np.nonzero(inpl_test[:, testid])
        tail, b = np.nonzero(inpr_test[:, testid])
        test_pairs = [i for i in zip(head, tail)]
        rel = idx2entity[rid]
        if rel in rel2actualrev:  # check if the reverse of test rel is available
            rev = rel2actualrev[rel]
            rev_id = entity2idx[rev]
            if np.nonzero(inpo_test[rev_id])[1].any() == True:
                ###***1
                # check if the reverse of test triple exist in the test itself
                rel_row, rel_col = np.nonzero(inpo_test[rev_id])
                row, col = np.nonzero(inpl_test[:, rel_col])
                test_heads = row[np.argsort(col)]
                row, col = np.nonzero(inpr_test[:, rel_col])
                test_tails = row[np.argsort(col)]
                test_pairs_rev = [i for i in zip(test_tails, test_heads)]
                revpairs = list(set(test_pairs) & set(test_pairs_rev))
                if len(revpairs) != 0:
                    data[testid, 3] = 4

            # check if the reverse of test triple exist in the training data
            # Train
            rel_row, rel_col = np.nonzero(inpo[rev_id])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs_rev = [i for i in zip(train_tails, train_heads)]
            revpairs = list(set(test_pairs) & set(train_pairs_rev))
            if len(revpairs) != 0:
                data[testid, 0] = 1
                redundant_pairs += revpairs

        if rid in rel2rev:  # check if the reverse duplicate of test_rel is available
            rev_id = rel2rev[rid]

            rel_row, rel_col = np.nonzero(inpo_test[rev_id])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            test_pairs_rev = [i for i in zip(test_tails, test_heads)]
            revpairs = list(set(test_pairs) & set(test_pairs_rev))
            if len(revpairs) != 0:
                data[testid, 4] = 5

            # Train
            rel_row, rel_col = np.nonzero(inpo[rev_id])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs_rev = [i for i in zip(train_tails, train_heads)]
            revpairs = list(set(test_pairs) & set(train_pairs_rev))
            if len(revpairs) != 0:
                data[testid, 1] = 2
                redundant_pairs += revpairs
        if rid in rel2same:  # check if the duplicate of test_rel is available
            same_id = rel2same[rid]
            rel_row, rel_col = np.nonzero(inpo_test[same_id])
            row, col = np.nonzero(inpl_test[:, rel_col])
            test_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr_test[:, rel_col])
            test_tails = row[np.argsort(col)]
            test_pairs2 = [i for i in zip(test_heads, test_tails)]
            samepairs = list(set(test_pairs) & set(test_pairs2))
            if len(samepairs) != 0:
                data[testid, 5] = 6

            # Train
            rel_row, rel_col = np.nonzero(inpo[same_id])
            row, col = np.nonzero(inpl[:, rel_col])
            train_heads = row[np.argsort(col)]
            row, col = np.nonzero(inpr[:, rel_col])
            train_tails = row[np.argsort(col)]
            train_pairs = [i for i in zip(train_heads, train_tails)]
            samepairs = list(set(test_pairs) & set(train_pairs))
            if len(samepairs) != 0:
                data[testid, 2] = 3
                redundant_pairs += samepairs
        redundant_pairs = list(set(redundant_pairs))
        if not len(redundant_pairs):
            remaining_pairs = test_pairs
        for i in redundant_pairs:
            h = idx2entity[i[0]]
            t = idx2entity[i[1]]
            g.write('{}\t{}\t{}\t{}\n'.format(h, rel, t, 1))
        for i in remaining_pairs:
            h = idx2entity[i[0]]
            t = idx2entity[i[1]]
            g.write('{}\t{}\t{}\t{}\n'.format(h, rel, t, 0))
    g.close()
    b = np.unique(data, axis=0)
    print len(b)

    with open('./results/stat_test_matrix.pkl', 'wb') as g:
        cPickle.dump(data, g)

#test_redundancy()
with open('./results/FB15k/stat_test_matrix.pkl', 'rb') as g:
    data = cPickle.load(g)
unq_rows, count = np.unique(data, axis=0, return_counts=True)
out = {tuple(i): j for i, j in zip(unq_rows, count)}
nums = []
for i in out:
    s = ''
    for j in i:
        if j != 0:
            s += '1'
        else:
            s += '0'
    print '{}\t{}'.format(s, out[i])
    nums.append(out[i])
nums.sort()
print nums

# combine some cases
h, w = 59071, 4
data2 = np.zeros((h, w))
for j in range(59071):
    if data[j, 0] == 1:
        data2[j, 0] = 1
    if data[j, 1] == 2:
        data2[j, 1] = 1
    if data[j, 2] == 3:
        data2[j, 1] = 1
    if data[j, 3] == 4:
        data2[j, 2] = 1
    if data[j, 4] == 5:
        data2[j, 3] = 1
    if data[j, 5] == 6:
        data2[j, 3] = 1
unq_rows, count = np.unique(data2, axis=0, return_counts=True)
out = {tuple(i): j for i, j in zip(unq_rows, count)}
nums = []
for i in out:
    s = ''
    for j in i:
        if j != 0:
            s += '1'
        else:
            s += '0'

    print '{}\t{}'.format(s, out[i])
    nums.append(out[i])
nums.sort()
print nums
