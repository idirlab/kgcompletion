import cPickle
import scipy
import numpy as np
import scipy.sparse as sp
from itertools import islice
from collections import Counter


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))


def read_alloutput(input_files, dataset, relidx):
    path = '../data/'
    res_path = './AMIE_res_by_rel/'
    with open(path + '%s_idx2entity.pkl' % dataset, 'rb') as f:
        idx2entity = cPickle.load(f)
    inpo_test = load_file(path + '%s-test-rel.pkl' % dataset)
    stdcount = open(res_path + '%s_std_count-%s.txt' % (input_files, dataset), 'w')
    all_hit10, all_fhit10, all_hit1, all_fhit1 = {}, {}, {}, {}
    for i in range(1, 5):
        all_hit10[i], all_fhit10[i], all_hit1[i], all_fhit1[i] = 0, 0, 0, 0

    for rel in relidx:  ##FB15k:14951 to 16295  FB15k-237: 14505 to 14741
        rel_name = idx2entity[rel]
        j = 0
        test_no = len(np.nonzero(inpo_test[rel])[1])
        if test_no > 0:
            with open('./AMIE_LinkPrediction_results/%s_linkprediction_%s/%s.out' % (input_files, dataset, str(rel)),
                      'r') as f:
                for results in f:
                    nextlines = list(islice(f, test_no))
                    hit10, fhit10, fhit1, hit1 = 0, 0, 0, 0
                    j += 1
                    for lines in nextlines:
                        try:
                            h, r, t, h10, fh10, h1, fh1 = lines.split(' ')
                        except ValueError:
                            print rel
                        fhit10 += int(fh10)  # sum of hit10 of a rel
                        hit10 += int(h10)
                        fhit1 += int(fh1)
                        hit1 += int(h1)
                        all_hit10[j] += int(h10)  # sum of hit10 of all rels
                        all_fhit10[j] += int(fh10)
                        all_hit1[j] += int(h1)
                        all_fhit1[j] += int(fh1)

                    if j == 2:
                        # left/right link prediction results for each relation in the test set
                        stdcount.write('{}\t{}\t{}\n'.format(rel_name, hit10, fhit10))
    stdcount.close()
    return all_hit10, all_fhit10, all_hit1, all_fhit1


def combine_leftright(dataset):
    path = '../data/'
    with open(path + '%s_entity2idx.pkl' % dataset, 'rb') as f:
        entity2idx = cPickle.load(f)
    res_path = './AMIE_res_by_rel/'
    f = open(res_path + 'std_count_%s.txt' % dataset, 'w')  # link prediction results for each relation in the test set
    inpo_test = load_file(path + '%s-test-rel.pkl' % dataset)
    with open(res_path + 'left_std_count-%s.txt' % dataset, 'r') as leftres, open(
            res_path + 'right_std_count-%s.txt' % dataset, 'r') as rightres:
        for lines in zip(leftres, rightres):
            rel, left_hit10, left_fhit10 = lines[0][:-1].split('\t')
            rel, right_hit10, right_fhit10 = lines[1][:-1].split('\t')
            test_no = len(np.nonzero(inpo_test[entity2idx[rel]])[1])
            hit10 = round((float(left_hit10) + float(right_hit10)) / (2 * test_no) * 100, 2)
            fhit10 = round((float(left_fhit10) + float(right_fhit10)) / (2 * test_no) * 100, 2)
            f.write('{}\t{}\t{}\t{}\n'.format(rel, test_no, hit10, fhit10))


if __name__ == '__main__':
    # FB15k237 results
    numtest = 20466
    relidx = range(14505, 14742)
    left_hit10, left_fhit10, left_hit1, left_fhit1 = read_alloutput('left', 'FB15k237', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1 = read_alloutput('right', 'FB15k237', relidx)
    fhit10 = (float(left_fhit10[2]) + float(right_fhit10[2])) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1[2]) + float(right_fhit1[2])) / (2 * numtest) * 100
    print 'FHits@10 of FB15k237: {}'.format(round(fhit10, 2))
    print 'FHits@1 of FB15k237: {}'.format(round(fhit1, 2))
    # FB15k results
    numtest = 59071
    relidx = range(14951, 16295)
    left_hit10, left_fhit10, left_hit1, left_fhit1 = read_alloutput('left', 'FB15k', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1 = read_alloutput('right', 'FB15k', relidx)
    fhit10 = (float(left_fhit10[2]) + float(right_fhit10[2])) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1[2]) + float(right_fhit1[2])) / (2 * numtest) * 100
    print 'FHits@10 of FB15k: {}'.format(round(fhit10, 2))
    print 'FHits@1 of FB15k: {}'.format(round(fhit1, 2))
    combine_leftright('FB15k237')
    combine_leftright('FB15k')
