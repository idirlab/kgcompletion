import cPickle
import scipy
import numpy as np
import scipy.sparse as sp
from itertools import islice
from collections import Counter
from random import shuffle


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))


def read_alloutput(input_files, dataset, relidx):
    path = '../data/'
    with open(path + '%s/%s_idx2entity.pkl' % (dataset,dataset), 'rb') as f:
        idx2entity = cPickle.load(f)

    # train
    inpl = load_file(path + '%s/%s-train-lhs.pkl'%(dataset,dataset))
    inpr = load_file(path + '%s/%s-train-rhs.pkl'%(dataset,dataset))
    inpo = load_file(path + '%s/%s-train-rel.pkl'%(dataset,dataset))

    # test
    inpl_test = load_file(path + '%s/%s-test-lhs.pkl'%(dataset,dataset))
    inpr_test = load_file(path + '%s/%s-test-rhs.pkl'%(dataset,dataset))
    inpo_test = load_file(path + '%s/%s-test-rel.pkl'%(dataset,dataset))

    # valid
    inpl_valid = load_file(path + '%s/%s-valid-lhs.pkl'%(dataset,dataset))
    inpr_valid = load_file(path + '%s/%s-valid-rhs.pkl'%(dataset,dataset))
    inpo_valid = load_file(path + '%s/%s-valid-rel.pkl'%(dataset,dataset))



    res_path = './AMIE_res_by_rel/'

    all_entities=[]
    with open('../datasets/%s/entities.txt'%dataset,'r') as f:
        for lines in f:
            all_entities.append(lines[:-1])

    all_hit10, all_fhit10, all_hit1, all_fhit1, all_rank, all_frank , all_rrank, all_frrank = 0, 0, 0, 0, 0, 0, 0, 0
    output = open(res_path + './res_by_rel_%s_%s.txt' % (input_files, dataset), 'w')
    for rel in relidx:
        #print rel
        rel_name = idx2entity[rel]
        test_no = len(np.nonzero(inpo_test[rel])[1])
        rid = rel
        #print rel
        #print test_no
        # Test
        rel_row, rel_col = np.nonzero(inpo_test[rid])
        row, col = np.nonzero(inpl_test[:, rel_col])
        test_head = row[np.argsort(col)]
        heads_name = [idx2entity[e] for e in test_head]
        row, col = np.nonzero(inpr_test[:, rel_col])
        test_tails = row[np.argsort(col)]
        tails_name = [idx2entity[e] for e in test_tails]
        test_pairs = [i for i in zip(heads_name, tails_name)]
        # Train
        rel_row, rel_col = np.nonzero(inpo[rid])
        row, col = np.nonzero(inpl[:, rel_col])
        train_head = row[np.argsort(col)]
        heads_name = [idx2entity[e] for e in train_head]
        row, col = np.nonzero(inpr[:, rel_col])
        train_tails = row[np.argsort(col)]
        tails_name = [idx2entity[e] for e in train_tails]
        train_pairs = [i for i in zip(heads_name, tails_name)]
        # valid
        rel_row, rel_col = np.nonzero(inpo_valid[rid])
        row, col = np.nonzero(inpl_valid[:, rel_col])
        valid_head = row[np.argsort(col)]
        heads_name = [idx2entity[e] for e in valid_head]
        row, col = np.nonzero(inpr_valid[:, rel_col])
        valid_tails = row[np.argsort(col)]
        tails_name = [idx2entity[e] for e in valid_tails]
        valid_pairs = [i for i in zip(heads_name, tails_name)]
        if test_no > 0:
            with open('./AMIE_LinkPrediction_results/%s/%s/%s.out' % (dataset, input_files, str(rel)),
                      'r') as f:
                test_fhit10, test_frank, test_frrank, test_fhit1 = 0, 0, 0, 0
                test_hit10, test_rank, test_rrank, test_hit1 = 0, 0, 0, 0
                for lines in f:

                    try:
                        h, r, t, ra,fra,h10, fh10, h1, fh1 = lines.split(' ')

                        all_hit10 += int(h10)  # sum of hit10 of all rels
                        all_fhit10 += int(fh10)
                        all_hit1 += int(h1)
                        all_fhit1 += int(fh1)
                        all_rank += int(ra)
                        all_frank += int(fra)
                        all_rrank += np.reciprocal(float(ra))
                        all_frrank += np.reciprocal(float(fra))

                        test_frank += int(fra)
                        test_fhit10 += int(fh10)
                        test_frrank += np.reciprocal(float(fra))
                        test_fhit1 += int(fh1)

                        test_rank += int(ra)
                        test_hit10 += int(h10)
                        test_rrank += np.reciprocal(float(ra))
                        test_hit1 += int(h1)


                    except ValueError:

                        try:
                            h, r, t, h10, fh10, h1, fh1 = lines.split(' ')

                            shuffle(all_entities)
                            if input_files=='left':
                                random_res = [(i, t) for i in all_entities]
                            if input_files=='right':
                                random_res = [(h, i) for i in all_entities]

                            ra = random_res.index((h,t)) +1

                            one=set(random_res[:ra-1]) & set(test_pairs)
                            two=set(random_res[:ra-1]) & set(train_pairs)
                            three=set(random_res[:ra-1]) & set(valid_pairs)
                            fra= ra-len(one)-len(two)-len(three)


                            all_hit10 += int(h10)  # sum of hit10 of all rels
                            all_fhit10 += int(fh10)
                            all_hit1 += int(h1)
                            all_fhit1 += int(fh1)
                            all_rank += int(ra)
                            all_frank += int(fra)
                            all_rrank += np.reciprocal(float(ra))
                            all_frrank += np.reciprocal(float(fra))

                            test_frank += int(fra)
                            test_fhit10 += int(fh10)
                            test_frrank += np.reciprocal(float(fra))
                            test_fhit1 += int(fh1)

                            test_rank += int(ra)
                            test_hit10 += int(h10)
                            test_rrank += np.reciprocal(float(ra))
                            test_hit1 += int(h1)
                        except:
                            print rel

            output.write(
                        '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(rel_name, test_no, test_rank, test_hit10,
                                                                          test_hit1, test_rrank, test_frank,
                                                                          test_fhit10, test_fhit1, test_frrank))


    return all_hit10, all_fhit10, all_hit1, all_fhit1, all_rank, all_frank, all_rrank, all_frrank


def combine_leftright(dataset):
    path = '../data/'
    with open(path + '%s/%s_entity2idx.pkl' % (dataset,dataset), 'rb') as f:
        entity2idx = cPickle.load(f)
    res_path = './AMIE_res_by_rel/'
    f = open(res_path + 'res_by_rel_%s.txt' %dataset , 'w')  # link prediction results for each relation in the test set
    inpo_test = load_file(path + '%s/%s-test-rel.pkl' % (dataset,dataset))
    with open(res_path + 'res_by_rel_left_%s.txt' %dataset , 'r') as leftres, open(
            res_path + 'res_by_rel_right_%s.txt' %dataset , 'r') as rightres:
        for lines in zip(leftres, rightres):
            rel, t_no1,left_ra,left_h10,left_h1,left_rr,left_fra,left_fh10,left_fh1,left_frr = lines[0][:-1].split('\t')
            rel,t_no2, right_ra, right_h10,right_h1, right_rr,right_fra, right_fh10, right_fh1,right_frr = lines[1][:-1].split('\t')
            test_no = len(np.nonzero(inpo_test[entity2idx[rel]])[1])
            if test_no!=int(t_no1):
                print test_no
            if test_no !=int(t_no2):
                print test_no
            hit10 = round((float(left_h10) + float(right_h10)) / (2 * test_no) * 100, 2)
            hit1 = round((float(left_h1) + float(right_h1)) / (2 * test_no) * 100, 2)
            mr = round((float(left_ra) + float(right_ra)) / (2 * test_no), 2)
            mrr = round((float(left_rr) + float(right_rr)) / (2 * test_no), 3)
            fhit10 = round((float(left_fh10) + float(right_fh10)) / (2 * test_no) * 100, 2)
            fhit1 = round((float(left_fh1) + float(right_fh1)) / (2 * test_no) * 100, 2)
            fmr = round((float(left_fra) + float(right_fra)) / (2 * test_no), 2)
            fmrr = round((float(left_frr) + float(right_frr)) / (2 * test_no), 3)
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(rel, test_no,mr, hit10,hit1,mrr, fmr, fhit10,fhit1,fmrr))

if __name__ == '__main__':

    # FB15k237 results
    numtest = 20466
    relidx = range(14505, 14742)
    left_hit10, left_fhit10, left_hit1, left_fhit1,left_rank, left_frank, left_rrank, left_frrank = read_alloutput('left', 'FB15k-237', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1,right_rank, right_frank, right_rrank, right_frrank = read_alloutput('right', 'FB15k-237', relidx)
    fhit10 = (float(left_fhit10) + float(right_fhit10)) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1) + float(right_fhit1)) / (2 * numtest) * 100
    hit10 = (float(left_hit10) + float(right_hit10)) / (2 * numtest) * 100
    hit1 = (float(left_hit1) + float(right_hit1)) / (2 * numtest) * 100
    rank = (float(left_rank) + float(right_rank)) / (2 * numtest)
    frank = (float(left_frank) + float(right_frank)) / (2 * numtest)
    rrank = (float(left_rrank) + float(right_rrank)) / (2 * numtest)
    frrank = (float(left_frrank) + float(right_frrank)) / (2 * numtest)

    print '------Raw Results of FB15k-237-------'
    print 'MR: {}'.format(round(rank,2))
    print 'Hits@10: {}'.format(round(hit10,2))
    print 'Hits@1: {}'.format(round(hit1, 2))
    print 'MRR: {}'.format(round(rrank,3))

    print '------Filtered Results of FB15k-237-------'
    print 'FMR: {}'.format(round(frank,2))
    print 'FHits@10: {}'.format(round(fhit10, 2))
    print 'FHits@1: {}'.format(round(fhit1, 2))
    print 'FMRR: {}'.format(round(frrank,3))



    #FB15K
    numtest = 59071
    relidx = range(14951, 16295)
    left_hit10, left_fhit10, left_hit1, left_fhit1,left_rank, left_frank, left_rrank, left_frrank = read_alloutput('left', 'FB15k', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1,right_rank, right_frank, right_rrank, right_frrank = read_alloutput('right', 'FB15k', relidx)
    fhit10 = (float(left_fhit10) + float(right_fhit10)) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1) + float(right_fhit1)) / (2 * numtest) * 100
    hit10 = (float(left_hit10) + float(right_hit10)) / (2 * numtest) * 100
    hit1 = (float(left_hit1) + float(right_hit1)) / (2 * numtest) * 100
    rank = (float(left_rank) + float(right_rank)) / (2 * numtest)
    frank = (float(left_frank) + float(right_frank)) / (2 * numtest)
    rrank = (float(left_rrank) + float(right_rrank)) / (2 * numtest)
    frrank = (float(left_frrank) + float(right_frrank)) / (2 * numtest)

    print '------Raw Results of FB15k-------'
    print 'MR: {}'.format(round(rank, 2))
    print 'Hits@10: {}'.format(round(hit10, 2))
    print 'Hits@1: {}'.format(round(hit1, 2))
    print 'MRR: {}'.format(round(rrank, 3))

    print '------Filtered Results of FB15k-------'
    print 'FMR: {}'.format(round(frank, 2))
    print 'FHits@10: {}'.format(round(fhit10, 2))
    print 'FHits@1: {}'.format(round(fhit1, 2))
    print 'FMRR: {}'.format(round(frrank, 3))

    #WN18
    numtest = 5000
    relidx = range(40943, 40961)
    left_hit10, left_fhit10, left_hit1, left_fhit1, left_rank, left_frank, left_rrank, left_frrank = read_alloutput(
        'left', 'WN18', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1, right_rank, right_frank, right_rrank, right_frrank = read_alloutput(
        'right', 'WN18', relidx)
    fhit10 = (float(left_fhit10) + float(right_fhit10)) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1) + float(right_fhit1)) / (2 * numtest) * 100
    hit10 = (float(left_hit10) + float(right_hit10)) / (2 * numtest) * 100
    hit1 = (float(left_hit1) + float(right_hit1)) / (2 * numtest) * 100
    rank = (float(left_rank) + float(right_rank)) / (2 * numtest)
    frank = (float(left_frank) + float(right_frank)) / (2 * numtest)
    rrank = (float(left_rrank) + float(right_rrank)) / (2 * numtest)
    frrank = (float(left_frrank) + float(right_frrank)) / (2 * numtest)

    print '------Raw Results of WN18-------'
    print 'MR: {}'.format(round(rank, 2))
    print 'Hits@10: {}'.format(round(hit10, 2))
    print 'Hits@1: {}'.format(round(hit1, 2))
    print 'MRR: {}'.format(round(rrank, 3))

    print '------Filtered Results of WN18-------'
    print 'FMR: {}'.format(round(frank, 2))
    print 'FHits@10: {}'.format(round(fhit10, 2))
    print 'FHits@1: {}'.format(round(fhit1, 2))
    print 'FMRR: {}'.format(round(frrank, 3))

    #WN18RR
    numtest = 3134
    relidx = range(40943, 40954)
    left_hit10, left_fhit10, left_hit1, left_fhit1, left_rank, left_frank, left_rrank, left_frrank = read_alloutput(
        'left', 'WN18RR', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1, right_rank, right_frank, right_rrank, right_frrank = read_alloutput(
        'right', 'WN18RR', relidx)
    fhit10 = (float(left_fhit10) + float(right_fhit10)) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1) + float(right_fhit1)) / (2 * numtest) * 100
    hit10 = (float(left_hit10) + float(right_hit10)) / (2 * numtest) * 100
    hit1 = (float(left_hit1) + float(right_hit1)) / (2 * numtest) * 100
    rank = (float(left_rank) + float(right_rank)) / (2 * numtest)
    frank = (float(left_frank) + float(right_frank)) / (2 * numtest)
    rrank = (float(left_rrank) + float(right_rrank)) / (2 * numtest)
    frrank = (float(left_frrank) + float(right_frrank)) / (2 * numtest)

    print '------Raw Results of WN18RR-------'
    print 'MR: {}'.format(round(rank, 2))
    print 'Hits@10: {}'.format(round(hit10, 2))
    print 'Hits@1: {}'.format(round(hit1, 2))
    print 'MRR: {}'.format(round(rrank, 3))

    print '------Filtered Results of WN18RR-------'
    print 'FMR: {}'.format(round(frank, 2))
    print 'FHits@10: {}'.format(round(fhit10, 2))
    print 'FHits@1: {}'.format(round(fhit1, 2))
    print 'FMRR: {}'.format(round(frrank, 3))

    #YAGO3-10
    numtest = 5000
    relidx = range(123182, 123219)
    left_hit10, left_fhit10, left_hit1, left_fhit1, left_rank, left_frank, left_rrank, left_frrank = read_alloutput(
        'left', 'YAGO3-10', relidx)
    right_hit10, right_fhit10, right_hit1, right_fhit1, right_rank, right_frank, right_rrank, right_frrank = read_alloutput(
        'right', 'YAGO3-10', relidx)
    fhit10 = (float(left_fhit10) + float(right_fhit10)) / (2 * numtest) * 100
    fhit1 = (float(left_fhit1) + float(right_fhit1)) / (2 * numtest) * 100
    hit10 = (float(left_hit10) + float(right_hit10)) / (2 * numtest) * 100
    hit1 = (float(left_hit1) + float(right_hit1)) / (2 * numtest) * 100
    rank = (float(left_rank) + float(right_rank)) / (2 * numtest)
    frank = (float(left_frank) + float(right_frank)) / (2 * numtest)
    rrank = (float(left_rrank) + float(right_rrank)) / (2 * numtest)
    frrank = (float(left_frrank) + float(right_frrank)) / (2 * numtest)

    print '------Raw Results of YAGO3-10-------'
    print 'MR: {}'.format(round(rank, 2))
    print 'Hits@10: {}'.format(round(hit10, 2))
    print 'Hits@1: {}'.format(round(hit1, 2))
    print 'MRR: {}'.format(round(rrank, 3))

    print '------Filtered Results of YAGO3-10-------'
    print 'FMR: {}'.format(round(frank, 2))
    print 'FHits@10: {}'.format(round(fhit10, 2))
    print 'FHits@1: {}'.format(round(fhit1, 2))
    print 'FMRR: {}'.format(round(frrank, 3))




    combine_leftright('FB15k-237')
    combine_leftright('FB15k')
    combine_leftright('WN18')
    combine_leftright('WN18RR')
    combine_leftright('YAGO3-10')




