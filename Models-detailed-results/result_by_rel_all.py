def res_by_rel(testfile, inputfile):
    with open('../data/%s' % testfile, 'r') as f:
        lines = f.readlines()
    rels = []
    for num_rel in lines:
        h, r, t = num_rel.split('\t')
        rels.append(r)
    rels = set(rels)
    g = open('./models_res_by_rel/res-by-rel-%s.txt' % inputfile, 'w')
    g.write(
        '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('Relation', 'Num', 'MR', 'hits@10', 'hits@1', 'MRR', 'FMR',
                                                          'Fhits@10', 'Fhits@1', 'FMRR'))
    mr, fmr, allhit10, allfhit10, allhit1, allfhit1, allmrr, allfmrr = 0, 0, 0, 0, 0, 0, 0, 0
    for rel in rels:
        num_rel = 0
        hit10_left, fhit10_left, hit10_right, fhit10_right = 0, 0, 0, 0
        hit1_left, fhit1_left, hit1_right, fhit1_right = 0, 0, 0, 0
        rank, frank = 0, 0
        mrr, fmrr = 0, 0

        with open('./test_results/test-%s.txt' % inputfile,
                  'r') as f:  # input files contain the results of each test triple for seperate models
            for line in f:
                if rel == line.split('\t')[1]:
                    num_rel += 1
                    # each input file contains test triple plus filtered rank, rank, filtered reciprocal rank, reciprocal rank for left and head enitities
                    h, r, t, frank_left, rank_left, frr_left, rr_left, frank_right, rank_right, frr_right, rr_right = line.split(
                        '\t')
                    frank_left = float(frank_left)
                    rank_left = float(rank_left)
                    frr_left = float(frr_left)
                    rr_left = float(rr_left)

                    frank_right = float(frank_right)
                    rank_right = float(rank_right)
                    frr_right = float(frr_right)
                    rr_right = float(rr_right)

                    if rank_left <= 10:
                        hit10_left += 1
                    if rank_right <= 10:
                        hit10_right += 1
                    if frank_left <= 10:
                        fhit10_left += 1
                    if frank_right <= 10:
                        fhit10_right += 1

                    if rank_left <= 1:
                        hit1_left += 1
                    if rank_right <= 1:
                        hit1_right += 1
                    if frank_left <= 1:
                        fhit1_left += 1
                    if frank_right <= 1:
                        fhit1_right += 1

                    rank += rank_left + rank_right
                    frank += frank_left + frank_right
                    mrr += rr_left + rr_right
                    fmrr += frr_left + frr_right

            # results for whole testset
            mr += rank
            fmr += frank
            allmrr += mrr
            allfmrr += fmrr
            allhit10 += hit10_left + hit10_right
            allfhit10 += fhit10_left + fhit10_right
            allhit1 += hit1_left + hit1_right
            allfhit1 += fhit1_left + fhit1_right

            # results for each relation
            rank = round(rank / (2 * num_rel), 2)
            frank = round(frank / (2 * num_rel), 2)

            mrr = round(mrr * 100 / (2 * num_rel), 2)
            fmrr = round(fmrr * 100 / (2 * num_rel), 2)

            hit10 = float((hit10_left + hit10_right)) * 100 / (2 * num_rel)
            hit10 = round(hit10, 2)
            fhit10 = float((fhit10_left + fhit10_right)) * 100 / (2 * num_rel)
            fhit10 = round(fhit10, 2)

            hit1 = float((hit1_left + hit1_right)) * 100 / (2 * num_rel)
            hit1 = round(hit1, 2)
            fhit1 = float((fhit1_left + fhit1_right)) * 100 / (2 * num_rel)
            fhit1 = round(fhit1, 2)

            g.write(
                '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(rel, num_rel, rank, hit10, hit1, mrr, frank, fhit10,
                                                                  fhit1, fmrr))

    g.close()
    mr = round(float(mr) / (2 * 20466), 2)
    fmr = round(float(fmr) / (2 * 20466), 2)
    allmrr = round(float(allmrr) / (2 * 20466), 2)
    allfmrr = round(float(allfmrr) / (2 * 20466), 2)
    allhit10 = round(float(allhit10) * 100 / (2 * 20466), 2)
    allfhit10 = round(float(allfhit10) * 100 / (2 * 20466), 2)
    allhit1 = round(float(allhit1) * 100 / (2 * 20466), 2)
    allfhit1 = round(float(allfhit1) * 100 / (2 * 20466), 2)

    print '{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(inputfile, mr, allhit10, allhit1, allmrr, fmr,
                                                                          allfhit10, allfhit1, allfmrr)

#results of models on FB15k-237
print '{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('Model', 'MR', 'Hits@10', 'Hits@1', 'MRR', 'FMR',
                                                                      'FHits@10', 'FHits@1', 'FMRR')

for models in ['conve-237', 'distmult-237', 'analogy-237', 'complex-237', 'transe-237']:
    res_by_rel('FB15k237-test.txt', models)