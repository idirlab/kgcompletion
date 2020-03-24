def result_by_rel_category(dataset):
    embedding_res_path = './test_results/'
    AMIE_res_path = '../AMIE/AMIE_res_by_rel/'
    rel_category_path = '../datasets/'
    one_one, one_n, n_one, n_n = 0, 0, 0, 0
    reltype = {}
    with open(rel_category_path + '%s/reltype.txt'%dataset, 'r')as g:
        for lines in g:
            r, t = lines[:-1].split('\t')
            reltype[r] = t

    models = ['transe', 'distmult', 'complex', 'conve', 'rotate', 'tucker']
    print 'FHits@10 by category of relations on {}'.format(dataset)
    print '{:<15}{:^20}{:^20}{:^20}{:^20}'.format('', '1-to-1', '1-to-n', 'n-to-1', 'n-to-m')
    print '{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('Model', 'L_FH10', 'R_FH10', 'L_FH10',
                                                                          'R_FH10', 'L_FH10', 'R_FH10', 'L_FH10',
                                                                          'R_FH10')
    for k in range(len(models)):
        head_res1, tail_res1 = 0, 0  # 1-1
        head_res2, tail_res2 = 0, 0  # 1-n
        head_res3, tail_res3 = 0, 0  # n-1
        head_res4, tail_res4 = 0, 0  # n-n
        i = 0

        with open(embedding_res_path + '%s/test-%s-%s.txt' %(dataset, models[k], dataset), 'r') as f:
            for line in f:
                try:
                    le, rel, re, fmr_left, mr_left, fmrr_left, mrr_left, fmr_right, mr_right, fmrr_right, mrr_right = line.split(
                        '\t')
                except:
                    le, rel, re, fmr_left, fmrr_left, fmr_right, fmrr_right, = line.split('\t')
                fmr_left = float(fmr_left)
                fmr_right = float(fmr_right)
                if reltype[rel] == '1-1':
                    one_one += 1
                    if fmr_left <= 10:
                        head_res1 += 1
                    if fmr_right <= 10:
                        tail_res1 += 1

                if reltype[rel] == '1-n':
                    one_n += 1
                    if fmr_left <= 10:
                        head_res2 += 1
                    if fmr_right <= 10:
                        tail_res2 += 1
                if reltype[rel] == 'n-1':
                    n_one += 1
                    if fmr_left <= 10:
                        head_res3 += 1
                    if fmr_right <= 10:
                        tail_res3 += 1
                if reltype[rel] == 'n-n':
                    n_n += 1
                    if fmr_left <= 10:
                        head_res4 += 1
                    if fmr_right <= 10:
                        tail_res4 += 1

            if k == 0:
                reltype_no = [one_one, one_n, n_one, n_n]
            head_res1 = round(float(head_res1) / reltype_no[0] * 100, 2)
            tail_res1 = round(float(tail_res1) / reltype_no[0] * 100, 2)

            head_res2 = round(float(head_res2) / reltype_no[1] * 100, 2)
            tail_res2 = round(float(tail_res2) / reltype_no[1] * 100, 2)

            head_res3 = round(float(head_res3) / reltype_no[2] * 100, 2)
            tail_res3 = round(float(tail_res3) / reltype_no[2] * 100, 2)

            head_res4 = round(float(head_res4) / reltype_no[3] * 100, 2)
            tail_res4 = round(float(tail_res4) / reltype_no[3] * 100, 2)

            print '{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(models[k], head_res1, tail_res1,
                                                                                  head_res2,
                                                                                  tail_res2, head_res3, tail_res3,
                                                                                  head_res4,
                                                                                  tail_res4)
    head_res1, tail_res1 = 0, 0  # 1-1
    head_res2, tail_res2 = 0, 0  # 1-n
    head_res3, tail_res3 = 0, 0  # n-1
    head_res4, tail_res4 = 0, 0  # n-n
    with open(AMIE_res_path + 'res_by_rel_left_%s.txt'%dataset, 'r') as leftres, open(
            AMIE_res_path + 'res_by_rel_right_%s.txt'%dataset, 'r') as rightres:
        for lines in zip(leftres, rightres):
            rel, test_no, mr, hit10_left, hit1_left, mrr_left, fmr_left, fhit10_left, fhit1_left, fmrr_left = lines[0][
                                                                                                              :-1].split(
                '\t')
            rel, test_no, mr, hit10_right, hit1_right, mrr_right, fmr_right, fhit10_right, fhit1_right, fmrr_right = \
            lines[1][:-1].split('\t')
            if reltype[rel] == '1-1':
                head_res1 += int(fhit10_left)
                tail_res1 += int(fhit10_right)

            if reltype[rel] == '1-n':
                head_res2 += int(fhit10_left)
                tail_res2 += int(fhit10_right)

            if reltype[rel] == 'n-1':
                head_res3 += int(fhit10_left)
                tail_res3 += int(fhit10_right)

            if reltype[rel] == 'n-n':
                head_res4 += int(fhit10_left)
                tail_res4 += int(fhit10_right)

        head_res1 = round(float(head_res1) / reltype_no[0] * 100, 2)
        tail_res1 = round(float(tail_res1) / reltype_no[0] * 100, 2)

        head_res2 = round(float(head_res2) / reltype_no[1] * 100, 2)
        tail_res2 = round(float(tail_res2) / reltype_no[1] * 100, 2)

        head_res3 = round(float(head_res3) / reltype_no[2] * 100, 2)
        tail_res3 = round(float(tail_res3) / reltype_no[2] * 100, 2)

        head_res4 = round(float(head_res4) / reltype_no[3] * 100, 2)
        tail_res4 = round(float(tail_res4) / reltype_no[3] * 100, 2)

        print '{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('AMIE', head_res1, tail_res1, head_res2,
                                                                              tail_res2, head_res3, tail_res3, head_res4,
                                                                              tail_res4)

datasets=['FB15k-237','WN18RR','YAGO3-10']

for dataset in datasets:
    result_by_rel_category(dataset)