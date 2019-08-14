import numpy as np


def sort_results(input_file, output_file):

    with open(input_file, 'r') as input:
        lines = input.readlines()
    output = open(output_file, 'w')
    for line in sorted(lines, key=lambda line: (line.split('\t')[1], line.split('\t')[0], line.split('\t')[2]),
                       reverse=False):
        output.write(line)
    output.close()



def revers_percenatge():
    a_fmr, a_hit10, a_hit1, a_fmrr = [], [], [], []  # ANALOGY

    c_fmr, c_hit10, c_hit1, c_fmrr = [], [], [], []  # ComplEx

    d_fmr, d_hit10, d_hit1, d_fmrr = [], [], [], []  # DistMult

    ce_fmr, ce_hit10, ce_hit1, ce_fmrr = [], [], [], []  # ConvE

    with open('./test_results/sorted-test-analogy-15.txt', 'r')as k1, open('./test_results/sorted-test-complex-15.txt',
                                                                           'r')as k2, \
            open('./test_results/sorted-test-distmult-15.txt', 'r') as k3, open(
        './test_results/sorted-test-conve-15.txt', 'r') as k4, \
            open('./test_results/sorted-test-transe-15.txt', 'r') as k5, open(
        '../FB15k-redundancy/results/sorted-test-redundancies.txt', 'r') as k6:
        for analogy_res, complex_res, distmult_res, conve_res, trans_res, n in zip(k1, k2, k3, k4, k5, k6):
            h, r, t, rev = n[:-1].split('\t')
            rev = int(rev)
            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = trans_res.split(
                '\t')  # transe
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            transe_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            transe_fmrr = (round((left_fmrr + right_fmrr) / 2, 2))
            if left_fmr <= 10:
                left_fhit10 = 1
            else:
                left_fhit10 = 0
            if right_fmr <= 10:
                right_fhit10 = 1
            else:
                right_fhit10 = 0

            if left_fmr <= 1:
                left_fhit1 = 1
            else:
                left_fhit1 = 0
            if right_fmr <= 1:
                right_fhit1 = 1
            else:
                right_fhit1 = 0
            transe_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))
            transe_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = analogy_res.split(
                '\t')  # analogy
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            analogy_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            analogy_fmrr = (round((left_fmrr + right_fmrr) / 2, 2))
            if left_fmr <= 10:
                left_fhit10 = 1
            else:
                left_fhit10 = 0
            if right_fmr <= 10:
                right_fhit10 = 1
            else:
                right_fhit10 = 0

            if left_fmr <= 1:
                left_fhit1 = 1
            else:
                left_fhit1 = 0
            if right_fmr <= 1:
                right_fhit1 = 1
            else:
                right_fhit1 = 0

            analogy_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))
            analogy_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))

            if analogy_fmr < transe_fmr:
                a_fmr.append(rev)
            if analogy_fmrr > transe_fmrr:
                a_fmrr.append(rev)
            if analogy_hit10 > transe_hit10:
                a_hit10.append(rev)
            if analogy_hit1 > transe_hit1:
                a_hit1.append(rev)

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = complex_res.split(
                '\t')  # complex
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            complex_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            complex_fmrr = (round((left_fmrr + right_fmrr) / 2, 2))
            if left_fmr <= 10:
                left_fhit10 = 1
            else:
                left_fhit10 = 0
            if right_fmr <= 10:
                right_fhit10 = 1
            else:
                right_fhit10 = 0

            if left_fmr <= 1:
                left_fhit1 = 1
            else:
                left_fhit1 = 0
            if right_fmr <= 1:
                right_fhit1 = 1
            else:
                right_fhit1 = 0

            complex_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))
            complex_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))
            if complex_fmr < transe_fmr:
                c_fmr.append(rev)
            if complex_fmrr > transe_fmrr:
                c_fmrr.append(rev)
            if complex_hit10 > transe_hit10:
                c_hit10.append(rev)
            if complex_hit1 > transe_hit1:
                c_hit1.append(rev)

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = distmult_res.split(
                '\t')  # distmult
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            distmult_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            distmult_fmrr = (round((left_fmrr + right_fmrr) / 2, 2))
            if left_fmr <= 10:
                left_fhit10 = 1
            else:
                left_fhit10 = 0
            if right_fmr <= 10:
                right_fhit10 = 1
            else:
                right_fhit10 = 0

            if left_fmr <= 1:
                left_fhit1 = 1
            else:
                left_fhit1 = 0
            if right_fmr <= 1:
                right_fhit1 = 1
            else:
                right_fhit1 = 0

            distmult_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))
            distmult_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))
            if distmult_fmr < transe_fmr:
                d_fmr.append(rev)
            if distmult_fmrr > transe_fmrr:
                d_fmrr.append(rev)
            if distmult_hit10 > transe_hit10:
                d_hit10.append(rev)
            if distmult_hit1 > transe_hit1:
                d_hit1.append(rev)

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = conve_res.split(
                '\t')  # conve
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            conve_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            conve_fmrr = (round((left_fmrr + right_fmrr) / 2, 2))
            if left_fmr <= 10:
                left_fhit10 = 1
            else:
                left_fhit10 = 0
            if right_fmr <= 10:
                right_fhit10 = 1
            else:
                right_fhit10 = 0

            if left_fmr <= 1:
                left_fhit1 = 1
            else:
                left_fhit1 = 0
            if right_fmr <= 1:
                right_fhit1 = 1
            else:
                right_fhit1 = 0

            conve_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))
            conve_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))
            if conve_fmr < transe_fmr:
                ce_fmr.append(rev)
            if conve_fmrr > transe_fmrr:
                ce_fmrr.append(rev)
            if conve_hit10 > transe_hit10:
                ce_hit10.append(rev)
            if conve_hit1 > transe_hit1:
                ce_hit1.append(rev)
    # ANALOGY
    models_reverse_percentage = np.zeros(shape=(4, 4))
    models_reverse_percentage[0, 0] = round(sum(a_fmr) * 100 / float(len(a_fmr)), 2)
    models_reverse_percentage[0, 1] = round(sum(a_hit10) * 100 / float(len(a_hit10)), 2)
    models_reverse_percentage[0, 2] = round(sum(a_hit1) * 100 / float(len(a_hit1)), 2)
    models_reverse_percentage[0, 3] = round(sum(a_fmrr) * 100 / float(len(a_fmrr)), 2)
    # ComplEx
    models_reverse_percentage[1, 0] = round(sum(c_fmr) * 100 / float(len(c_fmr)), 2)
    models_reverse_percentage[1, 1] = round(sum(c_hit10) * 100 / float(len(c_hit10)), 2)
    models_reverse_percentage[1, 2] = round(sum(c_hit1) * 100 / float(len(c_hit1)), 2)
    models_reverse_percentage[1, 3] = round(sum(c_fmrr) * 100 / float(len(c_fmrr)), 2)
    # DistMult
    models_reverse_percentage[2, 0] = round(sum(d_fmr) * 100 / float(len(d_fmr)), 2)
    models_reverse_percentage[2, 1] = round(sum(d_hit10) * 100 / float(len(d_hit10)), 2)
    models_reverse_percentage[2, 2] = round(sum(d_hit1) * 100 / float(len(d_hit1)), 2)
    models_reverse_percentage[2, 3] = round(sum(d_fmrr) * 100 / float(len(d_fmrr)), 2)
    # ConvE
    models_reverse_percentage[3, 0] = round(sum(ce_fmr) * 100 / float(len(ce_fmr)), 2)
    models_reverse_percentage[3, 1] = round(sum(ce_hit10) * 100 / float(len(ce_hit10)), 2)
    models_reverse_percentage[3, 2] = round(sum(ce_hit1) * 100 / float(len(ce_hit1)), 2)
    models_reverse_percentage[3, 3] = round(sum(ce_fmrr) * 100 / float(len(ce_fmrr)), 2)
    print 'Most test triples in FB15k on which various models outperformed TransE have reverse and duplicate triples in training set'
    row_format = "{:<15}" * 5
    metrics_list = ['FMR', 'FHits@10', 'Fhits@1', 'FMRR']
    models_list = ['ANALOGY', 'ComplEx', 'DistMult', 'ConvE']
    print(row_format.format("", *metrics_list))
    for model, row in zip(models_list, models_reverse_percentage):
        print(row_format.format(model, *row))

sort_results('../FB15k-redundancy/results/test_redundancies.txt', '../FB15k-redundancy/results/sorted-test-redundancies.txt')
input_models = ['analogy', 'complex', 'distmult', 'conve', 'transe']
for model in input_models:
    sort_results('./test_results/test-%s-15.txt'%model, './test_results/sorted-test-%s-15.txt'%model)
revers_percenatge()
