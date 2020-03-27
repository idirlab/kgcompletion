import numpy as np


def sort_results(input_file, output_file):

    with open(input_file, 'r') as input:
        lines = input.readlines()
    output = open(output_file, 'w')
    for line in sorted(lines, key=lambda line: (line.split('\t')[1], line.split('\t')[0], line.split('\t')[2]),
                       reverse=False):
        output.write(line)
    output.close()



def revers_percenatge(dataset):
    c_fmr, c_hit10, c_hit1, c_fmrr = [], [], [], []  # ComplEx
    d_fmr, d_hit10, d_hit1, d_fmrr = [], [], [], []  # DistMult
    ce_fmr, ce_hit10, ce_hit1, ce_fmrr = [], [], [], []  # ConvE
    re_fmr, re_hit10, re_hit1, re_fmrr = [], [], [], []  # Rotate
    t_fmr, t_hit10, t_hit1, t_fmrr = [], [], [], []  # Tucker


    with    open('./test_results/%s/sorted-test-complex_%s.txt' % (dataset,dataset), 'r')as k2, \
            open('./test_results/%s/sorted-test-distmult_%s.txt' % (dataset,dataset), 'r') as k3, \
            open('./test_results/%s/sorted-test-conve_%s.txt' % (dataset,dataset), 'r') as k4, \
            open('./test_results/%s/sorted-test-transe_%s.txt' % (dataset,dataset), 'r') as k5, \
            open('./test_results/%s/sorted-test-RotatE_%s.txt' % (dataset,dataset), 'r') as k6,\
            open('./test_results/%s/sorted-test-TuckER_%s.txt' % (dataset,dataset), 'r') as k7,\
            open('../data-redundancy/results/%s/sorted-test-redundancies.txt' % dataset, 'r') as k8:
        for complex_res, distmult_res, conve_res, trans_res,rotate_res,tucker_res, n in zip(k2, k3, k4, k5, k6,k7,k8):
            h, r, t, rev = n[:-1].split('\t')
            rev = int(rev)
            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = trans_res.split(
                '\t')  # transe
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            transe_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            transe_fmrr = (round((left_fmrr + right_fmrr) / 2, 3))
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

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = complex_res.split(
                '\t')  # complex
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            complex_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            complex_fmrr = (round((left_fmrr + right_fmrr) / 2, 3))
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
            distmult_fmrr = (round((left_fmrr + right_fmrr) / 2, 3))
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
            conve_fmrr = (round((left_fmrr + right_fmrr) / 2, 3))
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

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = rotate_res.split(
                '\t')  # rotate
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            rotate_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            rotate_fmrr = (round((left_fmrr + right_fmrr) / 2, 3))
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

            rotate_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))
            rotate_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))
            if rotate_fmr < transe_fmr:
                re_fmr.append(rev)
            if rotate_fmrr > transe_fmrr:
                re_fmrr.append(rev)
            if rotate_hit10 > transe_hit10:
                re_hit10.append(rev)
            if rotate_hit1 > transe_hit1:
                re_hit1.append(rev)

            le, rel, re, left_fmr, left_mr, left_fmrr, left_mrr, right_fmr, right_mr, right_fmrr, right_mrr = tucker_res.split(
                '\t')  # tucker
            left_fmr = float(left_fmr)
            right_fmr = float(right_fmr)
            tucker_fmr = (round((left_fmr + right_fmr) / 2, 2))
            left_fmrr = float(left_fmrr)
            right_fmrr = float(right_fmrr)
            tucker_fmrr = (round((left_fmrr + right_fmrr) / 2, 3))
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

            tucker_hit1 = (round((right_fhit1 + left_fhit1) / 2, 2))
            tucker_hit10 = (round((right_fhit10 + left_fhit10) / 2, 2))
            if tucker_fmr < transe_fmr:
                t_fmr.append(rev)
            if tucker_fmrr > transe_fmrr:
                t_fmrr.append(rev)
            if tucker_hit10 > transe_hit10:
                t_hit10.append(rev)
            if tucker_hit1 > transe_hit1:
                t_hit1.append(rev)

    models_reverse_percentage = np.zeros(shape=(5, 4))
    # DistMult
    models_reverse_percentage[0, 0] = round(sum(d_fmr) * 100 / float(len(d_fmr)), 2)
    models_reverse_percentage[0, 1] = round(sum(d_hit10) * 100 / float(len(d_hit10)), 2)
    models_reverse_percentage[0, 2] = round(sum(d_hit1) * 100 / float(len(d_hit1)), 2)
    models_reverse_percentage[0, 3] = round(sum(d_fmrr) * 100 / float(len(d_fmrr)), 2)
    # ComplEx
    models_reverse_percentage[1, 0] = round(sum(c_fmr) * 100 / float(len(c_fmr)), 2)
    models_reverse_percentage[1, 1] = round(sum(c_hit10) * 100 / float(len(c_hit10)), 2)
    models_reverse_percentage[1, 2] = round(sum(c_hit1) * 100 / float(len(c_hit1)), 2)
    models_reverse_percentage[1, 3] = round(sum(c_fmrr) * 100 / float(len(c_fmrr)), 2)

    # ConvE
    models_reverse_percentage[2, 0] = round(sum(ce_fmr) * 100 / float(len(ce_fmr)), 2)
    models_reverse_percentage[2, 1] = round(sum(ce_hit10) * 100 / float(len(ce_hit10)), 2)
    models_reverse_percentage[2, 2] = round(sum(ce_hit1) * 100 / float(len(ce_hit1)), 2)
    models_reverse_percentage[2, 3] = round(sum(ce_fmrr) * 100 / float(len(ce_fmrr)), 2)
    # RoatE
    models_reverse_percentage[3, 0] = round(sum(re_fmr) * 100 / float(len(re_fmr)), 2)
    models_reverse_percentage[3, 1] = round(sum(re_hit10) * 100 / float(len(re_hit10)), 2)
    models_reverse_percentage[3, 2] = round(sum(re_hit1) * 100 / float(len(re_hit1)), 2)
    models_reverse_percentage[3, 3] = round(sum(re_fmrr) * 100 / float(len(re_fmrr)), 2)
    # TuckER
    models_reverse_percentage[4, 0] = round(sum(t_fmr) * 100 / float(len(t_fmr)), 2)
    models_reverse_percentage[4, 1] = round(sum(t_hit10) * 100 / float(len(t_hit10)), 2)
    models_reverse_percentage[4, 2] = round(sum(t_hit1) * 100 / float(len(t_hit1)), 2)
    models_reverse_percentage[4, 3] = round(sum(t_fmrr) * 100 / float(len(t_fmrr)), 2)
    print 'Most test triples in {} on which various models outperformed TransE have reverse and duplicate triples in training set'.format(dataset)
    row_format = "{:<15}" * 5
    metrics_list = ['FMR', 'FHits@10', 'Fhits@1', 'FMRR']
    models_list = ['DistMult','ComplEx', 'ConvE','RotatE', 'TuckER']
    print(row_format.format("", *metrics_list))
    for model, row in zip(models_list, models_reverse_percentage):
        print(row_format.format(model, *row))

sort_results('../data-redundancy/results/WN18/test-redundancies.txt', '../data-redundancy/results/WN18/sorted-test-redundancies.txt')
input_models = ['complex', 'distmult', 'conve', 'transe', 'TuckER','RotatE']
#for model in input_models:
    #sort_results('./test_results/test-%s_FB15k.txt'%model, './test_results/sorted-test-%s_FB15k.txt'%model)
revers_percenatge('FB15k')
revers_percenatge('WN18')
