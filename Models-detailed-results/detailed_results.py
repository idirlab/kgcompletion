import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def models_detailed_results(dataset):

    fmr, fhit10, fhit1, fmrr = [], [], [], []
    minlist_fmr = []
    maxlist_fhit10 = []
    maxlist_fhit1 = []
    maxlist_fmrr = []

    one_one, one_n, n_one, n_n = 0, 0, 0, 0
    reltype = {}
    with open('../datasets/%s/reltype.txt' %dataset, 'r') as rt:
        for rtype in rt:
            r, t = rtype[:-1].split('\t')
            if t == '1-1':
                reltype[r] = 0
                one_one += 1
            if t == '1-n':
                reltype[r] = 1
                one_n += 1
            if t == 'n-1':
                reltype[r] = 2
                n_one += 1
            if t == 'n-n':
                reltype[r] = 3
                n_n += 1

    best_fmrr = np.zeros(shape=(7, 4))

    with open('./models_res_by_rel/result-by-rel-all/result-by-rel-all-%s.txt'%dataset, 'r') as in_file:  # TransE,  DistMult, ComplEx,ConvE, RotatE, TuckER, AMIE
        for line in in_file:
            rel, num, tmr, thit10, thit1, tmrr, tfmr, tfhit10, tfhit1, tfmrr, \
            dmr, dhit10, dhit1, dmrr, dfmr, dfhit10, dfhit1, dfmrr, \
            cmr, chit10, chit1, cmrr, cfmr, cfhit10, cfhit1, cfmrr, \
            cemr, cehit10, cehit1, cemrr, cefmr, cefhit10, cefhit1, cefmrr, \
            remr, rehit10, rehit1, remrr, refmr, refhit10, refhit1, refmrr, \
            tumr, tuhit10, tuhit1, tumrr, tufmr, tufhit10, tufhit1, tufmrr, \
            amiemr, amiehit10, amiehit1, amiemrr, amiefmr, amiefhit10, amiefhit1, amiefmrr = line.split('\t')


            for i in [tfmr, dfmr, cfmr, cefmr, refmr, tufmr, amiefmr]:
                i = float(i)
                fmr.append(i)
            for i in [tfhit10, dfhit10, cfhit10, cefhit10, refhit10, tufhit10, amiefhit10]:
                i = float(i)
                fhit10.append(i)
            for i in [tfhit1, dfhit1, cfhit1, cefhit1, refhit1, tufhit1, amiefhit1]:
                i = float(i)
                fhit1.append(i)
            for i in [tfmrr, dfmrr, cfmrr, cefmrr, refmrr, tufmrr, amiefmrr]:
                i = float(i)
                fmrr.append(i)



            # filtered metrics
            min_fmr = min(fmr)
            minlist_fmr += [idx for idx, p in enumerate(fmr) if p == min_fmr]

            max_fhit10 = max(fhit10)
            if max_fhit10 != 0:
                maxlist_fhit10 += [idx for idx, p in enumerate(fhit10) if p == max_fhit10]

            max_fhit1 = max(fhit1)
            if max_fhit1 != 0:
                maxlist_fhit1 += [idx for idx, p in enumerate(fhit1) if p == max_fhit1]

            max_fmrr = max(fmrr)
            maxlist_fmrr += [idx for idx, p in enumerate(fmrr) if p == max_fmrr]

            # fmrr by rel type amie
            fmrrlist = [idx for idx, p in enumerate(fmrr) if p == max_fmrr]  # best by FMRR
            for item in fmrrlist:
                best_fmrr[item, reltype[rel]] += 1


            fmr, fhit10, fhit1, fmrr = [], [], [], []

    models_best_res = np.zeros(shape=(7, 4))
    #######################AMIE#################################

    # 'FMR'
    for ind in range(7):
        best = minlist_fmr.count(ind)
        models_best_res[ind, 0] = best
    # 'FHits@10'
    for ind in range(7):
        best = maxlist_fhit10.count(ind)
        models_best_res[ind, 1] = best
    # 'FHits@1'
    for ind in range(7):
        best = maxlist_fhit1.count(ind)
        models_best_res[ind, 2] = best
    # 'FMRR'
    for ind in range(7):
        best = maxlist_fmrr.count(ind)
        models_best_res[ind, 3] = best

    print 'Number of relations on which each model is the most accurate on {}:'.format(dataset)
    row_format = "{:<15}" * 5
    metrics_list = ['FMR', 'FHits@10', 'FHits@1', 'FMRR']
    models_list = ['TransE', 'DistMult', 'ComplEx', 'ConvE', 'RotatE', 'TuckER', 'AMIE']
    print(row_format.format("", *metrics_list))
    for model, row in zip(models_list, models_best_res):
        print(row_format.format(model, *row))

    # Categorizing the relations on which each method has thebest result
    index = ['TransE', 'DistMult', 'ComplEx', 'ConvE', 'RotatE', 'TuckER', 'AMIE']
    df = pd.DataFrame({'1-1': best_fmrr[:, 0], '1-n': best_fmrr[:, 1], 'n-1': best_fmrr[:, 2], 'n-m': best_fmrr[:, 3]},
                      index=index)
    ax = df.plot.bar(rot=0, fontsize=12)

    best_fmrr[:, 0] = best_fmrr[:, 0] / one_one * 100
    best_fmrr[:, 1] = best_fmrr[:, 1] / one_n * 100
    best_fmrr[:, 2] = best_fmrr[:, 2] / n_one * 100
    best_fmrr[:, 3] = best_fmrr[:, 3] / n_n * 100

    # Break-down of methods achieving best performance on each type of relations
    index = ['1-1', '1-n', 'n-1', 'n-n']
    df = pd.DataFrame(
        {'TransE': best_fmrr[0], 'DistMult': best_fmrr[1], 'ComplEx': best_fmrr[2], 'ConvE': best_fmrr[3],
         'RotatE': best_fmrr[4], 'TuckER': best_fmrr[5], 'AMIE': best_fmrr[6]}, index=index)
    ax = df.plot.bar(rot=0)

    plt.show()


models_detailed_results('FB15k-237')
models_detailed_results('YAGO3-10')
models_detailed_results('WN18RR')