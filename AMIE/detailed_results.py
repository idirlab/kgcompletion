import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mr, hit10, hit1, mrr = [], [], [], []
fmr, fhit10, fhit1, fmrr = [], [], [], []
minlist_mr, minlist_fmr = [], []
maxlist_hit10, maxlist_fhit10, maxlist_fhit10_amie = [], [], []
maxlist_hit1, maxlist_fhit1 = [], []
maxlist_mrr, maxlist_fmrr = [], []

one_one, one_n, n_one, n_n = 0, 0, 0, 0
reltype = {}
with open('../data/reltype.txt', 'r') as rt:
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
best_fmrr = np.zeros(shape=(5, 4))
best_fhi10 = np.zeros(shape=(6, 4))
z = np.zeros(shape=(5, 4))  # amie vs embedding models
with open('./AMIE_res_by_rel/res-by-rel-all-methods.txt', 'r') as in_file:  # ANALOGY, ComplEx, DistMult, ConvE, TransE
    for line in in_file:
        rel, num, amr, ahit10, ahit1, amrr, afmr, afhit10, afhit1, afmrr, \
        cmr, chit10, chit1, cmrr, cfmr, cfhit10, cfhit1, cfmrr, \
        dmr, dhit10, dhit1, dmrr, dfmr, dfhit10, dfhit1, dfmrr, \
        cemr, cehit10, cehit1, cemrr, cefmr, cefhit10, cefhit1, cefmrr, \
        tmr, thit10, thit1, tmrr, tfmr, tfhit10, tfhit1, tfmrr, amiehit10 = line.split('\t')
        for i in [amr, cmr, dmr, cemr, tmr]:
            i = float(i)
            mr.append(i)
        for i in [ahit10, chit10, dhit10, cehit10, thit10]:
            i = float(i)
            hit10.append(i)
        for i in [ahit1, chit1, dhit1, cehit1, thit1]:
            i = float(i)
            hit1.append(i)
        for i in [amrr, cmrr, dmrr, cemrr, tmrr]:
            i = float(i)
            mrr.append(i)

        for i in [afmr, cfmr, dfmr, cefmr, tfmr]:
            i = float(i)
            fmr.append(i)
        for i in [afhit10, cfhit10, dfhit10, cefhit10, tfhit10, amiehit10]:
            i = float(i)
            fhit10.append(i)
        for i in [afhit1, cfhit1, dfhit1, cefhit1, tfhit1]:
            i = float(i)
            fhit1.append(i)
        for i in [afmrr, cfmrr, dfmrr, cefmrr, tfmrr]:
            i = float(i)
            fmrr.append(i)

        ### AMIE vs Embedding models
        if fhit10[5] > fhit10[0]:  # amie vs analogy
            z[0, reltype[rel]] += 1
        if fhit10[5] > fhit10[1]:  # amie vs complex
            z[1, reltype[rel]] += 1
        if fhit10[5] > fhit10[2]:  # amie vs distmult
            z[2, reltype[rel]] += 1
        if fhit10[5] > fhit10[3]:  # amie vs conve
            z[3, reltype[rel]] += 1
        if fhit10[5] > fhit10[4]:  # amie vs transe
            z[4, reltype[rel]] += 1

        # raw metrics
        min_mr = min(mr)
        minlist_mr += [idx for idx, p in enumerate(mr) if p == min_mr]

        max_hit10 = max(hit10)
        if max_hit10 != 0:
            maxlist_hit10 += [idx for idx, p in enumerate(hit10) if p == max_hit10]

        max_hit1 = max(hit1)
        if max_hit1 != 0:
            maxlist_hit1 += [idx for idx, p in enumerate(hit1) if p == max_hit1]

        max_mrr = max(mrr)
        maxlist_mrr += [idx for idx, p in enumerate(mrr) if p == max_mrr]

        # filtered metrics
        min_fmr = min(fmr)
        minlist_fmr += [idx for idx, p in enumerate(fmr) if p == min_fmr]

        max_fhit10 = max(fhit10[:-1])
        if max_fhit10 != 0:
            maxlist_fhit10 += [idx for idx, p in enumerate(fhit10[:-1]) if p == max_fhit10]

        max_fhit1 = max(fhit1)
        if max_hit1 != 0:
            maxlist_fhit1 += [idx for idx, p in enumerate(fhit1) if p == max_fhit1]

        max_fmrr = max(fmrr)
        maxlist_fmrr += [idx for idx, p in enumerate(fmrr) if p == max_fmrr]

        fmrrlist = [idx for idx, p in enumerate(fmrr) if p == max_fmrr]  # best by FMRR
        for item in fmrrlist:
            best_fmrr[item, reltype[rel]] += 1

        max_fhit10 = max(fhit10)  # best by FHit@10, AMIE included
        if max_fhit10 != 0:
            maxlist_fhit10_amie += [idx for idx, p in enumerate(fhit10) if p == max_fhit10]
            fhit10list = [idx for idx, p in enumerate(fhit10) if p == max_fhit10]
            for item in fhit10list:
                best_fhi10[item, reltype[rel]] += 1

        mr, hit10, hit1, mrr = [], [], [], []
        fmr, fhit10, fhit1, fmrr = [], [], [], []

models_best_res = np.zeros(shape=(5, 8))
models_best_res_amie = np.zeros(shape=(1, 6))
# 'MR'
for ind in range(5):
    best = minlist_mr.count(ind)
    models_best_res[ind, 0] = best
# Hits@10'
for ind in range(5):
    best = maxlist_hit10.count(ind)
    models_best_res[ind, 1] = best
# 'Hits@1'
for ind in range(5):
    best = maxlist_hit1.count(ind)
    models_best_res[ind, 2] = best
# 'MRR'
for ind in range(5):
    best = maxlist_mrr.count(ind)
    models_best_res[ind, 3] = best
# 'FMR'
for ind in range(5):
    best = minlist_fmr.count(ind)
    models_best_res[ind, 4] = best
# 'FHits@10'
for ind in range(5):
    best = maxlist_fhit10.count(ind)
    models_best_res[ind, 5] = best
# 'FHits@1'
for ind in range(5):
    best = maxlist_fhit1.count(ind)
    models_best_res[ind, 6] = best
# 'FMRR'
for ind in range(5):
    best = maxlist_fmrr.count(ind)
    models_best_res[ind, 7] = best
# Number of relations on which each model is the most accurate on FB15k-237, AMIE included
for ind in range(6):
    best = maxlist_fhit10_amie.count(ind)
    models_best_res_amie[0, ind] = best

print 'Number of relations on which each model is the most accurate on FB15k-237:'
row_format = "{:<15}" * 9
metrics_list = ['MR', 'Hits@10', 'Hits@1', 'MRR', 'FMR', 'FHits@10', 'FHits@1', 'FMRR']
models_list = ['ANALOGY', 'ComplEx', 'DistMult', 'ConvE', 'TransE']
print(row_format.format("", *metrics_list))
for model, row in zip(models_list, models_best_res):
    print(row_format.format(model, *row))

print 'Number of relations on which each model is the most accurate on FB15k-237, AMIE included:'
row_format = "{:<15}" * 7
models_list = ['ANALOGY', 'ComplEx', 'DistMult', 'ConvE', 'TransE', 'AMIE']
print(row_format.format("", *models_list))
for row in models_best_res_amie:
    print(row_format.format('Hits@10', *row))

# Categorizing the relations on which each method has thebest result
index = ['ANALOGY', 'ComplEx', 'DistMult', 'ConvE', 'TransE']
df = pd.DataFrame({'1-1': best_fmrr[:, 0], '1-n': best_fmrr[:, 1], 'n-1': best_fmrr[:, 2], 'n-m': best_fmrr[:, 3]},
                  index=index)
ax = df.plot.bar(rot=0)

best_fmrr[:, 0] = best_fmrr[:, 0] / one_one * 100
best_fmrr[:, 1] = best_fmrr[:, 1] / one_n * 100
best_fmrr[:, 2] = best_fmrr[:, 2] / n_one * 100
best_fmrr[:, 3] = best_fmrr[:, 3] / n_n * 100

best_fhi10[:, 0] = best_fhi10[:, 0] / one_one * 100
best_fhi10[:, 1] = best_fhi10[:, 1] / one_n * 100
best_fhi10[:, 2] = best_fhi10[:, 2] / n_one * 100
best_fhi10[:, 3] = best_fhi10[:, 3] / n_n * 100

z[:, 0] = z[:, 0] / one_one * 100
z[:, 1] = z[:, 1] / one_n * 100
z[:, 2] = z[:, 2] / n_one * 100
z[:, 3] = z[:, 3] / n_n * 100

# Break-down of methods achieving best performance on each type of relations
index = ['1-1', '1-n', 'n-1', 'n-n']
df = pd.DataFrame({'ANALOGY': best_fmrr[0], 'ComplEx': best_fhi10[1], 'DistMult': best_fhi10[2], 'ConvE': best_fhi10[3],
                   'TransE': best_fhi10[4]}, index=index)
ax = df.plot.bar(rot=0)

# Break-down of methods achieving best performance on each type of relations, AMIE included,FB15k-237,FHits@10
index = ['1-1', '1-n', 'n-1', 'n-n']
df = pd.DataFrame(
    {'ANALOGY': best_fhi10[0], 'ComplEx': best_fhi10[1], 'DistMult': best_fhi10[2], 'ConvE': best_fhi10[3],
     'TransE': best_fhi10[4], 'AMIE': best_fhi10[5]}, index=index)
ax = df.plot.bar(rot=0)

# Head-to-head comparisons of AMIE with other models on FB15k-237 using FHits@10
index = ['amie vs. analogy', 'amie vs. complex', 'amie vs. distmult', 'amie vs. conve', 'amie vs. transe']
df = pd.DataFrame({'1-1': z[:, 0], '1-n': z[:, 1], 'n-1': z[:, 2], 'n-n': z[:, 3]}, index=index)
ax = df.plot.bar(rot=0)
plt.show()
