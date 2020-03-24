import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def sort_results(input_file, output_file):
    with open(input_file, 'r') as input:
        lines = input.readlines()
    output = open(output_file, 'w')
    for line in sorted(lines, key=lambda line: (line.split('\t')[1], line.split('\t')[0], line.split('\t')[2]),
                       reverse=False):
        output.write(line)
    output.close()


def heatmap(dataset):
    input_models = ['complex', 'distmult', 'conve', 'transe', 'rotate', 'tucker']
    for model in input_models:
        sort_results('./test_results/%s/test-%s-%s.txt' % (dataset, model, dataset),
                     './test_results/%s/sorted-test-%s-%s.txt' % (dataset, model, dataset))
    rels = []
    with open('./test_results/%s/test-%s-%s.txt' % (dataset, input_models[0], dataset), 'r')as f:
        for line in f:
            rels.append(line.split('\t')[1])
    rels = list(set(rels))
    rels.sort()

    mrr_f = []
    maxlist = []
    x = np.zeros(shape=(6, len(rels)))
    for rid, r in enumerate(rels):
        num = 0
        with open('./test_results/%s/sorted-test-transe-%s.txt' % (dataset, dataset), 'r') as k1, \
                open('./test_results/%s/sorted-test-distmult-%s.txt' % (dataset, dataset), 'r') as k2, \
                open('./test_results/%s/sorted-test-complex-%s.txt' % (dataset, dataset), 'r')as k3, \
                open('./test_results/%s/sorted-test-conve-%s.txt' % (dataset, dataset), 'r') as k4, \
                open('./test_results/%s/sorted-test-rotate-%s.txt' % (dataset, dataset), 'r') as k5, \
                open('./test_results/%s/sorted-test-tucker-%s.txt' % (dataset, dataset), 'r') as k6:
            for i, j, k, l, m, n in zip(k1, k2, k3, k4, k5, k6):
                le, rel, re, mr_lf, mr_l, mrr_lf, mrr_l, mr_rf, mr_r, mrr_rf, mrr_r = i.split('\t')
                if rel == r:
                    num += 1
                    mrr_lf = float(mrr_lf)
                    mrr_rf = float(mrr_rf)
                    mrr_f.append(round((mrr_lf + mrr_rf) / 2, 3))
                    le, rel, re, mr_lf, mr_l, mrr_lf, mrr_l, mr_rf, mr_r, mrr_rf, mrr_r = j.split('\t')
                    mrr_lf = float(mrr_lf)
                    mrr_rf = float(mrr_rf)
                    mrr_f.append(round((mrr_lf + mrr_rf) / 2, 3))
                    le, rel, re, mr_lf, mr_l, mrr_lf, mrr_l, mr_rf, mr_r, mrr_rf, mrr_r = k.split('\t')
                    mrr_lf = float(mrr_lf)
                    mrr_rf = float(mrr_rf)
                    mrr_f.append(round((mrr_lf + mrr_rf) / 2, 3))
                    le, rel, re, mr_lf, mr_l, mrr_lf, mrr_l, mr_rf, mr_r, mrr_rf, mrr_r = l.split('\t')
                    mrr_lf = float(mrr_lf)
                    mrr_rf = float(mrr_rf)
                    mrr_f.append(round((mrr_lf + mrr_rf) / 2, 3))
                    le, rel, re, mr_lf, mr_l, mrr_lf, mrr_l, mr_rf, mr_r, mrr_rf, mrr_r = m.split('\t')
                    mrr_lf = float(mrr_lf)
                    mrr_rf = float(mrr_rf)
                    mrr_f.append(round((mrr_lf + mrr_rf) / 2, 3))
                    le, rel, re, mr_lf, mr_l, mrr_lf, mrr_l, mr_rf, mr_r, mrr_rf, mrr_r = n.split('\t')
                    mrr_lf = float(mrr_lf)
                    mrr_rf = float(mrr_rf)
                    mrr_f.append(round((mrr_lf + mrr_rf) / 2, 3))
                    max_mrr = max(mrr_f)
                    maxlist += [idx for idx, p in enumerate(mrr_f) if p == max_mrr]
                    mrr_f = []

            for ind in range(6):
                best = maxlist.count(ind)
                best_p = round(float(best) / num, 2) * 100
                x[ind, rid] = best_p
            maxlist = []

    models = ['TransE', 'DistMult', 'ComplEx', 'ConvE', 'RotatE', 'TuckER']
    df = pd.DataFrame(x)
    fig, ax = plt.subplots()
    cbar_ax = fig.add_axes([.92, .1, .01, 0.78])
    sns.heatmap(df, yticklabels=models, xticklabels=10, cmap="coolwarm", ax=ax, cbar_ax=cbar_ax)  #
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        tick.set_fontsize(30)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(28)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    plt.show()


heatmap('WN18RR')
heatmap('FB15k-237')

