
def read_rotate_results(file_path):
    triple_result_dict = {}
    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            h, r, t = line[0], line[1], line[2]
            mr, mrr = line[3], line[4]
            triple_result_dict[(h, r, t)] = [mr, mrr]
            cnt += 1
    assert cnt == len(triple_result_dict)
    return triple_result_dict


def read_tucker_results(file_path):
    head_result_dict, tail_result_dict = {}, {}
    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            h, r, t = line[0], line[1], line[2]
            mr, mrr = line[3], line[4]
            if r.endswith('_reverse'):
                r = r.split('_reverse')[0]
                head_result_dict[(t, r, h)] = [mr, mrr]
            else:
                tail_result_dict[(h, r, t)] = [mr, mrr]
                cnt += 1
    assert cnt == len(tail_result_dict) == len(head_result_dict)
    return head_result_dict, tail_result_dict


def read_id2item_dict(file_path):
    id2item_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            id2item_dict[line[0]] = line[1]
    return id2item_dict


def run_rotate_results(data_folder, result_folder, out_file_path):
    id2ent = read_id2item_dict(data_folder+'entities.dict')
    id2rel = read_id2item_dict(data_folder+'relations.dict')
    head_filter = read_rotate_results(result_folder + 'head_results_filter.txt')
    tail_filter = read_rotate_results(result_folder + 'tail_results_filter.txt')
    head_raw = read_rotate_results(result_folder + 'head_results_raw.txt')
    tail_raw = read_rotate_results(result_folder + 'tail_results_raw.txt')
    assert len(head_filter) == len(head_raw) == len(tail_filter) == len(tail_raw)
    file = open(out_file_path, 'w', encoding='utf-8')
    for tr, res in head_filter.items():
        h, r, t = tr[0], tr[1], tr[2]
        l_f_mr, l_f_mrr = res[0], res[1]

        res = head_raw[tr]
        l_r_mr, l_r_mrr = res[0], res[1]

        res = tail_filter[tr]
        r_f_mr, r_f_mrr = res[0], res[1]

        res = tail_raw[tr]
        r_r_mr, r_r_mrr = res[0], res[1]

        file.write(id2ent[h]+'\t'+id2rel[r]+'\t'+id2ent[t]+'\t'+l_f_mr+'\t'+l_r_mr+'\t'+l_f_mrr+'\t'+l_r_mrr+'\t' +
                   r_f_mr+'\t'+r_r_mr+'\t'+r_f_mrr+'\t'+r_r_mrr+'\n')
    file.close()


def run_tucker_results(result_folder, out_file_path):
    head_filter, tail_filter = read_tucker_results(result_folder + 'results_filter.txt')
    head_raw, tail_raw = read_tucker_results(result_folder + 'results_raw.txt')
    assert len(head_filter) == len(head_raw) == len(tail_filter) == len(tail_raw)
    file = open(out_file_path, 'w', encoding='utf-8')
    for tr, res in head_filter.items():
        h, r, t = tr[0], tr[1], tr[2]
        l_f_mr, l_f_mrr = res[0], res[1]

        res = head_raw[tr]
        l_r_mr, l_r_mrr = res[0], res[1]

        res = tail_filter[tr]
        r_f_mr, r_f_mrr = res[0], res[1]

        res = tail_raw[tr]
        r_r_mr, r_r_mrr = res[0], res[1]

        file.write(h+'\t'+r+'\t'+t+'\t'+l_f_mr+'\t'+l_r_mr+'\t'+l_f_mrr+'\t'+l_r_mrr+'\t' +
                   r_f_mr+'\t'+r_r_mr+'\t'+r_f_mrr+'\t'+r_r_mrr+'\n')
    file.close()


if __name__ == '__main__':
    run_rotate_results('RotatE/data/wn18rr/', 'RotatE/models/RotatE_wn18rr_0/',
                       'Models-detailed-results/RotatE-TuckER-detailed-results/test-RotatE-WN18RR.txt')
    run_rotate_results('RotatE/data/YAGO3-10/', 'RotatE/models/RotatE_YAGO3-10_0/',
                       'Models-detailed-results/RotatE-TuckER-detailed-results/test-RotatE-YAGO3-10.txt')
    # run_tucker_results('TuckER/data/YAGO3-10/',
    #                    'Models-detailed-results/RotatE-TuckER-detailed-results/test-TuckER-YAGO3-10.txt')
