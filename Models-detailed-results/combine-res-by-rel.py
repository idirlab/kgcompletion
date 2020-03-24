def combine_res(dataset):
    filenames = ['res-by-rel-transe','res-by-rel-distmult','res-by-rel-complex','res-by-rel-conve']
    filenames+=['res-by-rel-rotate','res-by-rel-tucker','res-by-rel-amie']
    i=0
    for outfile,fname in zip(filenames,filenames) :
        with open('./models_res_by_rel/%s/%s-%s_sorted.txt'%(dataset,outfile,dataset), 'w') as k,open('./models_res_by_rel/%s/%s-%s.txt'%(dataset,fname,dataset),'r') as infile:
            lines = infile.readlines()[1:]
            print outfile
            print dataset
            print len(lines)
            for line in sorted(lines, key=lambda line: line.split('\t')[0],reverse=False):
                if i==0:
                    k.write(line)
                else:
                    g=line.split('\t')
                    newline='\t'.join(g[2:])
                    k.write(newline)
            k.close()
            i=i+1


    with open('./models_res_by_rel/result-by-rel-all/result-by-rel-all-%s.txt'%dataset, 'w') as writer:
        readers = [open('./models_res_by_rel/%s/%s-%s_sorted.txt'%(dataset,filename,dataset),'r') for filename in filenames]
        for lines in zip(*readers):
            newline='\t'.join([line.strip('\n') for line in lines])
            writer.write(newline+'\n')

combine_res('FB15k-237')
combine_res('WN18RR')
combine_res('YAGO3-10')