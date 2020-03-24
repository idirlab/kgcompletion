import cPickle

reltype=['1-1','1-n','n-1','n-n']
num2entity={}
num2rel={}
with open('entity2id.txt','r') as f:
    lines=f.readlines()[1:]
    for line in lines:
        ent,num=line[:-1].split('\t')
        num2entity[num]=ent

with open('relation2id.txt','r') as f:
    lines=f.readlines()[1:]
    for line in lines:
        rel,num=line[:-1].split('\t')
        num2rel[num]=rel
one_one=[]
one_n=[]
n_one=[]
n_n=[]
for rel in reltype:
    with open ('%s.txt'%rel,'r')as f:
        lines=f.readlines()[1:]
    for line in lines:
        h,t,r=line[:-1].split(' ')

        rname=num2rel[r]
        if rel=='1-1':
            one_one.append(rname)
        if rel=='1-n':
            one_n.append(rname)
        if rel=='n-1':
            n_one.append(rname)
        if rel=='n-n':
            n_n.append(rname)

g=open('reltype.txt','w')
for i in set(one_one):
    g.write(i+'\t'+'1-1\n')
for i in set(one_n):
    g.write(i+'\t'+'1-n\n')
for i in set(n_one):
    g.write(i+'\t'+'n-1\n')
for i in set(n_n):
    g.write(i+'\t'+'n-n\n')
g.close()
g=open('reltype.txt','r')
lines=g.readlines()
s=open('sorted-reltype.txt','w')
for line in sorted(lines, key=lambda line: line.split('\t')[0],reverse=False):
    s.write(line)
if set(one_one) & set(one_n) == True:
    print 'no'
if set(one_one) & set(n_one) == True:
    print 'no'
if set(one_one) & set(n_n) == True:
    print 'no'
if set(one_n) & set(n_one) == True:
    print 'no'
if set(one_n) & set(n_n)== True:
    print 'no'
if set(n_one) & set(n_n)== True:
    print 'no'

