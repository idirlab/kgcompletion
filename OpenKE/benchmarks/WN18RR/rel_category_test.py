reltype={}
a,b,c,d=0,0,0,0
with open('reltype.txt','r') as f:
    for lines in f:
        r,type=lines[:-1].split('\t')
        reltype[r]=type
        if reltype[r] == '1-1':
            a += 1
        if reltype[r] == '1-n':
            b += 1
        if reltype[r] == 'n-1':
            c += 1
        if reltype[r] == 'n-n':
            d += 1
    print a
    print b
    print c
    print d
    print round(float(a) / 11,2)*100
    print round(float(b) / 11,2)*100
    print round(float(c) / 11,2)*100
    print round(float(d) / 11,2)*100
a,b,c,d=0,0,0,0
num=0
with open('test.txt','r') as f:
    for lines in f:
        num+=1
        h,r,t=lines.split('\t')
        if reltype[r]=='1-1':
            a+=1
        if reltype[r]=='1-n':
            b+=1
        if reltype[r]=='n-1':
            c+=1
        if reltype[r]=='n-n':
            d+=1
print a
print b
print c
print d
print float(a)/num
print float(b)/num
print float(c)/num
print float(d)/num
