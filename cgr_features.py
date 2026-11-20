import numpy as np
LUT=np.full(256,4,np.int64)
for i,c in enumerate('ACGT'): LUT[ord(c)]=i; LUT[ord(c.lower())]=i
def read(T,s):
    S=[];Y=[];cur=[]
    for l in open('data/nt/%s_%s.fna'%(T,s)):
        l=l.strip()
        if l.startswith('>'):
            if cur: S.append(''.join(cur)); cur=[]
            Y.append(int(l.rsplit('|',1)[1]))
        elif l: cur.append(l)
    if cur: S.append(''.join(cur))
    return S,np.array(Y)
LEV=[(5,1),(4,4),(3,16),(3,64),(2,0)]
def kmers(a,k):
    n=len(a)-k+1; km=np.zeros(n,np.int64); bad=np.zeros(n,bool)
    for j in range(k): km=km*4+np.minimum(a[j:j+n],3); bad|=a[j:j+n]>3
    return km,bad,n
def feat(q,L0,lev=None,W=None):
    a=LUT[np.frombuffer(q.encode(),np.uint8)]; o=[]
    for k,r in (lev or LEV):
        km,bad,n=kmers(a,k); R=r if r else L0; M=W[k][1] if W else 4**k
        v=np.zeros(R*M,np.float32)
        if n>0:
            pos=np.arange(n)[~bad]; km=km[~bad]; b=(pos*r//n) if r else np.minimum(pos,R-1)
            v+=np.bincount(b*M+(W[k][0][km] if W else km),minlength=R*M)
        o.append(v)
    return np.log1p(np.concatenate(o))
def dims(L0,lev=None,W=None): return [ (r if r else L0)*(W[k][1] if W else 4**k) for k,r in (lev or LEV) ]
def norm(X,Xt):
    mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd,(Xt-mu)/sd
