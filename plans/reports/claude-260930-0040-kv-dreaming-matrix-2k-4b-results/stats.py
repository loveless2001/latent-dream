import json, zlib, re, os
from collections import Counter
rows=[]
for n in open('names.txt').read().split():
    t=open(f'{n}/text.txt',encoding='utf-8',errors='replace').read()
    tok=json.load(open(f'{n}/tokens.json'))
    s=json.load(open(f'{n}/summary.json'))
    b=t.encode(); gz=len(zlib.compress(b,9))/max(1,len(b))
    bi=list(zip(tok,tok[1:])); d2=len(set(bi))/max(1,len(bi))
    # loop onset: first index after which last-256-token window distinct ratio < 0.15
    onset=None
    for i in range(128,len(tok),32):
        w=tok[max(0,i-128):i]
        if len(set(w))/len(w)<0.12: onset=i; break
    letters=sum(c.isalpha() and c.isascii() for c in t)/max(1,len(t))
    cjk=sum('一'<=c<='鿿' for c in t)/max(1,len(t))
    fp=len(re.findall(r"\b(I|me|my|I'm)\b",t))
    kind,pol=n.split('-')[0],n.split('-')[1]
    rows.append(dict(n=n,kind=kind,pol=pol,stop=s['stop_reason'],tok=len(tok),chars=len(t),gz=round(gz,3),d2=round(d2,3),loop_onset=onset,ascii_letters=round(letters,2),cjk=round(cjk,2),first_person=fp))
for r in rows: print(r)
import statistics as st
print('\nBY START (both policies):')
for k in ('empty','random','soft'):
    g=[r for r in rows if r['kind']==k]
    print(k,'n',len(g),'median tokens',st.median(r['tok'] for r in g),'median gz',st.median(r['gz'] for r in g),'median d2',st.median(r['d2'] for r in g),
          'looped',sum(r['loop_onset'] is not None for r in g),'eos',sum(r['stop']=='eos' for r in g),'first-person total',sum(r['first_person'] for r in g))
json.dump(rows,open('stats.json','w'),indent=1)
