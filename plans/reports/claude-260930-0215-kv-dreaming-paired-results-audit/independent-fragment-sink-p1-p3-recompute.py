"""Independent P1-P3 recompute (no production imports) for no-sink vs BOS-sink fragment runs."""
import json, os, hashlib
def loop(tokens):  # preregistered: first 128-window, stride 32, unique fraction < 0.12
    return any(len(set(tokens[n-128:n]))/128 < 0.12 for n in range(128, len(tokens)+1, 32))
def analyze(root, only_fragment=True):
    P1=P2=P3=0; n=0; gates=0
    for g in sorted(os.listdir(root)):
        if only_fragment and not g.startswith('fragment'): continue
        for lead in ('evict-leader','merge-leader'):
            d=f'{root}/{g}/{lead}'; n+=1
            toks=json.load(open(f'{d}/tokens.json')); steps=[json.loads(l) for l in open(f'{d}/steps.jsonl')]
            meta=json.load(open(f'{d}/manifest.json'))['initialization_metadata']
            fr=next((s['step'] for s in steps if s['events']['leader'] or s['events']['follower']), None)
            gates+=sum(1 for s in steps if s['pre_reduction_gate'] is not None and s['pre_reduction_gate']['max_abs_diff']==0.0)
            P1+= fr is None
            P2+= loop(toks)
            window = toks if fr is None else toks[:fr+1]   # samples through first-reduction step
            tri=set(zip(window, window[1:], window[2:]))
            elig={tuple(t) for f in meta['fragments'] for t in f['eligible_trigrams']}
            P3+= bool(tri & elig)
    return dict(directions=n, P1_no_reduction=P1, P2_loops=P2, P3_prebound_fragment_hit=P3, exact_gates=gates)
R='/home/lenovo/projects/plans/reports'
print('no-sink v1 :', analyze(f'{R}/260930-kv-dreaming-paired-results/raw'))
print('bos-sink v1:', analyze(f'{R}/260930-kv-dreaming-fragment-sink-results/raw'))
for f in ('comparison.json','consistency.json','recall.json'):
    print(f, hashlib.sha256(open(f'{R}/260930-kv-dreaming-fragment-sink-results/{f}','rb').read()).hexdigest()[:12])
