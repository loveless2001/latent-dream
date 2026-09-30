"""Bounded teacher-forced mechanics against the actual pinned checkpoint."""

import argparse
import json
from pathlib import Path
import time
import torch

from kv_dreaming.runtime import QwenRuntime, BASE_MODEL, BASE_REVISION
from kv_dreaming.runner import source_hashes


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise SystemExit('output already exists')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    r=QwenRuntime.load(local_files_only=True)
    ids=r.tokenizer.encode('The small boat crossed the quiet lake. A lantern stood on the old wooden pier. '
                           'Beyond the trees, a narrow path climbed toward the village.',add_special_tokens=False)
    reference=r.model(torch.tensor([ids]),use_cache=False).logits[0].float()
    state=r.empty()
    actual=torch.stack([r.step(t,state) for t in ids])
    delta=(actual-reference).abs().max().item()
    probability_delta=(actual.softmax(-1)-reference.softmax(-1)).abs().max().item()
    torch.testing.assert_close(actual,reference,atol=2e-4,rtol=2e-4)
    history=ids[:24]
    suffix=ids[24:30]
    state=r.prefill(token_ids=history)
    state.select([i for i in range(24) if not 8<=i<16])
    compact=torch.stack([r.step(t,state) for t in suffix])
    n=24+len(suffix)
    mask=torch.zeros(1,1,n,n)
    mask.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
    mask[0,0,24:,8:16]=-torch.inf
    dense=r.model(torch.tensor([history+suffix]),attention_mask=mask,use_cache=False).logits[0,24:].float()
    evict_delta=(compact.softmax(-1)-dense.softmax(-1)).abs().max().item()
    torch.testing.assert_close(compact.softmax(-1),dense.softmax(-1),atol=2e-5,rtol=2e-4)
    report={'status':'PASS','model':BASE_MODEL,'revision':BASE_REVISION,'dtype':'float32',
            'device':'cpu','threads':4,'forward_tokens':len(ids),
            'max_logit_delta':delta,'max_probability_delta':probability_delta,
            'eviction_dense_masked_max_probability_delta':evict_delta,
            'eviction_argmax_equal':bool((compact.argmax(-1)==dense.argmax(-1)).all()),
            'source_hashes':source_hashes(),'elapsed_seconds':time.monotonic()-start}
    assert report['eviction_argmax_equal']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()

