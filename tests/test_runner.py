import json
from types import SimpleNamespace

import pytest
import torch

from kv_dreaming.cache import KVState
from kv_dreaming.runner import RunConfig, run
from kv_dreaming.cli import matrix_plan


class Tokenizer:
    added_tokens_decoder = {}

    def decode(self, tokens, **kwargs):
        return " ".join(map(str,tokens))


class FakeRuntime:
    tokenizer = Tokenizer()
    theta = 1e6
    inv_freq = torch.ones(1)

    def __init__(self, stop=True, fail=False):
        self.model=SimpleNamespace(config=SimpleNamespace(bos_token_id=151643,
                                                         max_position_embeddings=32768))
        self.seen=[]
        self.stop=stop
        self.fail=fail

    def step(self, token, state):
        self.seen.append(token)
        if self.fail: raise FloatingPointError("injected failure")
        z=torch.zeros(1,1,1,2)
        state.keys[0]=torch.cat((state.keys[0],z),2)
        state.values[0]=torch.cat((state.values[0],z),2)
        state.positions.append(float(state.next_position))
        state.counts.append(1)
        state.next_position+=1
        logits=torch.full((151669,),-1000.)
        logits[151643 if self.stop else 3]=10
        return logits


def empty():
    return KVState([torch.empty(1,1,0,2)],[torch.empty(1,1,0,2)])


def test_bootstrap_is_processed_but_generated_eos_stops(tmp_path):
    r=FakeRuntime()
    out=tmp_path/'eos'
    result=run(r,empty(),RunConfig(initial_state='empty'),out,identity={'test':True})
    assert r.seen==[151643]
    assert result['stop_reason']=='eos' and result['generated_tokens']==1
    assert json.loads((out/'tokens.json').read_text())==[151643]


def test_hard_limit_and_raw_logging(tmp_path):
    r=FakeRuntime(stop=False)
    out=tmp_path/'limit'
    result=run(r,empty(),RunConfig(initial_state='empty',max_tokens=4,snapshot_every=2),
               out,identity={'test':True})
    assert result['stop_reason']=='token_limit' and len(r.seen)==4
    records=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
    assert [x['step'] for x in records]==[0,1,2,3]
    assert all('uniform' in x and 'blocked_mass' in x and 'raw_entropy' in x for x in records)
    assert (out/'snapshot-000002.pt').exists()
    with pytest.raises(FileExistsError):
        run(r,empty(),RunConfig(initial_state='empty'),out,identity={'test':True})


def test_numerical_failure_has_explicit_artifact(tmp_path):
    out=tmp_path/'failure'
    with pytest.raises(RuntimeError,match='injected failure'):
        run(FakeRuntime(fail=True),empty(),RunConfig(initial_state='empty'),out,identity={})
    summary=json.loads((out/'summary.json').read_text())
    assert summary['stop_reason']=='error' and summary['generated_tokens']==0


def test_plan_has_36_unique_conditions_without_duplicate_empty_seeds():
    plan=matrix_plan()
    assert plan['count']==36
    assert len({json.dumps(x,sort_keys=True) for x in plan['runs']})==36
    assert len([x for x in plan['runs'] if x['initial_state']=='empty'])==4
