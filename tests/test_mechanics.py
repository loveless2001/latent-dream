import torch
import pytest

from kv_dreaming.cache import KVState, merge_pair, reduce_cache
from kv_dreaming.rope import rotate
from kv_dreaming.sampling import sample, uniforms


def test_rotary_round_trip_fractional():
    x = torch.randn(1, 2, 5, 8)
    p = [0, 1.5, 64, 501.25, 16000.75]
    torch.testing.assert_close(rotate(rotate(x, p, 1e6), p, 1e6, inverse=True), x,
                               atol=5e-7, rtol=2e-6)


@torch.inference_mode()
def test_full_forward_and_custom_logits(runtime):
    ids = [3, 8, 13, 6, 17, 24, 19]
    reference = runtime.model(torch.tensor([ids]), use_cache=False).logits[0]
    state = runtime.empty()
    actual = torch.stack([runtime.step(t, state) for t in ids])
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=2e-5)
    state.validate(finite=True)


@torch.inference_mode()
def test_eviction_matches_dense_masked_continuation(runtime):
    history, suffix = [3, 8, 13, 6, 17, 24, 19, 28, 41, 29], [11, 22, 34]
    state = runtime.prefill(token_ids=history)
    state.select([0,1,2,3,7,8,9])
    actual = torch.stack([runtime.step(t, state) for t in suffix])
    n, t = len(history + suffix), len(history)
    mask = torch.zeros(1, 1, n, n)
    mask.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1), -torch.inf)
    mask[0,0,t:,4:7] = -torch.inf
    reference = runtime.model(torch.tensor([history+suffix]), attention_mask=mask,
                               use_cache=False).logits[0,t:]
    torch.testing.assert_close(actual.softmax(-1), reference.softmax(-1), atol=1e-7, rtol=1e-5)


def scalar_state(n=20, bootstrap=4):
    x = torch.randn(1, 1, n, 8)
    return KVState([rotate(x, list(range(n)), 1e6)], [x.clone()],
                   list(map(float,range(n))), [1]*n, n, bootstrap)


def test_merge_weighted_fractional_position_and_protection():
    state = scalar_state()
    original = {p:state.keys[0][:,:,i].clone() for i,p in enumerate(state.positions) if p<=4 or p>=16}
    events = reduce_cache(state, policy="merge", budget=12, recent=4, theta=1e6)
    assert len(events)==8 and state.length==12 and sum(state.counts)==20
    assert events[0]["input_positions"] == [5.0,6.0]
    assert any(p != int(p) for p in state.positions)
    assert state.next_position==20
    assert events[0]["key_norms"][0]["merged_key_norms"]
    for p,k in original.items():
        torch.testing.assert_close(state.keys[0][:,:,state.positions.index(p)],k,atol=0,rtol=0)


def test_count_bias_preserves_duplicate_attention():
    k = torch.randn(1,1,1,8)
    v = torch.randn_like(k)
    other_k, other_v = torch.randn_like(k), torch.randn_like(k)
    state = KVState([torch.cat((k,k,other_k),2)], [torch.cat((v,v,other_v),2)],
                    [0.0,0.0,1.0],[1,1,1],2,1)
    q = torch.randn(1,1,1,8)
    old = (q @ state.keys[0].transpose(-1,-2) / 8**.5).softmax(-1) @ state.values[0]
    merge_pair(state,0,1e6)
    bias = torch.tensor(state.counts).float().log()
    new = (q @ state.keys[0].transpose(-1,-2) / 8**.5 + bias).softmax(-1) @ state.values[0]
    torch.testing.assert_close(old,new,atol=2e-7,rtol=1e-6)


@pytest.mark.parametrize("policy",["evict","merge"])
def test_repeated_reductions_bounded_and_bootstrap_retained(runtime, policy):
    state = runtime.empty(bootstrap_position=6)
    for i in range(80):
        runtime.step(3+i%40,state)
        reduce_cache(state,policy=policy,budget=16,recent=5,theta=1e6)
        assert state.length<=16
        if i>=6: assert 6.0 in state.positions
        assert all(state.counts[state.positions.index(float(p))]==1 for p in range(min(4,i+1)))
    state.validate(finite=True)


def test_sampler_mask_zero_uniform_and_stop():
    logits=torch.tensor([20.,0.,-2.,10.,4.])
    token,stats=sample(logits,0.0,[0,3],4)
    assert token==1 and stats["blocked_mass"]>.99
    for u in uniforms(42,100): assert sample(logits,u,[0,3],4)[0] not in {0,3}
    assert sample(torch.tensor([-100.,-100.,20.]),.5,[],2)[0]==2
    with pytest.raises(ValueError): sample(logits,.5,[4],4)
    with pytest.raises(FloatingPointError): sample(torch.tensor([0.,float('nan')]),.2,[],0)


def test_prefix_initialization_reproducibility(runtime):
    from kv_dreaming.seeding import CALIBRATION_SCHEMA, CALIBRATION_METHOD, initialize
    calibration={"schema":CALIBRATION_SCHEMA, "method":CALIBRATION_METHOD,
                 "key_means":[torch.zeros(1,2,1,8)]*2,
                 "key_stds":[torch.ones(1,2,1,8)]*2,
                 "value_means":[torch.zeros(1,2,1,8)]*2,
                 "value_stds":[torch.ones(1,2,1,8)]*2,
                 "embedding_mean":torch.zeros(32),"embedding_std":torch.ones(32)*.02}
    for kind in ("random","soft"):
        a=initialize(runtime,kind,seed=11,prefix_length=8,calibration=calibration)
        b=initialize(runtime,kind,seed=11,prefix_length=8,calibration=calibration)
        for x,y in zip(a.keys,b.keys): torch.testing.assert_close(x,y,atol=0,rtol=0)
        assert a.next_position==8 and a.bootstrap_position==8


def test_snapshot_restores_merged_state_and_continuation(runtime, tmp_path):
    from kv_dreaming.runner import RunConfig, snapshot, read_snapshot
    state=runtime.empty()
    for i in range(30):
        runtime.step(3+i%40,state)
        reduce_cache(state,policy="merge",budget=16,recent=5,theta=1e6)
    config=RunConfig(initial_state="empty",policy="merge",budget=16,recent=5)
    path=tmp_path/'snapshot.pt'
    snapshot(path,state=state,next_token=9,step=2,draws=[.1,.2,.3],tokens=[5,9],
             config=config,identity={"test":"fixed"},terminal=False)
    data,restored=read_snapshot(path,runtime=runtime,identity={"test":"fixed"})
    torch.testing.assert_close(runtime.step(9,state),runtime.step(data['next_token'],restored),atol=0,rtol=0)
    with pytest.raises(ValueError): read_snapshot(path,runtime=runtime,identity={"test":"wrong"})
