"""TODO 자가 확인(self-check).

각 TODO 셀 끝에서 호출되어, 채운 값/함수를 참조 구현(`sonic_lab.reference`)과 비교하고 ✅/❌를 출력합니다.
평가용이 아니라 "다음 셀로 넘어가도 되는지" 알려 주는 용도입니다.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import reference as ref

OK, NG = "✅", "❌"


def _say(ok, title, detail=""):
    print(f"{OK if ok else NG} {title}" + (f" — {detail}" if detail else ""))
    return ok


def _as_vec(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _diagnose(student, expected, atol=1e-5):
    """두 벡터가 어떻게 다른지 짧게 설명합니다."""
    s, e = _as_vec(student), _as_vec(expected)
    if s.shape != e.shape:
        return f"길이 {s.size} ≠ 기대 {e.size}"
    if np.allclose(s, e, atol=atol):
        return ""
    if np.allclose(np.sort(s), np.sort(e), atol=atol):
        return "값의 집합은 같지만 배치(순서)가 다릅니다"
    if np.allclose(s, -e, atol=atol):
        return "부호가 반대입니다"
    err = np.abs(s - e)
    return f"최대 오차 {err.max():.4g}, 불일치 원소 {(err > atol).sum()}/{e.size}개"


def _random_pose(rng):
    return R.from_euler("zyx", [rng.uniform(-np.pi, np.pi), rng.normal(0, .05), rng.normal(0, .05)]).as_quat()


# ---------------------------------------------------------------------------
# Lab 1
# ---------------------------------------------------------------------------

def onnx_io(enc_in, enc_out, dec_in, dec_out, title="ONNX 입출력 차원"):
    got = tuple(int(x) for x in (enc_in, enc_out, dec_in, dec_out))
    want = (1762, 64, 994, 29)
    return _say(got == want, title, "" if got == want else f"입력값 {got} — 위에 출력된 shape의 마지막 숫자를 다시 보세요")


def encoder_slices(lab, slices, title="Encoder 1762칸 블록 표"):
    """slices: {관측 이름: slice 또는 (start, stop)}"""
    expected = lab.contract.enc_slices
    ok = list(slices) == list(expected)
    if not ok:
        _say(False, "  블록 순서", "YAML의 enabled 관측 순서와 다릅니다")
    for name, exp in expected.items():
        got = slices.get(name)
        if got is None:
            continue
        start, stop = (got.start, got.stop) if isinstance(got, slice) else tuple(got)
        if (start, stop) != (exp.start, exp.stop):
            ok = _say(False, f"  {name}", f"[{start}:{stop}] — 이 블록의 크기를 다시 계산하세요")
    return _say(ok, title)


def root_quat_order(answer: str, title="G1 root_rot 쿼터니언 순서"):
    ok = str(answer).lower().strip() == "xyzw"
    return _say(ok, title, "" if ok else "오차가 ~0°인 쪽이 정답입니다")


def resampled_motion(lab, times, g1_q50, title="30 Hz → 50 Hz 정렬"):
    m = lab.motion
    times, g1_q50 = np.asarray(times), np.asarray(g1_q50)
    if times.shape != m.times.shape:
        return _say(False, title, f"프레임 수 {len(times)} (기대 {len(m.times)}) — 두 데이터가 모두 있는 구간(짧은 쪽)을 쓰세요")
    detail = _diagnose(g1_q50, m.g1_q, atol=1e-6)
    return _say(not detail, title, detail)


def encoder_obs(lab, fn, frames=(0, 137, 1000, 1995), seed=0, mode=2, title=None):
    """fn(motion, frame, robot_xyzw, alignment) -> (1762,)"""
    title = title or f"Mode-{mode} Encoder 입력 1762칸"
    rng = np.random.default_rng(seed)
    ok = True
    for frame in frames:
        robot = _random_pose(rng)
        alignment = R.from_euler("z", rng.uniform(-np.pi, np.pi))
        expected = lab.policy.encoder_obs(lab.motion, frame, robot, alignment, mode=mode)
        got = _as_vec(fn(lab.motion, frame, robot, alignment))
        if got.shape != expected.shape:
            return _say(False, title, f"길이 {got.size} ≠ 1762")
        for name, sl in lab.contract.enc_slices.items():
            detail = _diagnose(got[sl], expected[sl])
            if detail:
                ok = _say(False, f"  frame {frame} · {name}", detail)
    return _say(ok, title)


# ---------------------------------------------------------------------------
# Lab 2
# ---------------------------------------------------------------------------

def observation(lab, fn, seed=3, trials=4, title="로봇 관측(proprioception)"):
    """fn(sim, previous_action) -> dict(omega, q, dq, action, gravity)"""
    import mujoco
    rng = np.random.default_rng(seed)
    sim, ok, details = lab.sim, True, {}
    for _ in range(trials):
        sim.reset(lab.motion)
        sim.data.qpos[3:7] = np.roll(_random_pose(rng), 1)  # wxyz
        sim.data.qpos[sim.qaddr] += rng.normal(0, .2, 29)
        sim.data.qvel[:] = rng.normal(0, .5, sim.model.nv)
        mujoco.mj_forward(sim.model, sim.data)
        prev = rng.normal(size=29)
        expected = sim.observation(prev)
        got = fn(sim, prev.copy())
        for key in ref.HISTORY_KEYS:
            detail = "키 없음" if key not in got else _diagnose(got[key], expected[key])
            if detail:
                details[key] = detail
    sim.reset(lab.motion)
    for key, detail in details.items():
        ok = _say(False, f"  {key}", detail)
    return _say(ok, title)


def decoder_obs(lab, fn, seed=1, title="Decoder 입력 994칸"):
    """fn(token, history) -> (994,). history는 append로 쌓인 길이 10 목록(마지막이 최신)."""
    rng = np.random.default_rng(seed)
    token = rng.normal(size=64).astype(np.float32)
    history = [{"omega": rng.normal(size=3), "q": rng.normal(size=29), "dq": rng.normal(size=29),
                "action": rng.normal(size=29), "gravity": rng.normal(size=3)} for _ in range(10)]
    expected = ref.Policy.decoder_obs(token, history)
    got = _as_vec(fn(token, [dict(h) for h in history]))
    if got.shape != expected.shape:
        return _say(False, title, f"길이 {got.size} ≠ 994")
    ok, start = True, 0
    for name, size in ref.DEC_DIMS.items():
        detail = _diagnose(got[start:start + size], expected[start:start + size])
        if detail:
            ok = _say(False, f"  {name}", detail)
        start += size
    return _say(ok, title)


def action_to_target(lab, fn, seed=2, title="액션 → 관절 목표각"):
    """fn(action_il, contract) -> (29,) MuJoCo 순서 목표각"""
    rng = np.random.default_rng(seed)
    action = rng.normal(size=29)
    detail = _diagnose(fn(action.copy(), lab.contract), ref.action_to_target(action, lab.contract))
    zero = _diagnose(fn(np.zeros(29), lab.contract), lab.contract.default)
    if zero:
        detail = (detail + " / " if detail else "") + "action=0일 때 기본 자세가 나오지 않습니다"
    return _say(not detail, title, detail)


# ---------------------------------------------------------------------------
# Lab 3
# ---------------------------------------------------------------------------

def body_groups(groups: dict, title="부위별 관절 묶음"):
    """{이름: range} — MuJoCo 순서 29관절을 다리 12 / 허리 3 / 팔·손목 14로 빠짐없이 나눴는지."""
    try:
        idx = sorted(i for r in groups.values() for i in r)
    except TypeError:
        return _say(False, title, "range(시작, 끝)를 모두 채우세요")
    sizes = [len(r) for r in groups.values()]
    names = [n.replace("_joint", "") for n in ref.MJ_NAMES]
    if idx != list(range(29)):
        return _say(False, title, "0–28번 관절이 빠짐없이 한 번씩 들어가야 합니다")
    ok = sizes == [12, 3, 14]
    detail = "" if ok else f"묶음 크기 {sizes} (기대 [12, 3, 14]) — 계약서 2절 MuJoCo 순서표를 보세요"
    if ok:
        for (name, r) in groups.items():
            print(f"   {name}: {names[r.start]} … {names[r.stop - 1]}")
    return _say(ok, title, detail)
