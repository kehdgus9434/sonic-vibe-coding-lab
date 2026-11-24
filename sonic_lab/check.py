"""미션 체크포인트.

학생이 바이브 코딩으로 작성한 함수를 참조 구현(`sonic_lab.reference`)과 수치 비교합니다.
불일치하면 '어느 블록이 어떤 식으로' 다른지만 알려주고, 고치는 방법은 알려주지 않습니다.
(이 파일과 reference.py를 열어 정답을 베끼는 것은 실습 취지에 맞지 않습니다.)
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import reference as ref

OK, NG = "✅", "❌"
ORACLE_CALLS = {}   # Day 2: 정답 비교(오라클) 호출 횟수


def _count(name):
    ORACLE_CALLS[name] = ORACLE_CALLS.get(name, 0) + 1
    print(f"🔎 정답 비교 '{name}' {ORACLE_CALLS[name]}회째")


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


# ---------------------------------------------------------------------------
# Day 1
# ---------------------------------------------------------------------------

def onnx_io(enc_in, enc_out, dec_in, dec_out):
    """Encoder/Decoder 입력·출력의 마지막 차원."""
    got = tuple(int(x) for x in (enc_in, enc_out, dec_in, dec_out))
    want = (1762, 64, 994, 29)
    return _say(got == want, "Mission 1-1 ONNX 입출력 차원", "" if got == want else f"입력값 {got} — 세션의 get_inputs()/get_outputs()를 확인하세요")


def encoder_slices(lab, slices):
    """{관측 이름: slice 또는 (start, stop)} 형태."""
    expected = lab.contract.enc_slices
    ok = True
    names = list(slices)
    if names != list(expected):
        ok = _say(False, "Mission 1-2 Encoder 블록 순서", "이름 목록/순서가 YAML의 enabled 관측 순서와 다릅니다")
    for name, exp in expected.items():
        got = slices.get(name)
        if got is None:
            continue
        start, stop = (got.start, got.stop) if isinstance(got, slice) else tuple(got)
        if (start, stop) != (exp.start, exp.stop):
            ok = _say(False, f"  {name}", f"[{start}:{stop}] — 이 블록의 길이 또는 시작 위치를 다시 계산하세요")
    total = max((s.stop if isinstance(s, slice) else s[1]) for s in slices.values()) if slices else 0
    return _say(ok and total == 1762, "Mission 1-2 Encoder 1762차원 블록 표", f"총 길이 {total}")


def root_quat_order(answer: str):
    ok = str(answer).lower().strip() == "xyzw"
    return _say(ok, "Mission 1-3 G1 root_rot 쿼터니언 순서",
                "pose_aa[:,0]와 비교해 오차가 ~0인 쪽을 고르세요" if not ok else "")


def resampled_motion(lab, times, g1_q50):
    m = lab.motion
    times, g1_q50 = np.asarray(times), np.asarray(g1_q50)
    ok_t = _say(times.shape == m.times.shape and np.allclose(times, m.times), "Mission 1-4 50 Hz 시간축",
                f"길이 {len(times)} (기대 {len(m.times)}) — 두 파일의 fps와 길이가 다릅니다. 짧은 쪽에 맞추세요" if times.shape != m.times.shape else "")
    if not ok_t:
        return False
    detail = _diagnose(g1_q50, m.g1_q, atol=1e-6)
    return _say(not detail, "Mission 1-4 G1 관절각 30→50 Hz 보간", detail)


def _random_pose(rng):
    return R.from_euler("zyx", [rng.uniform(-np.pi, np.pi), rng.normal(0, .05), rng.normal(0, .05)]).as_quat()


def encoder_obs(lab, fn, frames=(0, 137, 1000, 1995), seed=0):
    """fn(motion, frame, robot_xyzw, alignment) -> (1762,) Mode-2 encoder 입력."""
    rng = np.random.default_rng(seed)
    ok = True
    for frame in frames:
        robot = _random_pose(rng)
        alignment = R.from_euler("z", rng.uniform(-np.pi, np.pi))
        expected = lab.policy.encoder_obs(lab.motion, frame, robot, alignment, mode=2)
        got = _as_vec(fn(lab.motion, frame, robot, alignment))
        if got.shape != expected.shape:
            return _say(False, "Mission 1-5 Encoder 입력", f"길이 {got.size} ≠ 1762")
        for name, sl in lab.contract.enc_slices.items():
            detail = _diagnose(got[sl], expected[sl])
            if detail:
                ok = _say(False, f"  frame {frame} · {name}", detail)
    if ok:
        token = lab.policy.encode_obs(fn(lab.motion, 0, lab.sim.base_xyzw(), R.identity()))
        print(f"   토큰 미리보기(앞 6개): {np.round(token[:6], 3)}")
    return _say(ok, "Mission 1-5 Mode-2 Encoder 입력 1762차원")


def encoder_obs_mode0(lab, fn, frames=(0, 137, 1000, 1995), seed=4):
    """[심화] fn(motion, frame, robot_xyzw, alignment) -> (1762,) Mode-0(G1 참조) encoder 입력."""
    rng = np.random.default_rng(seed)
    ok = True
    for frame in frames:
        robot = _random_pose(rng)
        alignment = R.from_euler("z", rng.uniform(-np.pi, np.pi))
        expected = lab.policy.encoder_obs(lab.motion, frame, robot, alignment, mode=0)
        got = _as_vec(fn(lab.motion, frame, robot, alignment))
        if got.shape != expected.shape:
            return _say(False, "심화 Mode-0 Encoder 입력", f"길이 {got.size} ≠ 1762")
        for name, sl in lab.contract.enc_slices.items():
            detail = _diagnose(got[sl], expected[sl])
            if detail:
                ok = _say(False, f"  frame {frame} · {name}", detail)
    return _say(ok, "심화 Mode-0 Encoder 입력 1762차원")


# ---------------------------------------------------------------------------
# Day 2
# ---------------------------------------------------------------------------

def _random_history(rng):
    return [{"omega": rng.normal(size=3), "q": rng.normal(size=29), "dq": rng.normal(size=29),
             "action": rng.normal(size=29), "gravity": rng.normal(size=3)} for _ in range(10)]


def decoder_obs(lab, fn, seed=1):
    """fn(token, history) -> (994,). history는 오래된 것 → 최신 순서의 10개 dict."""
    rng = np.random.default_rng(seed)
    token, history = rng.normal(size=64).astype(np.float32), _random_history(rng)
    expected = ref.Policy.decoder_obs(token, history)
    got = _as_vec(fn(token, [dict(h) for h in history]))
    if got.shape != expected.shape:
        return _say(False, "Decoder 입력 994차원", f"길이 {got.size} ≠ 994")
    ok, start = True, 0
    for name, size in ref.DEC_DIMS.items():
        detail = _diagnose(got[start:start + size], expected[start:start + size])
        if detail:
            ok = _say(False, f"  {name}", detail)
        start += size
    return _say(ok, "Decoder 입력 994차원")


def action_to_target(lab, fn, seed=2):
    """fn(action_il, contract) -> (29,) MuJoCo 순서 PD 목표 관절각."""
    rng = np.random.default_rng(seed)
    action = rng.normal(size=29)
    detail = _diagnose(fn(action.copy(), lab.contract), ref.action_to_target(action, lab.contract))
    zero = _diagnose(fn(np.zeros(29), lab.contract), lab.contract.default)
    if zero:
        detail = (detail + " / " if detail else "") + "action=0일 때 기본 자세가 나오지 않습니다"
    return _say(not detail, "액션 → PD 목표 관절각", detail)


def observation(lab, fn, seed=3, trials=4):
    """fn(sim, previous_action) -> dict(omega, q, dq, action, gravity)."""
    import mujoco
    rng = np.random.default_rng(seed)
    sim, ok = lab.sim, True
    diffs, details = {}, {}
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
            if key not in got:
                sim.reset(lab.motion)
                return _say(False, "로봇 관측", f"키 '{key}' 없음")
            detail = _diagnose(got[key], expected[key])
            if detail:
                details[key] = detail
                diffs.setdefault(key, []).append(_as_vec(got[key]) - _as_vec(expected[key]))
    for key, detail in details.items():
        d = np.asarray(diffs[key])
        if len(d) == trials and np.allclose(d, d[0], atol=1e-6):
            detail += " · 상태와 무관한 '고정 오프셋'만큼 다릅니다"
        ok = _say(False, f"  {key}", detail)
    sim.reset(lab.motion)
    return _say(ok, "로봇 관측(proprioception)")


def bughunt(lab, pipeline, frames=250):
    """[오라클] exercises/buggy_pipeline.py의 Pipeline을 정답과 비교하고 짧은 폐루프를 실행합니다.

    Day 2에서는 Mission 2-3(최종 판정)에서만 사용합니다. 호출 횟수는 기록됩니다.
    """
    _count("bughunt")
    print("── 구성요소 검사 ──")
    results = {
        "observation": observation(lab, pipeline.observation),
        "decoder_obs": decoder_obs(lab, pipeline.decoder_obs),
        "action_to_target": action_to_target(lab, pipeline.action_to_target),
    }
    print(f"── 폐루프 {frames}프레임({frames / 50:.0f}초) 실행 ──")
    summary, _ = ref.run_physics(lab.sim, lab.policy, lab.motion, 2, max_frames=frames, pipeline=pipeline)
    _say(summary["completed"], "Human 모드 2 폐루프",
         f'{summary["simulated_frames"]}/{frames} 프레임, {summary["stop_reason"]}')
    fixed = sum(results.values())
    print(f"\n구성요소 {fixed}/3 통과. (버그는 모두 4개이며 한 구성요소에 2개가 있을 수 있습니다)")
    return summary


class _Correct:
    def observation(self, sim, previous_action):
        return sim.observation(previous_action)

    def decoder_obs(self, token, history):
        return ref.Policy.decoder_obs(token, history)

    def action_to_target(self, action, contract):
        return ref.action_to_target(action, contract)


class _M1(_Correct):
    def observation(self, sim, previous_action):
        o = sim.observation(previous_action)
        o["q"] = sim.data.qpos[sim.qaddr][sim.contract.mujoco_to_isaaclab]
        return o


class _M2(_Correct):
    def observation(self, sim, previous_action):
        o = sim.observation(previous_action)
        o["gravity"] = R.from_quat(sim.data.qpos[3:7]).inv().apply([0, 0, -1])
        return o


class _M3(_Correct):
    def decoder_obs(self, token, history):
        return ref.Policy.decoder_obs(token, list(reversed(list(history))))


class _M4(_Correct):
    def action_to_target(self, action, contract):
        return contract.default + np.asarray(action)[contract.mujoco_to_isaaclab] * contract.scale


def invariant_tests(lab, test_fn):
    """[오라클] test_fn(pipeline, lab): 올바른 파이프라인은 통과, 버그 변형 4종은 AssertionError여야 합니다."""
    _count("invariant_tests")
    try:
        test_fn(_Correct(), lab)
        ok = _say(True, "올바른 파이프라인에서 통과")
    except Exception as error:  # noqa: BLE001
        lab.sim.reset(lab.motion)
        return _say(False, "올바른 파이프라인에서도 실패", f"{type(error).__name__}: {error} — 테스트 자체가 틀렸습니다")
    caught = 0
    for i, mutant in enumerate([_M1, _M2, _M3, _M4], 1):
        try:
            test_fn(mutant(), lab)
            print(f"   버그 유형 {i}: 놓침")
        except AssertionError:
            caught += 1
            print(f"   버그 유형 {i}: 잡음")
        except Exception as error:  # noqa: BLE001
            print(f"   버그 유형 {i}: 테스트가 AssertionError 대신 {type(error).__name__}로 중단")
        lab.sim.reset(lab.motion)
    return _say(ok and caught == 4, f"불변식 테스트: 버그 {caught}/4 검출")


# ---------------------------------------------------------------------------
# Day 3
# ---------------------------------------------------------------------------

def experiment_card(card: dict):
    need = ["가설", "변인", "지표", "통제조건", "결과", "해석", "한계"]
    missing = [k for k in need if not str(card.get(k, "")).strip()]
    return _say(not missing, "실험 카드", "빈 항목: " + ", ".join(missing) if missing else "")
