"""저장소 관리자용 오프라인 테스트: pytest tests/

가짜 ONNX·합성 샘플(tests/make_fake_assets.py) + 실제 G1 자산으로 파이프라인과 체크 로직을 검사합니다.
실제 정책 성능은 검사하지 않습니다(강사 사전 점검 노트북에서 Colab으로 확인).
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path(os.environ.get("SONIC_LAB_FAKE", "/tmp/sonic_fake"))


@pytest.fixture(scope="session")
def lab():
    if not (CACHE / "models/model_encoder.onnx").exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "tests/make_fake_assets.py"), str(CACHE)], check=True)
    os.environ["SONIC_LAB_CACHE"] = str(CACHE)
    import sonic_lab
    return sonic_lab.start(quiet=True)


def load_buggy():
    spec = importlib.util.spec_from_file_location("bp", ROOT / "exercises/buggy_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Pipeline()


def test_contract(lab):
    c = lab.contract
    assert np.array_equal(c.isaaclab_to_mujoco[c.mujoco_to_isaaclab], np.arange(29))
    assert sum(s.stop - s.start for s in c.enc_slices.values()) == 1762
    assert lab.sim.substeps == 4


def test_reference_passes_checks(lab):
    from sonic_lab import check, reference as ref
    assert check.encoder_obs(lab, lambda m, f, r, a: lab.policy.encoder_obs(m, f, r, a, 2))
    assert check.decoder_obs(lab, ref.Policy.decoder_obs)
    assert check.action_to_target(lab, ref.action_to_target)
    assert check.observation(lab, lambda sim, prev: sim.observation(prev))


def test_buggy_pipeline_fails_every_component(lab):
    from sonic_lab import check
    p = load_buggy()
    assert not check.observation(lab, p.observation)
    assert not check.decoder_obs(lab, p.decoder_obs)
    assert not check.action_to_target(lab, p.action_to_target)


def test_run_physics_modes(lab):
    from sonic_lab import reference as ref
    for mode in ["reference_pd", 0, 2]:
        summary, arrays = ref.run_physics(lab.sim, lab.policy, lab.motion, mode, max_frames=20)
        assert summary["simulated_frames"] > 0 and arrays["qpos"].shape[1] == lab.sim.model.nq


def test_mirrored_sample_loads(lab):
    m = lab.load_motion(mirrored=True)
    assert m.local24.shape[1:] == (24, 3)


def test_mode0_check(lab):
    from sonic_lab import check
    assert check.encoder_obs_mode0(lab, lambda m, f, r, a: lab.policy.encoder_obs(m, f, r, a, 0))
    assert not check.encoder_obs_mode0(lab, lambda m, f, r, a: lab.policy.encoder_obs(m, f, r, a, 2))


def test_oracle_calls_are_counted(lab):
    from sonic_lab import check, reference as ref
    before = check.ORACLE_CALLS.get("invariant_tests", 0)

    def tests(p, lab):
        raise AssertionError("always")
    check.invariant_tests(lab, tests)
    assert check.ORACLE_CALLS["invariant_tests"] == before + 1
