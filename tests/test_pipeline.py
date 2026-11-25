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


def test_common_mistakes_are_detected(lab):
    """Lab 2의 '흔한 실수'는 자가 확인에서 ❌가 나와야 합니다."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from sonic_lab import check
    c = lab.contract
    assert not check.observation(lab, lambda sim, prev: dict(sim.observation(prev), q=sim.data.qpos[sim.qaddr][c.mujoco_to_isaaclab]))
    assert not check.observation(lab, lambda sim, prev: dict(sim.observation(prev), gravity=R.from_quat(sim.data.qpos[3:7]).inv().apply([0, 0, -1])))
    assert not check.decoder_obs(lab, lambda token, history: __import__("sonic_lab").reference.Policy.decoder_obs(token, list(reversed(history))))
    assert not check.action_to_target(lab, lambda a, c: c.default + np.asarray(a)[c.mujoco_to_isaaclab] * c.scale)


def test_run_physics_modes(lab):
    from sonic_lab import reference as ref
    for mode in ["reference_pd", 0, 2]:
        summary, arrays = ref.run_physics(lab.sim, lab.policy, lab.motion, mode, max_frames=20)
        assert summary["simulated_frames"] > 0 and arrays["qpos"].shape[1] == lab.sim.model.nq


def test_mirrored_sample_loads(lab):
    m = lab.load_motion(mirrored=True)
    assert m.local24.shape[1:] == (24, 3)


def test_lab1_checks(lab):
    from sonic_lab import check
    assert check.onnx_io(1762, 64, 994, 29) and not check.onnx_io(1762, 64, 436, 29)
    assert check.root_quat_order("xyzw") and not check.root_quat_order("wxyz")
    assert check.encoder_obs(lab, lambda m, f, r, a: lab.policy.encoder_obs(m, f, r, a, 0), mode=0)


def test_body_groups():
    from sonic_lab import check
    assert check.body_groups({"다리": range(0, 12), "허리": range(12, 15), "팔·손목": range(15, 29)})
    assert not check.body_groups({"다리": range(0, 13), "허리": range(13, 15), "팔·손목": range(15, 29)})
    assert not check.body_groups({"다리": range(0, 12), "허리": range(12, 15), "팔·손목": range(15, 28)})
