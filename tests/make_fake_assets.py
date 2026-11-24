"""오프라인 테스트용 가짜 자산 생성기 (저장소 관리자용).

Hugging Face에 접근할 수 없는 환경에서 파이프라인의 '기계적' 동작만 검사합니다.
- GitHub 자산(G1 MJCF, 정책 헤더, 시뮬레이터 설정)은 실제 고정 리비전에서 받습니다.
- ONNX는 입출력 이름·차원만 같은 작은 선형 모델, 샘플은 합성 걷기 데이터입니다.
- 실제 정책 성능은 Colab에서 notebooks/를 실행해 확인하세요.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonic_lab import assets as A  # noqa: E402

OBS_YAML = """observations:
  - {name: token_state, enabled: true}
  - {name: his_base_angular_velocity_10frame_step1, enabled: true}
  - {name: his_body_joint_positions_10frame_step1, enabled: true}
  - {name: his_body_joint_velocities_10frame_step1, enabled: true}
  - {name: his_last_actions_10frame_step1, enabled: true}
  - {name: his_gravity_dir_10frame_step1, enabled: true}
encoder:
  dimension: 64
  encoder_observations:
""" + "".join(f"    - {{name: {n}, enabled: true}}\n" for n in [
    "encoder_mode_4", "motion_joint_positions_10frame_step5", "motion_joint_velocities_10frame_step5",
    "motion_root_z_position_10frame_step5", "motion_root_z_position", "motion_anchor_orientation",
    "motion_anchor_orientation_10frame_step5", "motion_joint_positions_lowerbody_10frame_step5",
    "motion_joint_velocities_lowerbody_10frame_step5", "vr_3point_local_target", "vr_3point_local_orn_target",
    "smpl_joints_10frame_step1", "smpl_anchor_orientation_10frame_step1", "motion_joint_positions_wrists_10frame_step1"]) + """  encoder_modes:
    - {name: g1, mode_id: 0, required_observations: [encoder_mode_4, motion_joint_positions_10frame_step5, motion_joint_velocities_10frame_step5, motion_anchor_orientation_10frame_step5]}
    - {name: teleop, mode_id: 1, required_observations: [encoder_mode_4, motion_joint_positions_lowerbody_10frame_step5, motion_joint_velocities_lowerbody_10frame_step5, vr_3point_local_target, vr_3point_local_orn_target, motion_anchor_orientation]}
    - {name: smpl, mode_id: 2, required_observations: [encoder_mode_4, smpl_joints_10frame_step1, smpl_anchor_orientation_10frame_step1, motion_joint_positions_wrists_10frame_step1]}
"""


def linear_onnx(path, n_in, n_out, out_name, scale, seed):
    rng = np.random.default_rng(seed)
    w = numpy_helper.from_array((rng.normal(size=(n_in, n_out)) * scale).astype(np.float32), "W")
    node = helper.make_node("MatMul", ["obs_dict", "W"], [out_name])
    graph = helper.make_graph([node], "fake", [helper.make_tensor_value_info("obs_dict", TensorProto.FLOAT, [1, n_in])],
                              [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, n_out])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.save(model, path)


def fake_samples(root: Path, sample: str, default_mj):
    T_h, T_r = 2002, 1202
    t_h, t_r = np.arange(T_h) / 50, np.arange(T_r) / 30
    yaw_r = 0.05 * np.sin(t_r)
    root_rot = R.from_euler("z", yaw_r[:, None])
    pose_aa_r = np.zeros((T_r, 30, 3)); pose_aa_r[:, 0] = root_rot.as_rotvec()
    dof = np.tile(default_mj, (T_r, 1)) + 0.05 * np.sin(t_r)[:, None]
    robot = {"root_trans_offset": np.c_[0.3 * t_r, np.zeros(T_r), np.full(T_r, .79)], "pose_aa": pose_aa_r,
             "dof": dof, "root_rot": root_rot.as_quat(), "smpl_joints": np.zeros((T_r, 24, 3)), "fps": 30.0}
    rng = np.random.default_rng(0)
    base = rng.normal(0, .3, (24, 3)); base[15] = [0, .6, 0]
    pose_aa_h = np.zeros((T_h, 72)); pose_aa_h[:, :3] = R.from_euler("y", 0.05 * np.sin(t_h)[:, None]).as_rotvec()
    human = {"pose_aa": pose_aa_h, "transl": np.zeros((T_h, 3)), "fps": 50.0,
             "smpl_joints": base[None] + 0.02 * np.sin(t_h)[:, None, None],
             "original_pose_aa": pose_aa_h, "original_fps": 50.0}
    key = Path(sample).stem
    for rel, rec in [(Path("robot_filtered/210531") / sample, robot), (Path("smpl_filtered") / sample, human)]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({key: rec}, root / rel)


def main(root):
    root = Path(root)
    repo, models, samples = root / "official_source", root / "models", root / "sample_data"
    for rel in [A.HEADER_REL, A.SIM_CONFIG_REL, A.MODEL_REL / "scene_43dof.xml", A.MODEL_REL / "g1_29dof_with_hand.xml"]:
        A.download(f"{A.RAW}/{rel.as_posix()}", repo / rel)
    models.mkdir(parents=True, exist_ok=True)
    (models / "observation_config.yaml").write_text(OBS_YAML)
    linear_onnx(models / "model_encoder.onnx", 1762, 64, "encoded_tokens", 0.01, 0)
    linear_onnx(models / "model_decoder.onnx", 994, 29, "action", 0.0, 1)  # action=0 → 기본 자세 유지
    from sonic_lab import reference as ref
    a = A.Assets(root, repo, models, samples)
    c = ref.load_contract(a)
    for s in [A.SAMPLE, A.SAMPLE_MIRRORED]:
        fake_samples(samples, s, c.default)
    print("fake assets:", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sonic_fake")
