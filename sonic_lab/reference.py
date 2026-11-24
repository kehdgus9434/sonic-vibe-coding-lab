"""SONIC 참조 구현 (정답 파이프라인).

원본: AI인력양성사업 교육자료 07번 노트북(`07_ETRI_11-04_SONIC_G1_LocalCode_Aligned_CPU.ipynb`)의
`sonic_etri.py`. Colab CPU 런타임에서 공식 G1 모드 0/Human 모드 2 정책이 2,002프레임을
완주한 구현을 교육용으로 정리했습니다. 수식과 순서 규칙은 바꾸지 않았습니다.

- ONNX Runtime CPU provider만 사용합니다(로봇·DDS·ZMQ 연결 없음).
- 모든 공식 자산은 고정 리비전에서 받습니다(`sonic_lab.assets`).
- 실물 로봇에 이 코드의 출력을 보내지 마세요.
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import time

import joblib
import mujoco
import numpy as np
import onnxruntime as ort
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import yaml

from .assets import Assets, HEADER_REL, MODEL_REL, SIM_CONFIG_REL, SAMPLE

POLICY_HZ = 50
SMPL_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
MJ_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# 공식 observation_config.yaml의 순서. YAML 주석의 "436"은 오래된 값이며 실제 ONNX 입력은 1762/994입니다.
ENC_DIMS = {
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}
DEC_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}
HISTORY_KEYS = ["omega", "q", "dq", "action", "gravity"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. 입출력 계약: 공식 C++ 헤더 + observation_config.yaml
# ---------------------------------------------------------------------------

def _arithmetic(expression, constants):
    """공식 C++ 상수식을 eval 없이 읽습니다."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError(f"Unsupported expression in policy header: {expression}")
    return visit(ast.parse(expression.strip(), mode="eval"))


@dataclass
class Contract:
    isaaclab_to_mujoco: np.ndarray   # a_mj = a_il[isaaclab_to_mujoco]  (공식 헤더 배열 이름 그대로)
    mujoco_to_isaaclab: np.ndarray   # x_il = x_mj[mujoco_to_isaaclab]
    default: np.ndarray    # MuJoCo 순서 기본 관절각(rad)
    scale: np.ndarray      # MuJoCo 순서 액션 스케일
    kp: np.ndarray
    kd: np.ndarray
    enc_slices: dict
    mode_fields: dict
    metadata: dict = field(default_factory=dict)


def load_contract(assets: Assets) -> Contract:
    header = assets.repo / HEADER_REL
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", header.read_text(), flags=re.S)
    constants = {}
    for name, expr in re.findall(r"const double\s+(\w+)\s*=\s*([^;]+);", source):
        constants[name] = _arithmetic(expr, constants)

    def array(name, dtype=float):
        match = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{([^}]+)\}", source)
        if not match:
            raise ValueError(f"Missing array in official header: {name}")
        return np.asarray([_arithmetic(x, constants) for x in match[1].split(",") if x.strip()], dtype=dtype)

    isaaclab_to_mujoco, mujoco_to_isaaclab = array("isaaclab_to_mujoco", int), array("mujoco_to_isaaclab", int)
    if not np.array_equal(isaaclab_to_mujoco[mujoco_to_isaaclab], np.arange(29)):
        raise ValueError("Official joint mappings are not inverse permutations")
    config = yaml.safe_load((assets.model_dir / "observation_config.yaml").read_text())
    enc_names = [x["name"] for x in config["encoder"]["encoder_observations"] if x.get("enabled", True)]
    dec_names = [x["name"] for x in config["observations"] if x.get("enabled", True)]
    if enc_names != list(ENC_DIMS) or dec_names != list(DEC_DIMS):
        raise ValueError("Observation YAML differs from the verified 1762/994-dimensional contract")
    slices, start = {}, 0
    for name, size in ENC_DIMS.items():
        slices[name] = slice(start, start + size)
        start += size
    modes = {int(x["mode_id"]): set(x["required_observations"]) for x in config["encoder"]["encoder_modes"]}
    if start != 1762 or sum(DEC_DIMS.values()) != 994:
        raise ValueError("Invalid observation sizes")
    metadata = {"policy_header_sha256": sha256(header),
                "observation_config_sha256": sha256(assets.model_dir / "observation_config.yaml"),
                "encoder_blocks": {k: [v.start, v.stop] for k, v in slices.items()},
                "history_order": "oldest_to_newest", "action_order": "IsaacLab",
                "joint_order_output": "MuJoCo_named_29"}
    result = Contract(isaaclab_to_mujoco, mujoco_to_isaaclab, array("default_angles"), array("g1_action_scale"),
                      array("kps"), array("kds"), slices, modes, metadata)
    if any(len(getattr(result, key)) != 29 for key in ["default", "scale", "kp", "kd"]):
        raise ValueError("Policy parameters must contain exactly 29 joints")
    return result


# ---------------------------------------------------------------------------
# 2. 모션 데이터
# ---------------------------------------------------------------------------

def resample(values, source_hz, times):
    source_times = np.arange(len(values), dtype=float) / source_hz
    return interp1d(source_times, values, axis=0, bounds_error=True)(times)


def resample_rotation(rotation, source_hz, times):
    source_times = np.arange(len(rotation), dtype=float) / source_hz
    return Slerp(source_times, rotation)(np.clip(times, 0, source_times[-1]))


def yaw(rotation):
    matrix = rotation.as_matrix()
    return np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])


@dataclass
class Motion:
    name: str
    times: np.ndarray            # (T,) 50 Hz 시간축
    local24: np.ndarray          # (T,24,3) 루트 기준 SMPL 관절 좌표
    human_root_xyzw: np.ndarray  # (T,4) 인체 루트 회전(scipy xyzw)
    wrists_il: np.ndarray        # (T,6) G1 손목 6관절(IsaacLab 순서 23:29)
    metadata: dict
    g1_q: np.ndarray | None = None          # (T,29) MuJoCo 순서 G1 참조 관절각
    g1_dq: np.ndarray | None = None
    g1_root: np.ndarray | None = None       # (T,3)
    g1_root_xyzw: np.ndarray | None = None  # (T,4)
    world24: np.ndarray | None = None


def load_published_pickle(path):
    """NVIDIA 공식 샘플(pickle) 전용. 신뢰할 수 없는 pickle에는 쓰지 마세요."""
    record = joblib.load(path)
    if not any(key in record for key in ["dof", "smpl_joints"]):
        records = [value for value in record.values() if isinstance(value, dict)]
        if len(records) != 1:
            raise ValueError("Ambiguous official pickle motion")
        record = records[0]
    return record


def detect_root_quat_order(robot_record):
    """공식 G1 `root_rot`의 성분 순서를 `pose_aa[:,0]`(축-각)과 비교해 판정합니다."""
    robot_rotation = R.from_rotvec(np.asarray(robot_record["pose_aa"])[:, 0])
    raw = np.asarray(robot_record["root_rot"])
    errors = {order: float(np.max((robot_rotation.inv() * R.from_quat(q)).magnitude()))
              for order, q in [("xyzw", raw), ("wxyz", raw[:, [1, 2, 3, 0]])]}
    matching = [key for key, value in errors.items() if value < 1e-4]
    if len(matching) != 1:
        raise ValueError(f"root_rot order is ambiguous/inconsistent with pose_aa: {errors}")
    return matching[0], errors


def human_root_rotation(human_record):
    """학습 코드 규칙: Y-up→Z-up 변환 후 SMPL 기본 회전 제거."""
    return (R.from_euler("x", np.pi / 2) * R.from_rotvec(np.asarray(human_record["pose_aa"])[:, :3])
            * R.from_quat([.5, .5, .5, .5]).inv())


def load_official(assets: Assets, contract: Contract, sample: str = SAMPLE, hz: int = POLICY_HZ) -> Motion:
    human = load_published_pickle(assets.sample_root / "smpl_filtered" / sample)
    robot = load_published_pickle(assets.sample_root / "robot_filtered/210531" / sample)
    duration = min((len(human["smpl_joints"]) - 1) / human["fps"], (len(robot["dof"]) - 1) / robot["fps"])
    times = np.arange(int(np.floor(duration * hz + 1e-8)) + 1) / hz
    order, errors = detect_root_quat_order(robot)
    robot_rotation = R.from_rotvec(np.asarray(robot["pose_aa"])[:, 0])
    human50 = resample_rotation(human_root_rotation(human), float(human["fps"]), times)
    joints50 = resample(np.asarray(human["smpl_joints"]), float(human["fps"]), times)
    # 공개 smpl_joints는 모델 원점 기준 벡터입니다. 학습과 같은 원점을 유지합니다(관절 0으로 재중심화하지 않음).
    local = np.einsum("tji,tkj->tki", human50.as_matrix(), joints50)
    q = resample(np.asarray(robot["dof"]), float(robot["fps"]), times)
    root50 = resample_rotation(robot_rotation, float(robot["fps"]), times)
    metadata = {"source": "NVIDIA GEAR-SONIC sample_data (BONES-SEED-derived)", "sample": sample,
                "canonical_smpl": True, "published_robot_root_rot_order": order,
                "root_rot_order_max_error_rad": errors,
                "source_robot_hz": float(robot["fps"]), "source_human_hz": float(human["fps"]), "target_hz": hz}
    name = Path(sample).stem
    return Motion(name, times, local.astype(np.float32), human50.as_quat(),
                  q[:, contract.mujoco_to_isaaclab][:, 23:29].astype(np.float32), metadata,
                  g1_q=q, g1_dq=np.gradient(q, 1 / hz, axis=0),
                  g1_root=resample(np.asarray(robot["root_trans_offset"]), float(robot["fps"]), times),
                  g1_root_xyzw=root50.as_quat(), world24=joints50)


def load_opensim_npz(path, hz: int = POLICY_HZ) -> Motion:
    """(선택) OpenSim 계열 NPZ → SMPL24 '근사' 골격. 07번 노트북의 ETRI 변환 규칙과 같습니다.

    필요한 키: site_xpos, site_names(*_mimic), xpos, xquat(wxyz), body_names, body_quat, frequency, split_points.
    정식 SMPL 피팅이 아니며 G1 손목 각도는 0으로 대체합니다.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ["site_xpos", "site_names", "xpos", "xquat", "body_names",
                                                      "body_quat", "frequency", "split_points"]}
    sites = list(data["site_names"].astype(str))
    bodies = list(data["body_names"].astype(str))
    n = len(data["xpos"])
    if not np.array_equal(data["split_points"], [0, n]):
        raise ValueError("Multiple motion segments must be split before temporal resampling")
    source_hz = float(data["frequency"])
    if not 1 <= source_hz <= 1000 or n < 2:
        raise ValueError("Invalid frame count/frequency")

    def site(name):
        return data["site_xpos"][:, sites.index(name + "_mimic")].astype(float)

    def body(name):
        return data["xpos"][:, bodies.index(name)].astype(float)

    p, upper, head = site("pelvis"), site("upper_body"), site("head")
    ls, rs = site("left_shoulder"), site("right_shoulder")
    center = (ls + rs) / 2
    mix = lambda a, b, fraction: a * (1 - fraction) + b * fraction  # noqa: E731
    named = {"pelvis": p, "head": head, "spine1": mix(p, upper, .35), "spine2": mix(p, upper, .72),
             "spine3": upper, "neck": mix(center, head, .24), "left_collar": mix(center, ls, .5),
             "right_collar": mix(center, rs, .5)}
    for side, suffix in [("left", "l"), ("right", "r")]:
        for joint in ["hip", "knee", "foot", "shoulder", "elbow", "hand"]:
            named[f"{side}_{joint}"] = site(f"{side}_{joint}")
        named[f"{side}_ankle"] = body(f"talus_{suffix}")
        named[f"{side}_wrist"] = body(f"hand_{suffix}")
    world = np.stack([named[name] for name in SMPL_NAMES], axis=1)
    pelvis_id = bodies.index("pelvis")
    quats = data["xquat"][:, pelvis_id]
    pelvis = R.from_quat(quats[:, [1, 2, 3, 0]])
    rest = R.from_quat(data["body_quat"][pelvis_id, [1, 2, 3, 0]])
    root = pelvis * rest.inv()
    times = np.arange(int(np.floor((n - 1) / source_hz * hz + 1e-8)) + 1) / hz
    world50 = resample(world, source_hz, times)
    root50 = resample_rotation(root, source_hz, times)
    local = np.einsum("tji,tkj->tki", root50.as_matrix(), world50 - world50[:, :1])
    if np.median(local[:, 15, 2]) < .25 or np.max(np.abs(local[:, 0])) > 1e-5:
        raise ValueError("Anatomy/root coordinate validation failed")
    metadata = {"source_path": str(path), "source_hz": source_hz, "target_hz": hz, "canonical_smpl": False,
                "wrist_angles": "unavailable_neutral_zero_placeholder",
                "approximate_joints": ["spine1", "spine2", "spine3", "neck", "left_collar", "right_collar"]}
    return Motion(path.stem, times, local.astype(np.float32), root50.as_quat(),
                  np.zeros((len(times), 6), np.float32), metadata, world24=world50)


# ---------------------------------------------------------------------------
# 3. 정책 (Encoder → Token → Decoder)
# ---------------------------------------------------------------------------

def anchor_orientation_6d(robot_xyzw, alignment, root_xyzw_frames):
    """로봇 현재 방향 기준 참조 루트 회전 → 회전행렬 앞 두 열(6D), 프레임별로 평탄화."""
    rel = R.from_quat(robot_xyzw).inv() * alignment * R.from_quat(root_xyzw_frames)
    return rel.as_matrix()[..., :2].reshape(-1)


class Policy:
    def __init__(self, assets: Assets, contract: Contract):
        self.contract = contract
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.encoder = ort.InferenceSession(str(assets.model_dir / "model_encoder.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(assets.model_dir / "model_decoder.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        for session, n_in, n_out in [(self.encoder, 1762, 64), (self.decoder, 994, 29)]:
            if session.get_inputs()[0].shape[-1] != n_in or session.get_outputs()[0].shape[-1] != n_out:
                raise ValueError("ONNX shape does not match official observation contract")

    def encoder_obs(self, motion: Motion, frame: int, robot_xyzw, alignment, mode: int = 2) -> np.ndarray:
        c = self.contract
        x = np.zeros(1762, np.float32)
        blocks = {"encoder_mode_4": np.array([mode, 0, 0, 0], np.float32)}
        ids1 = np.minimum(frame + np.arange(10), len(motion.times) - 1)
        ids5 = np.minimum(frame + np.arange(10) * 5, len(motion.times) - 1)
        root = motion.human_root_xyzw if mode == 2 else motion.g1_root_xyzw
        if mode == 2:
            blocks.update({"smpl_joints_10frame_step1": motion.local24[ids1].reshape(-1),
                           "smpl_anchor_orientation_10frame_step1": anchor_orientation_6d(robot_xyzw, alignment, root[ids1]),
                           "motion_joint_positions_wrists_10frame_step1": motion.wrists_il[ids1].reshape(-1)})
        elif mode == 0 and motion.g1_q is not None:
            blocks.update({"motion_joint_positions_10frame_step5": motion.g1_q[ids5][:, c.mujoco_to_isaaclab].reshape(-1),
                           "motion_joint_velocities_10frame_step5": motion.g1_dq[ids5][:, c.mujoco_to_isaaclab].reshape(-1),
                           "motion_anchor_orientation_10frame_step5": anchor_orientation_6d(robot_xyzw, alignment, root[ids5])})
        else:
            raise ValueError("Mode 0 requires the paired official G1 reference")
        if set(blocks) != c.mode_fields[mode]:
            raise ValueError(f"Mode {mode} fields disagree with observation YAML")
        for name, values in blocks.items():
            x[c.enc_slices[name]] = values
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite encoder observation")
        return x

    def encode_obs(self, x) -> np.ndarray:
        token = self.encoder.run(None, {self.encoder.get_inputs()[0].name: np.asarray(x, np.float32)[None]})[0].reshape(64)
        if not np.isfinite(token).all():
            raise ValueError("Nonfinite encoder token")
        return token.astype(np.float32)

    def encode(self, motion, frame, robot_xyzw, alignment, mode=2):
        return self.encode_obs(self.encoder_obs(motion, frame, robot_xyzw, alignment, mode))

    @staticmethod
    def decoder_obs(token, history) -> np.ndarray:
        """history: 오래된 것 → 최신 순서의 10개 dict. 키별(term-major)로 이어 붙입니다."""
        x = np.concatenate([np.asarray(token).reshape(64)] +
                           [np.asarray([entry[key] for entry in history]).reshape(-1) for key in HISTORY_KEYS]
                           ).astype(np.float32)
        if x.shape != (994,) or not np.isfinite(x).all():
            raise ValueError("Invalid decoder observation")
        return x

    def decode_obs(self, x) -> np.ndarray:
        action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: np.asarray(x, np.float32)[None]})[0].reshape(29)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite policy action")
        return action

    def decode(self, token, history):
        return self.decode_obs(self.decoder_obs(token, history))


def action_to_target(action_il, contract: Contract) -> np.ndarray:
    """IsaacLab 순서 정책 출력 → MuJoCo 순서 PD 목표 관절각(rad)."""
    return contract.default + np.asarray(action_il)[contract.isaaclab_to_mujoco] * contract.scale


# ---------------------------------------------------------------------------
# 4. MuJoCo 시뮬레이터 (공식 scene_43dof.xml + 관절별 PD 토크)
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(self, assets: Assets, contract: Contract):
        self.contract = contract
        self.model = mujoco.MjModel.from_xml_path(str(assets.repo / MODEL_REL / "scene_43dof.xml"))
        config = yaml.safe_load((assets.repo / SIM_CONFIG_REL).read_text())
        self.model.opt.timestep = float(config["SIMULATE_DT"])
        self.substeps = round(1 / POLICY_HZ / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, 1 / POLICY_HZ):
            raise ValueError("Physics timestep must divide the 50 Hz policy period")
        self.jids = np.array([self.model.joint(name).id for name in MJ_NAMES])
        self.qaddr = self.model.jnt_qposadr[self.jids]
        self.vaddr = self.model.jnt_dofadr[self.jids]
        actuator_by_joint = {int(self.model.actuator_trnid[i, 0]): i for i in range(self.model.nu)
                             if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT}
        self.aids = np.array([actuator_by_joint[int(j)] for j in self.jids])
        if not np.allclose(self.model.actuator_gainprm[self.aids, 0], 1) or not np.allclose(self.model.actuator_biasprm[self.aids], 0):
            raise ValueError("Expected torque motors, not position actuators")
        feet = {self.model.body(name).id for name in ["left_ankle_roll_link", "right_ankle_roll_link"]}
        self.foot_geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in feet
                           and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                           and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])]
        if len(self.foot_geoms) != 2:
            raise ValueError("Expected the two official foot collision boxes")
        self.pelvis_id = self.model.body("pelvis").id
        self.data = mujoco.MjData(self.model)

    def reset(self, motion: Motion | None = None, use_reference: bool = True):
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0, 0, .83]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qaddr] = self.contract.default
        if use_reference and motion is not None and motion.g1_q is not None:
            d.qpos[:3] = motion.g1_root[0]
            d.qpos[3:7] = motion.g1_root_xyzw[0, [3, 0, 1, 2]]   # scipy xyzw → MuJoCo wxyz
            d.qpos[self.qaddr] = motion.g1_q[0]
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[i, 2] - np.abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i] for i in self.foot_geoms)
        ground_delta = .001 - lowest
        d.qpos[2] += ground_delta
        mujoco.mj_forward(m, d)
        return {"ground_height_correction_m": float(ground_delta), "external_support": False}

    def base_xyzw(self):
        return self.data.qpos[[4, 5, 6, 3]].copy()   # MuJoCo wxyz → scipy xyzw

    def observation(self, previous_action):
        c, d = self.contract, self.data
        rotation = R.from_quat(self.base_xyzw())
        return {"omega": d.qvel[3:6].copy(),
                "q": (d.qpos[self.qaddr] - c.default)[c.mujoco_to_isaaclab],
                "dq": d.qvel[self.vaddr][c.mujoco_to_isaaclab].copy(),
                "action": np.asarray(previous_action).copy(),
                "gravity": rotation.inv().apply([0, 0, -1])}

    def step(self, target):
        d, m, c = self.data, self.model, self.contract
        max_torque = 0.
        for _ in range(self.substeps):
            tau = c.kp * (target - d.qpos[self.qaddr]) - c.kd * d.qvel[self.vaddr]
            clipped = np.clip(tau, m.actuator_ctrlrange[self.aids, 0], m.actuator_ctrlrange[self.aids, 1])
            d.ctrl[:] = 0
            d.ctrl[self.aids] = clipped
            max_torque = max(max_torque, float(np.max(np.abs(clipped))))
            mujoco.mj_step(m, d)
        return max_torque


# ---------------------------------------------------------------------------
# 5. 폐루프 실행
# ---------------------------------------------------------------------------

def run_physics(sim: Simulator, policy: Policy, motion: Motion, mode, max_frames=None,
                pipeline=None, on_frame=None, verbose=False):
    """초기화 이후 루트 강제·탄성 지지 없이 정책만으로 실행합니다.

    mode: 0(G1 참조), 2(SMPL 인체), "reference_pd"(정책 없이 참조 관절각 PD 재생)
    pipeline: Day 2 실습용. `decoder_obs(token, history)`, `action_to_target(action, contract)`,
              `observation(sim, previous_action)`를 가진 객체를 넣으면 해당 단계를 대체합니다.
    on_frame: Day 3 실습용 콜백 `on_frame(frame, sim)` — 외란(xfrc_applied) 등을 넣을 때 사용.
    """
    c = sim.contract
    initialization = sim.reset(motion)
    root_series = motion.g1_root_xyzw if mode in (0, "reference_pd") else motion.human_root_xyzw
    alignment = R.from_euler("z", float(yaw(R.from_quat(sim.base_xyzw())) - yaw(R.from_quat(root_series[0]))))
    observe = (lambda prev: pipeline.observation(sim, prev)) if pipeline is not None and hasattr(pipeline, "observation") else sim.observation
    build_dec = pipeline.decoder_obs if pipeline is not None and hasattr(pipeline, "decoder_obs") else Policy.decoder_obs
    to_target = pipeline.action_to_target if pipeline is not None and hasattr(pipeline, "action_to_target") else action_to_target
    previous = np.zeros(29, np.float32)
    history = deque([observe(previous) for _ in range(10)], maxlen=10)
    n = min(len(motion.times), max_frames) if max_frames is not None else len(motion.times)
    rows = {key: [] for key in ["qpos", "qvel", "action_il", "target_mj", "token", "max_torque_nm", "up_cosine"]}
    reason = "completed_requested_interval"
    begin = time.perf_counter()
    for frame in range(n):
        history.append(observe(previous))
        if on_frame is not None:
            on_frame(frame, sim)
        if mode == "reference_pd":
            target, action, token = motion.g1_q[frame], np.zeros(29), np.zeros(64)
        else:
            token = policy.encode(motion, frame, sim.base_xyzw(), alignment, mode)
            action = policy.decode_obs(build_dec(token, list(history)))
            target = to_target(action, c)
        torque = sim.step(target)
        previous = np.asarray(action, np.float32).copy()
        if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
            reason = "nonfinite_simulation_state"
            break
        up = float(R.from_quat(sim.base_xyzw()).as_matrix()[2, 2])
        for key, value in {"qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(), "action_il": np.asarray(action).copy(),
                           "target_mj": np.asarray(target).copy(), "token": np.asarray(token).copy(),
                           "max_torque_nm": torque, "up_cosine": up}.items():
            rows[key].append(value)
        if int(sum(item.number for item in sim.data.warning)):
            reason = "mujoco_warning_or_automatic_reset"
            break
        if sim.data.qpos[2] < .35 or up < .2:
            reason = "fall_height_or_pelvis_tilt"
            break
        if verbose and (frame + 1) % 250 == 0:
            print(f"  {motion.name}/{mode}: {frame + 1}/{n} frames", flush=True)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["times_s"] = (np.arange(len(arrays["qpos"])) + 1) / POLICY_HZ
    summary = {"name": motion.name, "mode": mode, "requested_frames": int(n),
               "simulated_frames": int(len(arrays["qpos"])), "policy_hz": POLICY_HZ,
               "physics_dt": float(sim.model.opt.timestep), "physics_substeps": sim.substeps,
               "completed": len(arrays["qpos"]) == n and reason == "completed_requested_interval",
               "stop_reason": reason, "runtime_s": round(time.perf_counter() - begin, 2),
               "initialization": initialization}
    if len(arrays["qpos"]):
        xy = arrays["qpos"][:, :2]
        summary.update({"min_root_z_m": float(arrays["qpos"][:, 2].min()),
                        "min_pelvis_up_cosine": float(arrays["up_cosine"].min()),
                        "max_abs_raw_action": float(np.abs(arrays["action_il"]).max()),
                        "base_xy_path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0})
        if motion.g1_q is not None:
            ids = np.minimum(np.arange(len(xy)) + 1, len(motion.times) - 1)
            err = arrays["qpos"][:, sim.qaddr] - motion.g1_q[ids]
            summary["joint_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(err ** 2))))
            summary["mean_abs_joint_error_deg"] = float(np.degrees(np.mean(np.abs(err))))
    return summary, arrays
