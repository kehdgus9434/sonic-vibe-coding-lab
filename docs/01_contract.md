# 입출력 계약서 — SONIC Encoder/Decoder × G1 × MuJoCo

> AI가 짠 코드가 맞는지 판단하는 **유일한 기준**입니다. 바이브 코딩 도구에 이 문서를 통째로 컨텍스트로 넣으세요.
> 이 표에 없는 규칙은 "모른다"가 정답이며, AI가 지어낸 규칙은 공식 문서·코드로 확인하기 전까지 믿지 않습니다.

근거(고정 리비전):
[`observation_config.yaml`](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/observation_config.yaml) ·
[`policy_parameters.hpp`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4f5e118ae82e0448f2b2f441024d59fe74e0a4d9/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp) ·
[관측값 레퍼런스](https://nvlabs.github.io/GR00T-WholeBodyControl/references/observation_config.html) ·
`AI인력양성사업 교육자료` 05·06·07번 노트북(실제 디버깅 기록)

---

## 0. 전체 흐름 (50 Hz 제어 루프)

```
참조 모션(사람 SMPL 또는 G1)          로봇 상태(MuJoCo)
        │                                   │
        ▼                                   ▼
 Encoder 입력 1762 ──► model_encoder.onnx ──► token 64
                                              │
           고유수용 이력 10프레임 ─────────────┤
                                              ▼
                     Decoder 입력 994 ──► model_decoder.onnx ──► action 29 (IsaacLab 순서)
                                                                   │
                         target = default + action[isaaclab_to_mujoco] × scale
                                                                   │
                       PD 토크 τ = Kp(target − q) − Kd·q̇  (5 ms × 4 substep)
```

| 항목 | 값 |
|---|---|
| 정책 주기 | 50 Hz (20 ms) |
| 물리 주기 | `SIMULATE_DT` = 0.005 s → 정책 1스텝당 4 substep |
| ONNX 입력/출력 이름 | encoder `obs_dict [1,1762]` → `encoded_tokens [1,64]` / decoder `obs_dict [1,994]` → `action [1,29]` |
| 실행 provider | `CPUExecutionProvider` (GPU 불필요) |
| 넘어짐 판정(실습 기준) | 골반 높이 < 0.35 m 또는 골반 up-cosine < 0.2 |

> ⚠️ `observation_config.yaml` 맨 위 주석은 `Total dimension: 436`이라고 적혀 있지만 **실제 ONNX 입력은 994**입니다.
> 공식 파일의 주석도 틀릴 수 있습니다. 차원은 ONNX 세션에서 직접 확인합니다.

---

## 1. 좌표계와 쿼터니언 규칙 — 가장 많이 틀리는 곳

| 대상 | 성분 순서 | 비고 |
|---|---|---|
| MuJoCo `qpos[3:7]` (free joint) | **w, x, y, z** | |
| `scipy.spatial.transform.Rotation.from_quat` | **x, y, z, w** | MuJoCo → scipy: `qpos[[4,5,6,3]]` |
| 공식 G1 샘플 `root_rot` | **x, y, z, w** | `pose_aa[:,0]`(축-각)과 비교해 확인. wxyz로 읽으면 최대 약 179° 오류(06번 노트북 실제 버그) |
| 월드 좌표 | Z-up | 중력 = (0, 0, −1) |

**인체(SMPL) 루트 회전** — 학습 코드 규칙, 그대로 사용:

```
R_root(t) = Rx(π/2) · R_axisangle(pose_aa[t, 0:3]) · R_quat_xyzw([.5,.5,.5,.5])⁻¹
```

**인체 관절 로컬 좌표** (공식 샘플): `local24[t,k] = R_root(t)⁻¹ · smpl_joints[t,k]`
(공개 `smpl_joints`는 모델 원점 기준 벡터입니다. 관절 0으로 재중심화하지 않습니다.)

**시간 정렬**: 두 데이터를 각자 fps에서 50 Hz로. 길이 = 둘 중 짧은 쪽(공식 샘플: 사람 50 fps 2002프레임, G1 30 fps 1202프레임 → 2002프레임 40.02 s).
위치·관절각은 선형 보간, 회전은 SLERP.

---

## 2. 관절 순서 (29 DOF)

정책의 입력·출력은 **IsaacLab 순서**, MuJoCo 모델·기본각·스케일·Kp/Kd는 **MuJoCo 순서**입니다.
공식 헤더의 두 배열을 이름 그대로 씁니다.

```python
x_il = x_mj[mujoco_to_isaaclab]     # MuJoCo 순서 벡터 → IsaacLab 순서
a_mj = a_il[isaaclab_to_mujoco]     # IsaacLab 순서 벡터 → MuJoCo 순서
```

| # | MuJoCo 순서 | IsaacLab 순서 |
|---:|---|---|
| 0 | left_hip_pitch | left_hip_pitch |
| 1 | left_hip_roll | right_hip_pitch |
| 2 | left_hip_yaw | waist_yaw |
| 3 | left_knee | left_hip_roll |
| 4 | left_ankle_pitch | right_hip_roll |
| 5 | left_ankle_roll | waist_roll |
| 6 | right_hip_pitch | left_hip_yaw |
| 7 | right_hip_roll | right_hip_yaw |
| 8 | right_hip_yaw | waist_pitch |
| 9 | right_knee | left_knee |
| 10 | right_ankle_pitch | right_knee |
| 11 | right_ankle_roll | left_shoulder_pitch |
| 12 | waist_yaw | right_shoulder_pitch |
| 13 | waist_roll | left_ankle_pitch |
| 14 | waist_pitch | right_ankle_pitch |
| 15 | left_shoulder_pitch | left_shoulder_roll |
| 16 | left_shoulder_roll | right_shoulder_roll |
| 17 | left_shoulder_yaw | left_ankle_roll |
| 18 | left_elbow | right_ankle_roll |
| 19 | left_wrist_roll | left_shoulder_yaw |
| 20 | left_wrist_pitch | right_shoulder_yaw |
| 21 | left_wrist_yaw | left_elbow |
| 22 | right_shoulder_pitch | right_elbow |
| 23 | right_shoulder_roll | left_wrist_roll |
| 24 | right_shoulder_yaw | right_wrist_roll |
| 25 | right_elbow | left_wrist_pitch |
| 26 | right_wrist_roll | right_wrist_pitch |
| 27 | right_wrist_pitch | left_wrist_yaw |
| 28 | right_wrist_yaw | right_wrist_yaw |

(모든 이름 뒤에 `_joint`가 붙습니다. MuJoCo 장면 `scene_43dof.xml`에는 손가락 관절 14개가 더 있지만 정책은 제어하지 않습니다.)

---

## 3. Encoder 입력 (1762) — YAML의 enabled 순서대로 이어 붙임

| 블록 | 길이 | 구성 | 모드 0 (G1) | 모드 2 (SMPL) |
|---|---:|---|:-:|:-:|
| `encoder_mode_4` | 4 | `[mode, 0, 0, 0]` | ● | ● |
| `motion_joint_positions_10frame_step5` | 290 | 10프레임 × 29관절(IsaacLab) | ● | |
| `motion_joint_velocities_10frame_step5` | 290 | 10프레임 × 29관절(IsaacLab) | ● | |
| `motion_root_z_position_10frame_step5` | 10 | | | |
| `motion_root_z_position` | 1 | | | |
| `motion_anchor_orientation` | 6 | | | |
| `motion_anchor_orientation_10frame_step5` | 60 | 10프레임 × 6D | ● | |
| `motion_joint_positions_lowerbody_10frame_step5` | 120 | | | |
| `motion_joint_velocities_lowerbody_10frame_step5` | 120 | | | |
| `vr_3point_local_target` | 9 | | | |
| `vr_3point_local_orn_target` | 12 | | | |
| `smpl_joints_10frame_step1` | 720 | 10프레임 × 24관절 × xyz | | ● |
| `smpl_anchor_orientation_10frame_step1` | 60 | 10프레임 × 6D | | ● |
| `motion_joint_positions_wrists_10frame_step1` | 60 | 10프레임 × 손목 6관절(IsaacLab 23–28) | | ● |

- 해당 모드에서 쓰지 않는 블록은 **0**.
- 미래 프레임 인덱스: `step1` → `min(t + [0..9], T−1)`, `step5` → `min(t + 5·[0..9], T−1)`.
- 평탄화는 **프레임 우선**(frame-major): `(10, 24, 3).reshape(-1)`.
- **6D 방향**: 프레임마다 `R_rel = R_robot⁻¹ · A · R_root(t)`의 회전행렬 **앞 두 열**을 행 우선으로 → `[r00, r01, r10, r11, r20, r21]`.
  - `R_robot`: 현재 MuJoCo 기저 회전(매 스텝 갱신)
  - `A = Rz(yaw(R_robot(0)) − yaw(R_root(0)))`: 시작 시 한 번 계산하는 헤딩 정렬
  - `R_root`: 모드 2는 인체 루트, 모드 0은 G1 참조 루트
- 손목 6관절: 공식 샘플은 G1 참조 관절각의 IsaacLab 23–28번, 그 외 데이터는 0(검증되지 않은 대체값).

## 4. Decoder 입력 (994) — 블록(항목) 우선, 각 블록 안은 **오래된 → 최신**

| 블록 | 길이 | 한 프레임 값 |
|---|---:|---|
| `token_state` | 64 | Encoder 출력 |
| `his_base_angular_velocity_10frame_step1` | 30 | `qvel[3:6]` (기저 좌표계 각속도) |
| `his_body_joint_positions_10frame_step1` | 290 | `(q_mj − default)[mujoco_to_isaaclab]` ← **기본각을 뺀 값** |
| `his_body_joint_velocities_10frame_step1` | 290 | `q̇_mj[mujoco_to_isaaclab]` |
| `his_last_actions_10frame_step1` | 290 | 직전 정책 출력 원값(IsaacLab, 클리핑 없음) |
| `his_gravity_dir_10frame_step1` | 30 | `R_base⁻¹ · (0,0,−1)` |

- 이력은 길이 10 큐. 시작 시 초기 상태 관측(직전 액션 0)을 10번 채움.
- 매 정책 스텝: **새 관측을 먼저 append → Encoder → Decoder → 액션 적용**.
- 이력 순서는 공식 C++ `GatherHis*`의 `newest_first=false`에 해당(05번 노트북에서 수정된 버그).

## 5. 액션 → 토크

```python
target_mj = default + action_il[isaaclab_to_mujoco] * scale        # rad, MuJoCo 순서
tau = kp * (target_mj - q_mj) - kd * qd_mj                         # 매 substep(5 ms)마다 재계산
tau = clip(tau, actuator_ctrlrange)                                # scene_43dof.xml의 토크 모터
```

`default`, `scale(g1_action_scale)`, `kp(kps)`, `kd(kds)`는 모두 `policy_parameters.hpp`(MuJoCo 순서)에서 읽습니다.
MuJoCo Menagerie의 위치 액추에이터 모델을 쓰면 이 계약이 깨집니다(07번 노트북에서 교체한 이유).

## 6. 알려진 한계 (결과를 해석할 때 반드시 고려)

- 이 실습의 물리 루프는 공식 C++/TensorRT 배포 스택이 아니라 Python 재구현입니다. 완주 = "지정 구간을 넘어지지 않고 실행"이며 **동작 품질 검증이 아닙니다**.
- 이 저장소 코드의 실제 모델 결과(GitHub Actions `real-model-check`, 원본 07번 노트북의 Colab 기록과 동일): 공식 샘플 모드 0·모드 2는 2002/2002 프레임 완주(참조 대비 관절 RMSE 각각 약 5.1°·6.1°), 정책 없는 참조 PD 재생은 55프레임에서 넘어짐.
- OpenSim 계열 데이터를 SMPL24로 바꾸는 `load_opensim_npz`는 규칙 기반 **근사**입니다(정식 SMPL 피팅 아님, 손목 각도 0).
- 어떤 출력도 실물 로봇에 보내지 않습니다.
