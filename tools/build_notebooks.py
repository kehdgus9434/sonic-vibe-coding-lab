"""notebooks/day*.ipynb 생성기. 노트북 내용을 고칠 때는 이 파일을 고친 뒤 다시 실행하세요.

    python tools/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO_SLUG = "kehdgus9434/sonic-vibe-coding-lab"   # tools/set_repo.py가 바꿉니다


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata["cellView"] = "form"
    return cell


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO_SLUG}/blob/main/notebooks/{name})")


SETUP = f'''
#@title 0. 실습 환경 준비 — 처음 한 번 실행 (약 1–2분)
REPO = "https://github.com/{REPO_SLUG}"   # 에이전트형 도구 사용자는 내 Fork 주소로 바꾸세요
import os, sys, subprocess
from pathlib import Path
if Path("/content").exists():                              # Google Colab
    LAB_DIR = Path("/content/sonic-vibe-coding-lab")
    if not LAB_DIR.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(LAB_DIR)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(LAB_DIR / "requirements-colab.txt")], check=True)
else:                                                      # 로컬 Jupyter (저장소 안에서 실행)
    LAB_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(LAB_DIR)
sys.path.insert(0, str(LAB_DIR))
import numpy as np
import sonic_lab
from sonic_lab import check, reference as ref, render, analysis
lab = sonic_lab.start()
'''

LOG = '''
### 📝 프롬프트 로그 (이 셀을 더블클릭해 편집)
| # | 내가 한 요청 | AI 답 요약 | 내가 확인한 근거 | 채택? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
'''

MISSION_RULE = "> 🧭 **방법**: 아래 설명 + `docs/01_contract.md` + 함수 시그니처를 바이브 코딩 도구에 주고 ✍️ 셀을 채운 뒤, ✅ 셀로 확인합니다."


# ---------------------------------------------------------------------------
def day1():
    name = "day1_anatomy.ipynb"
    cells = [md(f"""
# Day 1 · SONIC 해부하기 — 모델과 데이터의 '계약' 읽기
{badge(name)}

**오늘의 결과물**: 공식 걷기 샘플 → *내가 만든* Encoder 입력(1762) → 64차원 토큰 시계열

| 시간 | 내용 |
|---|---|
| 0:00–0:15 | 환경 준비 · 목적지 미리보기 · 도구 연결 |
| 0:15–0:30 | Mission 1-1 ONNX 입출력 해부 |
| 0:30–0:50 | Mission 1-2 Encoder 1762차원 블록 표 |
| 0:50–1:10 | Mission 1-3 쿼터니언 순서 탐정 |
| 1:10–1:25 | Mission 1-4 30 Hz ↔ 50 Hz 시간 정렬 |
| 1:25–1:50 | Mission 1-5 Mode-2 Encoder 입력 조립 → 토큰 |
| 1:50–2:00 | 정리 · 프롬프트 로그 |

**규칙**: `sonic_lab/reference.py`, `sonic_lab/check.py`는 정답 파일이라 열지 않습니다. 근거는 `docs/01_contract.md`뿐입니다.
⚠️ 이 실습의 어떤 출력도 실물 로봇에 보내지 않습니다.
"""),
        code(SETUP, hidden=True),
        md("""
## 목적지 미리보기 (5분)
참조 구현으로 3일 뒤 도달할 결과를 먼저 봅니다: **사람 걷기 데이터 → SONIC → MuJoCo G1이 넘어지지 않고 걷기**.
"""),
        code('''
summary, arrays = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=250)
print({k: summary[k] for k in ["simulated_frames", "requested_frames", "completed", "stop_reason", "runtime_s"]})
render.show(render.skeleton_video(lab, arrays, "outputs/day1_preview.mp4", title="reference pipeline", seconds=5))
'''),
        md(LOG),
        md(f"""
---
## Mission 1-1 · ONNX 입출력 해부 (15분)
{MISSION_RULE}

- **목표**: `model_encoder.onnx`, `model_decoder.onnx`의 입력·출력 이름과 shape를 확인한다.
- **주어진 것**: `lab.assets.model_dir` (ONNX 2개 + `observation_config.yaml`)
- **완료 기준**: ✅ 셀 통과
- **생각해 볼 것**: `observation_config.yaml` 맨 위 주석의 *Total dimension* 값과 실제 Decoder 입력 길이를 비교해 보세요. 어느 쪽을 믿어야 할까요?
"""),
        code('''
# ✍️ Mission 1-1
def inspect_onnx(model_dir):
    """반환: (encoder 입력 길이, encoder 출력 길이, decoder 입력 길이, decoder 출력 길이)"""
    raise NotImplementedError

# from my_work.day1 import *    # 에이전트형 도구로 my_work/day1.py에 작성했다면 주석 해제
'''),
        code('''
# ✅ 확인
check.onnx_io(*inspect_onnx(lab.assets.model_dir))
'''),
        md(f"""
---
## Mission 1-2 · Encoder 1762차원 블록 표 (20분)
{MISSION_RULE}

- **목표**: `observation_config.yaml`을 읽어 Encoder 관측 14개 블록의 `(start, stop)` 표를 코드로 만든다.
- **함정**: YAML에는 **길이가 적혀 있지 않습니다.** 이름(`10frame`, `step5`, `lowerbody`, `vr_3point` …)과 계약서 3절로 길이를 정하고, 합이 ONNX 입력 길이와 같은지 `assert`로 확인하게 하세요.
- **완료 기준**: ✅ 셀 통과
"""),
        code('''
# ✍️ Mission 1-2
def build_encoder_slices(yaml_path):
    """반환: {관측 이름: (start, stop)} — YAML의 enabled 순서"""
    raise NotImplementedError

# from my_work.day1 import *
'''),
        code('''
# ✅ 확인
slices = build_encoder_slices(lab.assets.model_dir / "observation_config.yaml")
check.encoder_slices(lab, slices)
'''),
        md(f"""
---
## Mission 1-3 · 쿼터니언 순서 탐정 (20분)
{MISSION_RULE}

공식 G1 샘플에는 루트 회전이 **두 가지 표현**으로 들어 있습니다: `pose_aa[:, 0]`(축-각 벡터)과 `root_rot`(쿼터니언).
실제 연구 과정(06번 노트북)에서 `root_rot`를 잘못된 성분 순서로 읽어 **최대 약 179°** 회전 오류가 난 적이 있습니다.

- **목표**: 두 표현을 비교해 `root_rot`가 `xyzw`인지 `wxyz`인지 **데이터로** 판정하는 함수를 만든다.
- **요청 팁**: "어느 쪽일 것 같아?"가 아니라 "두 가정 각각의 최대 각도 오차를 계산해서 비교하는 코드"를 요청하세요.
- **완료 기준**: ✅ 셀 통과 + 틀린 순서일 때 오차가 몇 도인지 출력
"""),
        code('''
# 데이터 불러오기 (제공)
from sonic_lab.reference import load_published_pickle
from sonic_lab.assets import SAMPLE
robot = load_published_pickle(lab.assets.sample_root / "robot_filtered/210531" / SAMPLE)
human = load_published_pickle(lab.assets.sample_root / "smpl_filtered" / SAMPLE)
print("G1  :", {k: np.shape(v) for k, v in robot.items()})
print("SMPL:", {k: np.shape(v) for k, v in human.items()})
'''),
        code('''
# ✍️ Mission 1-3
def detect_root_quat_order(robot_record):
    """반환: 'xyzw' 또는 'wxyz' (판정 근거가 되는 오차도 출력할 것)"""
    raise NotImplementedError

# from my_work.day1 import *
'''),
        code('''
# ✅ 확인
check.root_quat_order(detect_root_quat_order(robot))
'''),
        md(f"""
---
## Mission 1-4 · 30 Hz ↔ 50 Hz 시간 정렬 (15분)
{MISSION_RULE}

- G1 참조는 30 fps, 사람(SMPL)은 50 fps이고 길이도 조금 다릅니다. 정책은 50 Hz로 돕니다.
- **목표**: 두 데이터 중 **짧은 쪽 길이**에 맞춘 50 Hz 시간축과, 그 시간에서 선형 보간한 G1 관절각 `(T, 29)`를 만든다.
- **완료 기준**: ✅ 셀 통과
"""),
        code('''
# ✍️ Mission 1-4
def resample_g1(robot_record, human_record, hz=50):
    """반환: (times (T,), g1_q50 (T, 29))"""
    raise NotImplementedError

# from my_work.day1 import *
'''),
        code('''
# ✅ 확인
times, g1_q50 = resample_g1(robot, human)
check.resampled_motion(lab, times, g1_q50)
'''),
        md(f"""
---
## Mission 1-5 · Mode-2 Encoder 입력 조립 → 토큰 (25분)
{MISSION_RULE}

`lab.motion`은 이미 50 Hz로 정렬된 공식 샘플입니다(Mission 1-3/1-4를 못 끝냈어도 진행 가능).

| 필드 | shape | 의미 |
|---|---|---|
| `lab.motion.local24` | (T, 24, 3) | 루트 기준 SMPL 관절 좌표 |
| `lab.motion.human_root_xyzw` | (T, 4) | 인체 루트 회전 (scipy xyzw) |
| `lab.motion.wrists_il` | (T, 6) | G1 손목 6관절 (IsaacLab 23–28) |
| `lab.motion.times` | (T,) | 50 Hz 시간축 |

- **목표**: 계약서 3절대로 `frame` 시점의 1762차원 Mode-2 Encoder 입력을 만든다. `robot_xyzw`는 현재 로봇 기저 회전(xyzw), `alignment`는 scipy `Rotation`(헤딩 정렬 A)입니다.
- **완료 기준**: ✅ 셀 통과 → 토큰 시계열 그림
"""),
        code('''
# ✍️ Mission 1-5
from scipy.spatial.transform import Rotation as R

def build_encoder_obs_mode2(motion, frame, robot_xyzw, alignment):
    """반환: (1762,) float32"""
    raise NotImplementedError

# from my_work.day1 import *
'''),
        code('''
# ✅ 확인
check.encoder_obs(lab, build_encoder_obs_mode2)
'''),
        code('''
# 내 Encoder 입력으로 만든 토큰 시계열 (처음 10초, 로봇이 참조와 같은 방향이라고 가정)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
A = R.identity()
tokens = np.array([lab.policy.encode_obs(build_encoder_obs_mode2(lab.motion, t, lab.motion.human_root_xyzw[t], A))
                   for t in range(0, 500, 2)])
plt.figure(figsize=(11, 3))
plt.plot(lab.motion.times[0:500:2], tokens[:, :8]); plt.xlabel("time (s)"); plt.title("SONIC token dims 0–7 (walk sample)")
plt.show()
'''),
        md("""
---
## 정리 (10분)
- 오늘 AI가 **틀렸던** 순간이 있었나요? 무엇으로 알았나요? → 프롬프트 로그에 기록
- 토큰 그래프가 걸음 주기를 따라 주기적으로 움직이나요? 그렇다고 해서 *입력이 맞다*고 할 수 있을까요?
- **Day 2 예고**: 오류 없이 돌아가지만 로봇이 넘어지는 'AI가 작성한' 파이프라인에서 버그 4개를 찾습니다.
"""),
    ]
    return name, cells


# ---------------------------------------------------------------------------
def day2():
    name = "day2_bughunt.ipynb"
    cells = [md(f"""
# Day 2 · 버그 사냥 — "돌아간다"와 "맞다" 사이
{badge(name)}

**오늘의 결과물**: 버그 4개를 고친 폐루프 파이프라인 + 참조 구현 없이도 버그를 잡는 불변식 테스트 + 40초 완주 영상

| 시간 | 내용 |
|---|---|
| 0:00–0:10 | 환경 준비 · 증상 관찰 |
| 0:10–1:05 | Mission 2-1 버그 사냥 (4개) |
| 1:05–1:30 | Mission 2-2 불변식 테스트 만들기 |
| 1:30–1:50 | Mission 2-3 40초 완주 + 영상 |
| 1:50–2:00 | 회고 |
"""),
        code(SETUP, hidden=True),
        code('''
# 버그 사냥 대상 불러오기 — 파일을 고칠 때마다 이 셀부터 다시 실행
import importlib.util
def load_buggy(path="exercises/buggy_pipeline.py"):
    spec = importlib.util.spec_from_file_location("buggy_pipeline", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.Pipeline()
pipeline = load_buggy()
'''),
        md("""
## 증상 관찰 (5분)
AI가 작성한 파이프라인(`exercises/buggy_pipeline.py`)은 오류 없이 실행됩니다. 결과를 보세요.
"""),
        code('''
bug_summary, bug_arrays = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=250, pipeline=pipeline)
print({k: bug_summary[k] for k in ["simulated_frames", "requested_frames", "stop_reason"]})
render.show(render.skeleton_video(lab, bug_arrays, "outputs/day2_buggy.mp4", title="buggy pipeline", seconds=5))
'''),
        md(LOG),
        md("""
---
## Mission 2-1 · 버그 사냥 (55분)

- **대상**: `exercises/buggy_pipeline.py` (Colab 왼쪽 📁 → `sonic-vibe-coding-lab/exercises/`에서 더블클릭해 편집)
- **버그**: 계약서와 다른 곳이 **4개**. 세 함수(`observation`, `decoder_obs`, `action_to_target`)에 흩어져 있고, 한 함수에 2개가 있을 수 있습니다.
- **진행**
  1. 파일 전체 + `docs/01_contract.md` + 아래 ✅ 출력을 AI에게 주고 **원인 후보와 그것을 가려낼 테스트**를 먼저 요청합니다(`docs/02_vibe_coding_guide.md`의 디버깅 템플릿).
  2. 테스트로 확인된 것만 고칩니다. 고칠 때마다 파일 맨 위 `BUGLOG`에 "무엇을, 어떤 근거로"를 적습니다.
  3. 파일 저장 → `load_buggy` 셀 → ✅ 셀 재실행.
- **완료 기준**: 구성요소 3/3 통과 + 폐루프 250프레임 완주
- ⚠️ Colab 런타임이 끊기면 파일 수정도 사라집니다. 고친 코드를 수시로 아래 메모 셀에 복사해 두세요.
"""),
        code('''
# ✅ 확인 (고칠 때마다 load_buggy 셀 → 이 셀)
pipeline = load_buggy()
_ = check.bughunt(lab, pipeline, frames=250)
'''),
        md("""
```python
# 📋 고친 코드 백업용 메모 (런타임이 끊기면 여기서 복구)
```
"""),
        md(f"""
---
## Mission 2-2 · 불변식 테스트 만들기 (25분)
{MISSION_RULE}

방금은 정답(참조 구현)과 비교하는 `check`가 있어서 버그를 찾을 수 있었습니다. **실제 연구에서는 정답이 없습니다.**
참조 구현 없이도 같은 버그를 잡을 수 있는 테스트를 AI와 함께 만드세요.

- **입력**: `pipeline`(검사 대상), `lab`(`lab.sim`, `lab.contract` 사용 가능 · `ref.*` 정답 함수 결과와 비교 금지)
- **방법**: 물리적으로 당연한 성질을 `assert`로. 예를 들어 이런 질문에서 시작하세요.
  - 로봇이 기본 자세로 서 있으면 관측 `q`는 어떤 값이어야 할까?
  - 기저가 똑바로 서 있으면 `gravity`는? 기저를 z축으로 90° 돌리면?
  - 이력에 프레임 번호를 값으로 넣으면 Decoder 입력 안에서 어떤 순서로 나타나야 할까?
  - `action = 0`이면 목표 관절각은? 한 관절만 1이면 **어떤 이름의** 관절이 움직여야 할까?
- **완료 기준**: 올바른 파이프라인은 통과, 버그 4유형을 각각 따로 넣은 변형은 모두 `AssertionError`
"""),
        code('''
# ✍️ Mission 2-2
import mujoco
from scipy.spatial.transform import Rotation as R

def invariant_tests(pipeline, lab):
    """참조 구현 없이 버그를 잡는 테스트. 문제가 있으면 AssertionError."""
    sim, c = lab.sim, lab.contract
    raise NotImplementedError

# from my_work.day2 import *
'''),
        code('''
# ✅ 확인 — 버그 변형 4종에 대해 내 테스트를 돌립니다
check.invariant_tests(lab, invariant_tests)
'''),
        md("""
---
## Mission 2-3 · 40초 완주와 영상 (20분)
고친 파이프라인으로 공식 샘플 전체(2,002프레임)를 실행하고, 참조 구현과 수치를 비교합니다.
같아야 할까요, 달라도 될까요? 다르다면 얼마나까지 괜찮을까요?
"""),
        code('''
pipeline = load_buggy()
fixed = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, pipeline=pipeline, verbose=True)
reference = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, verbose=True)
analysis.table({"내 파이프라인": fixed, "참조 구현": reference})
'''),
        code('''
# 영상 (GPU 런타임이면 EGL, CPU면 OSMesa 소프트웨어 렌더링 · 6초)
video = render.mesh_video(lab, fixed[1], "outputs/day2_fixed.mp4", label="fixed pipeline / mode 2", seconds=6)
render.show(video or render.skeleton_video(lab, fixed[1], "outputs/day2_fixed_skeleton.mp4", title="fixed", seconds=6))
'''),
        md("""
---
## 회고 (10분)
- 4개 버그 중 AI가 **먼저** 찾은 것과 **내가 먼저** 의심한 것은?
- "에러 없이 돌아가는 코드"가 왜 더 위험했나요?
- Mission 2-2의 테스트 중 실제 연구 코드에 그대로 남겨 두고 싶은 것은?
- **Day 3 예고**: 직접 실험을 설계하고 3분 발표합니다. 어떤 질문을 해 볼지 생각해 오세요(과제 아님).
"""),
    ]
    return name, cells


# ---------------------------------------------------------------------------
def day3():
    name = "day3_experiment.ipynb"
    cells = [md(f"""
# Day 3 · 실험 설계와 발표 — 기준 조건 위에서 질문하기
{badge(name)}

**오늘의 결과물**: 기준 실험 표 + 나만의 실험 1개(가설·지표·결과·한계) + 3분 발표

| 시간 | 내용 |
|---|---|
| 0:00–0:10 | 환경 준비 · 기준/통제 조건이란 |
| 0:10–0:35 | Mission 3-1 기준 실험 3조건 + 지표 추가 |
| 0:35–1:20 | Mission 3-2 자유 실험 |
| 1:20–1:30 | Mission 3-3 결과 영상 |
| 1:30–1:55 | 발표 (1인/1조 3분) |
| 1:55–2:00 | 마무리 체크리스트 |
"""),
        code(SETUP, hidden=True),
        md(LOG),
        md("""
---
## Mission 3-1 · 기준 실험 3조건 (25분)
실제 연구(06→07번 노트북)에서 원인을 가려낸 방법입니다. 같은 샘플·같은 시뮬레이터에서 세 조건을 비교합니다.

| 조건 | 정책 입력 | 의미 |
|---|---|---|
| `reference_pd` | 없음 — 참조 관절각을 PD로 그대로 재생 | 정책 없이도 걸을 수 있나? |
| `0` (G1 모드) | G1 참조 관절각 | 로봇 모션을 추종 |
| `2` (SMPL 모드) | 사람 관절 위치 | 사람 모션을 추종 |
"""),
        code('''
N = 1000   # 20초. 시간이 남으면 None(전체 40초)
baseline = {f"cond:{m}": ref.run_physics(lab.sim, lab.policy, lab.motion, mode=m, max_frames=N) for m in ["reference_pd", 0, 2]}
display(analysis.table(baseline))
analysis.plot(baseline, lab);
'''),
        md(f"""
{MISSION_RULE}

1. **해석**: `reference_pd`가 먼저 넘어지는 이유를 AI와 토론하고, 근거가 되는 수치(골반 높이 그래프 등)를 하나 찾으세요.
2. **지표 추가**: 관절 오차를 **부위별**(다리 12 / 허리 3 / 팔·손목 14, MuJoCo 순서 계약서 2절)로 나누는 함수를 바이브 코딩으로 만들고 모드 0과 2를 비교하세요.
"""),
        code('''
# ✍️ Mission 3-1 — 부위별 관절 오차 (deg)
def joint_error_by_group(arrays, lab):
    """반환 예: {"legs": 3.1, "waist": 1.2, "arms": 7.8}"""
    raise NotImplementedError

for label in ["cond:0", "cond:2"]:
    print(label, joint_error_by_group(baseline[label][1], lab))
'''),
        md("""
**해석 메모**: (여기에 3문장)
"""),
        md("""
---
## Mission 3-2 · 자유 실험 (45분)
아래에서 하나를 고르거나 직접 제안하세요. **실험 카드를 먼저 채우고** 코드를 작성합니다.

| 카드 | 질문 | 힌트 |
|---|---|---|
| A. 좌우 반전 | 반전 모션도 같은 품질로 추종할까? | `lab.load_motion(mirrored=True)` (공식 `_M` 샘플) |
| B. 외란 강건성 | 골반을 몇 N으로 밀면 넘어질까? | `run_physics(..., on_frame=fn)`에서 `sim.data.xfrc_applied[sim.pelvis_id, :3]` 설정. **힘을 끄는 것도 잊지 말 것** |
| C. 입력 잡음·지연 | 사람 관절 위치에 잡음 σ cm, 또는 k프레임 지연이 생기면? | `dataclasses.replace(lab.motion, local24=...)` |
| D. 재생 속도 | 0.8배/1.2배 속도 걷기도 따라갈까? | 시간축 재보간: 위치는 선형, 회전은 SLERP (계약서 1절) |
| E. 손목 정보 제거 | 손목 각도를 0으로 바꾸면(근사 데이터의 한계)? | `replace(lab.motion, wrists_il=np.zeros_like(...))` |
| F. 내 데이터 | 다른 동작 데이터도 될까? | 강사가 OpenSim NPZ를 주면 `ref.load_opensim_npz(path)` (근사 변환) |

`run_physics`는 `(summary, arrays)`를 돌려줍니다. 여러 조건을 `{"라벨": 결과}`로 모아 `analysis.table`, `analysis.plot`에 넣으세요.
"""),
        code('''
# 실험 카드 (먼저 채우기)
card = {
    "가설": "",
    "변인": "",
    "지표": "",
    "통제조건": "",
    "결과": "",
    "해석": "",
    "한계": "",
}
'''),
        code('''
# ✍️ Mission 3-2 — 실험 코드
import dataclasses
results = {"기준(mode 2)": ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=N)}

# results["내 조건"] = ...

display(analysis.table(results))
analysis.plot(results, lab);
'''),
        code('''
# ✅ 카드 확인 (결과·해석·한계까지 채운 뒤)
check.experiment_card(card)
'''),
        md("""
---
## Mission 3-3 · 결과 영상 (10분)
발표에 쓸 6초 영상을 만듭니다. 영상은 *보여주기*일 뿐, 판단은 표의 수치로 합니다.
"""),
        code('''
label = list(results)[-1]
video = render.mesh_video(lab, results[label][1], f"outputs/day3_{len(results)}.mp4", label=label, seconds=6)
render.show(video or render.skeleton_video(lab, results[label][1], "outputs/day3_skeleton.mp4", title=label, seconds=6))
'''),
        md("""
---
## 발표 (3분) — `templates/presentation.md`
1. 질문과 가설 → 2. 기준 vs 내 조건 (표 1개 + 그림/영상 1개) → 3. 해석과 한계 → 4. 바이브 코딩 회고 1줄

## 마무리 — `docs/03_checklist.md`
내 결과에 대해 체크리스트를 채울 수 있나요? 채울 수 없는 항목이 다음 연구의 첫 번째 할 일입니다.
"""),
    ]
    return name, cells


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for build in (day1, day2, day3):
        name, cells = build()
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {"colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}
        nbf.write(nb, out / name)
        print("wrote", out / name, len(cells), "cells")


if __name__ == "__main__":
    main()
