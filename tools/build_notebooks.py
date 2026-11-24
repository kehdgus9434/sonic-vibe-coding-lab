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


def predict(question):
    """실행 전에 사람이 직접 적는 예측 칸 (에이전트에게 맡기지 않음)."""
    return md(f"""
> 🔮 **예측 — 아래 셀을 실행하기 전에 직접 적으세요 (AI에게 맡기지 않기)**
> {question}
>
> 내 예측: ______
""")


def verify(question):
    """체크 통과 뒤 사람이 적는 검증 기록."""
    return md(f"""
> 🧾 **검증 기록 — 직접 적으세요**
> {question}
>
> 근거: ______
""")


SETUP = f'''
#@title 0. 실습 환경 준비 — 처음 한 번 실행 (약 1–2분)
REPO = "https://github.com/{REPO_SLUG}"   # Fork로 작업하면 내 Fork 주소로 바꾸세요
import os, sys, subprocess
from pathlib import Path
if Path("/content").exists():                              # Google Colab
    LAB_DIR = Path("/content/sonic-vibe-coding-lab")
    if not LAB_DIR.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(LAB_DIR)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(LAB_DIR / "requirements-colab.txt")], check=True)
else:                                                      # 로컬 Jupyter / VS Code (저장소 안에서 실행)
    LAB_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(LAB_DIR)
sys.path.insert(0, str(LAB_DIR))
import numpy as np
import sonic_lab
from sonic_lab import check, reference as ref, render, analysis
lab = sonic_lab.start()
'''

AGENT_POLICY = """
**AI 사용 규칙 (이 과정은 AI 에이전트 사용을 전제로 합니다)**
- 코드는 어떤 도구로 써도 됩니다. Colab MCP·VS Code 확장으로 에이전트가 노트북을 직접 실행해도 됩니다(`docs/00_setup.md`).
- 대신 **평가는 코드가 아니라 당신의 판단**입니다: 🔮 예측(실행 전), 🧾 검증 기록(체크 후), 마지막 📮 제출 카드는 **본인이 직접** 씁니다.
- `sonic_lab/reference.py`·`check.py`는 정답입니다. 에이전트에게도 읽지 말라고 지시하세요(`AGENTS.md`에 규칙이 있음).
"""

REFLECTION = '''
# 📮 오늘의 제출 카드 — 본인이 직접 작성 (에이전트 작성 금지)
card = {
    "예측과 실제가 달랐던 것": "",
    "AI가 틀렸던 순간과 알아챈 방법": "",
    "가장 믿을 만한 ✅와 그 근거": "",
    "정답 비교 없이 내가 확인한 것": "",
}
check.reflection(card)
'''

MISSION_RULE = "> 🧭 아래 설명 + `docs/01_contract.md` + 함수 시그니처를 AI에게 주고 ✍️ 셀을 채운 뒤 ✅ 셀로 확인합니다."


# ---------------------------------------------------------------------------
def day1():
    name = "day1_anatomy.ipynb"
    cells = [md(f"""
# Day 1 · SONIC 해부하기 — 모델과 데이터의 '계약' 읽기
{badge(name)}

**오늘의 결과물**: 공식 걷기 샘플 → *내가 만든* Encoder 입력(1762) → 64차원 토큰 시계열 + 제출 카드

| 시간 | 내용 |
|---|---|
| 0:00–0:15 | 환경 준비 · 목적지 미리보기 · 도구 연결 |
| 0:15–0:30 | Mission 1-1 ONNX 입출력 해부 |
| 0:30–0:50 | Mission 1-2 Encoder 1762차원 블록 표 |
| 0:50–1:10 | Mission 1-3 쿼터니언 순서 탐정 |
| 1:10–1:25 | Mission 1-4 30 Hz ↔ 50 Hz 시간 정렬 |
| 1:25–1:50 | Mission 1-5 Mode-2 Encoder 입력 조립 → 토큰 |
| 1:50–2:00 | 📮 제출 카드 · (빨리 끝나면 🚀 심화) |

{AGENT_POLICY}
⚠️ 이 실습의 어떤 출력도 실물 로봇에 보내지 않습니다.
"""),
        code(SETUP, hidden=True),
        md("""
## 목적지 미리보기 (5분)
참조 구현으로 3일 뒤 도달할 결과를 먼저 봅니다: **사람 걷기 데이터 → SONIC → MuJoCo G1**.
"""),
        predict("사람(SMPL) 걷기 데이터를 SONIC에 넣으면, 처음 5초(250프레임) 동안 G1은 넘어지지 않을까요? 넘어진다면 몇 초쯤?"),
        code('''
summary, arrays = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=250)
print({k: summary[k] for k in ["simulated_frames", "requested_frames", "completed", "stop_reason", "runtime_s"]})
render.show(render.skeleton_video(lab, arrays, "outputs/day1_preview.mp4", title="reference pipeline", seconds=5))
'''),
        md(f"""
---
## Mission 1-1 · ONNX 입출력 해부 (15분)
{MISSION_RULE}

- **목표**: `model_encoder.onnx`, `model_decoder.onnx`의 입력·출력 이름과 shape를 확인한다.
- **주어진 것**: `lab.assets.model_dir` (ONNX 2개 + `observation_config.yaml`)
- **완료 기준**: ✅ 셀 통과
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
        verify("`observation_config.yaml` 맨 위 주석의 *Total dimension* 값과 실제 Decoder 입력 길이가 같나요? 어느 쪽을 믿어야 하고, 왜인가요?"),
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
        verify("AI가 블록 길이를 정할 때 계약서에 **없는** 규칙을 지어낸 적이 있었나요? `vr_3point_local_orn_target`이 왜 12인지 직접 설명할 수 있나요?"),
        md(f"""
---
## Mission 1-3 · 쿼터니언 순서 탐정 (20분)
{MISSION_RULE}

공식 G1 샘플에는 루트 회전이 **두 가지 표현**으로 들어 있습니다: `pose_aa[:, 0]`(축-각 벡터)과 `root_rot`(쿼터니언).
실제 연구 과정(06번 노트북)에서 `root_rot`를 잘못된 성분 순서로 읽어 큰 회전 오류가 난 적이 있습니다.

- **목표**: 두 표현을 비교해 `root_rot`가 `xyzw`인지 `wxyz`인지 **데이터로** 판정하는 함수를 만든다.
- **요청 팁**: "어느 쪽일 것 같아?"가 아니라 "두 가정 각각의 최대 각도 오차를 계산해서 비교하는 코드"를 요청하세요.
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
        predict("성분 순서를 잘못 읽으면 최대 각도 오차는 몇 도쯤 될까요? (5°? 45°? 180°?)"),
        code('''
# ✍️ Mission 1-3
def detect_root_quat_order(robot_record):
    """반환: 'xyzw' 또는 'wxyz' (판정 근거가 되는 두 오차도 출력할 것)"""
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

`lab.motion`은 이미 50 Hz로 정렬된 공식 샘플입니다(앞 미션을 못 끝냈어도 진행 가능).

| 필드 | shape | 의미 |
|---|---|---|
| `lab.motion.local24` | (T, 24, 3) | 루트 기준 SMPL 관절 좌표 |
| `lab.motion.human_root_xyzw` | (T, 4) | 인체 루트 회전 (scipy xyzw) |
| `lab.motion.wrists_il` | (T, 6) | G1 손목 6관절 (IsaacLab 23–28) |
| `lab.motion.times` | (T,) | 50 Hz 시간축 |

- **목표**: 계약서 3절대로 `frame` 시점의 1762차원 Mode-2 Encoder 입력을 만든다. `robot_xyzw`는 현재 로봇 기저 회전(xyzw), `alignment`는 scipy `Rotation`(헤딩 정렬 A)입니다.
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
        verify("6D 방향 블록에서 회전행렬의 '열'과 '행'을 바꿔 넣어도 체크를 통과할까요? 왜 그렇게 생각하는지, 혹은 직접 바꿔 본 결과를 적으세요."),
        predict("걷기 샘플의 토큰 64차원 중 일부는 걸음 주기(약 1초)를 따라 주기적으로 움직일까요?"),
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
        md("---\n## 📮 제출 카드 (10분)\n오늘 실습에서 **내가 판단한 것**을 적습니다. 코드 품질은 평가하지 않습니다."),
        code(REFLECTION),
        md(f"""
---
## 🚀 심화 미션 (빨리 끝난 경우, 정답 비교가 있는 것과 없는 것)

**A. Mode-0 Encoder 입력 (체크 있음)** — G1 참조 관절각으로 Encoder 입력을 만듭니다(계약서 3절의 `step5` 블록).
`lab.motion.g1_q`(MuJoCo 순서 관절각), `lab.motion.g1_dq`(관절 속도, 제공), `lab.motion.g1_root_xyzw`(G1 루트 회전)를 씁니다.

**B. 좌우 반전 샘플의 토큰 (정답 없음)** — `lab.load_motion(mirrored=True)`로 반전 샘플을 불러 원본과 토큰을 비교하세요. 토큰 공간에서 '반전'이 보이나요? 무엇으로 그렇게 판단하나요?

**C. 공식 파일 주석의 출처 (정답 없음)** — YAML 주석의 `436 (64+12+116+116+116+12)`은 어떤 구성이었을까요? 29관절이 아니라면 몇 관절일까요? 공식 저장소에서 근거를 찾아보세요.
"""),
        code('''
# 🚀 A. ✍️ Mode-0 Encoder 입력
def build_encoder_obs_mode0(motion, frame, robot_xyzw, alignment):
    raise NotImplementedError

# from my_work.day1 import *
# check.encoder_obs_mode0(lab, build_encoder_obs_mode0)
'''),
    ]
    return name, cells


# ---------------------------------------------------------------------------
def day2():
    name = "day2_bughunt.ipynb"
    cells = [md(f"""
# Day 2 · 버그 사냥 — 정답 없이 버그를 잡을 수 있나?
{badge(name)}

**오늘의 결과물**: 내 불변식 테스트 → 그 테스트로 고친 파이프라인 → (마지막에만) 정답 비교 → 40초 완주 영상 + 제출 카드

실제 연구에는 정답 코드가 없습니다. 그래서 오늘은 **테스트를 먼저 만들고, 그 테스트만으로 버그를 고친 뒤, 마지막에 한 번 정답과 비교**합니다.

| 시간 | 내용 |
|---|---|
| 0:00–0:10 | 환경 준비 · 증상 관찰 |
| 0:10–0:45 | Mission 2-1 불변식 테스트 만들기 (정답 비교 없음) |
| 0:45–1:20 | Mission 2-2 내 테스트로 버그 고치기 (정답 비교 없음) |
| 1:20–1:35 | Mission 2-3 최종 판정 — 정답 비교(오라클) |
| 1:35–1:50 | Mission 2-4 40초 완주와 영상 |
| 1:50–2:00 | 📮 제출 카드 · (빨리 끝나면 🚀 심화) |

{AGENT_POLICY}
- 오늘은 추가로: `check.bughunt`·`check.invariant_tests`(정답 비교)는 **Mission 2-3 전에는 부르지 않습니다.** 호출 횟수가 기록되어 제출 카드에 함께 남습니다.
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
AI가 작성한 파이프라인(`exercises/buggy_pipeline.py`)은 오류 없이 실행되고, 주석도 그럴듯합니다. 파일을 한 번 훑어본 뒤 예측하세요.
"""),
        predict("이 파이프라인으로 G1은 몇 프레임(50 프레임 = 1초)쯤 버틸까요?"),
        code('''
bug_summary, bug_arrays = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=250, pipeline=pipeline)
print({k: bug_summary[k] for k in ["simulated_frames", "requested_frames", "stop_reason"]})
render.show(render.skeleton_video(lab, bug_arrays, "outputs/day2_buggy.mp4", title="buggy pipeline", seconds=5))
'''),
        md(f"""
---
## Mission 2-1 · 불변식 테스트 만들기 — 정답 비교 없음 (35분)
{MISSION_RULE}

- **목표**: `pipeline`의 세 함수(`observation`, `decoder_obs`, `action_to_target`)가 계약서를 지키는지 확인하는 `assert`를 **4개 이상** 만든다.
- **금지**: `ref.*`, `lab.policy.decoder_obs` 같은 정답 함수의 출력과 비교하기. 쓸 수 있는 것은 `lab.sim`(상태를 직접 설정), `lab.contract`(기본각·스케일·관절 순서), 계약서뿐입니다.
- **질문에서 시작하세요**
  - 로봇이 기본 자세로 똑바로 서 있으면 관측 `q`는? `gravity`는? 기저를 x축으로 90° 기울이면?
  - 이력에 **프레임 번호를 값으로** 넣으면 Decoder 입력 안에서 어떤 순서로 나타나야 할까?
  - `action = 0`이면 목표 관절각은? 한 관절만 1이면 **어떤 이름의** 관절이 움직여야 할까? (`lab.sim.model.joint(id).name`, `lab.sim.jids`)
- **완료 기준**: 아래 실행 셀에서 내 테스트가 **현재 버그 파이프라인에서 실패**한다(= 버그를 하나 이상 잡았다).
"""),
        code('''
# ✍️ Mission 2-1
import mujoco
from scipy.spatial.transform import Rotation as R

def invariant_tests(pipeline, lab):
    """정답 구현 없이 버그를 잡는 테스트. 문제가 있으면 AssertionError (메시지에 무엇이 틀렸는지)."""
    sim, c = lab.sim, lab.contract
    raise NotImplementedError

# from my_work.day2 import *
'''),
        code('''
# ▶ 내 테스트 실행 (정답 비교 아님) — 첫 번째로 실패한 assert를 보여 줍니다
def run_my_tests(p):
    try:
        invariant_tests(p, lab); print("내 테스트 모두 통과")
    except AssertionError as error:
        print("❗ 내 테스트 실패:", error)
    finally:
        lab.sim.reset(lab.motion)
run_my_tests(load_buggy())
'''),
        md(f"""
---
## Mission 2-2 · 내 테스트로 버그 고치기 — 정답 비교 없음 (35분)

- **대상**: `exercises/buggy_pipeline.py` (Colab 왼쪽 📁 → `sonic-vibe-coding-lab/exercises/`에서 더블클릭해 편집, 또는 에이전트가 직접 수정)
- 버그는 **4개**입니다. 내 테스트가 실패를 보여 주면 → 원인 후보를 계약서 조항과 연결 → 고침 → 파일 맨 위 `BUGLOG`에 "무엇을, 어떤 근거로" 기록.
- 테스트가 모두 통과하는데도 의심되면 **테스트를 늘리세요.** 짧은 폐루프 실행도 근거가 됩니다(아래 셀).
- ⚠️ Colab 런타임이 끊기면 파일 수정도 사라집니다. 고친 코드를 수시로 아래 메모 셀에 복사해 두세요.
"""),
        code('''
# ▶ 고칠 때마다: 내 테스트 + 5초 폐루프 (정답 비교 아님)
pipeline = load_buggy()
run_my_tests(pipeline)
s, _ = ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=250, pipeline=pipeline)
print("폐루프:", s["simulated_frames"], "/ 250 프레임,", s["stop_reason"])
'''),
        md("""
```python
# 📋 고친 코드 백업용 메모 (런타임이 끊기면 여기서 복구)
```
"""),
        predict("버그를 다 고쳤다고 생각하나요? 정답과 비교하면 4개 중 몇 개를 맞혔을까요? 내 테스트가 놓쳤을 것 같은 버그가 있다면?"),
        md("""
---
## Mission 2-3 · 최종 판정 — 정답 비교(오라클) (15분)
이제 처음으로 정답과 비교합니다. 두 가지를 봅니다.
1. `check.bughunt`: 내가 고친 파이프라인이 정답과 같은가?
2. `check.invariant_tests`: 내 테스트가 버그 4유형을 **각각 따로** 넣었을 때 몇 개를 잡는가?
"""),
        code('''
# ✅ 최종 판정 (호출 횟수 기록됨)
_ = check.bughunt(lab, load_buggy(), frames=250)
check.invariant_tests(lab, invariant_tests)
'''),
        verify("넘어지지 않는데도 남아 있던 버그가 있었나요? 그 버그는 폐루프 결과(완주 여부)로는 왜 드러나지 않았을까요? 내 테스트가 놓친 버그 유형과 그 이유는?"),
        md("""
---
## Mission 2-4 · 40초 완주와 영상 (15분)
고친 파이프라인으로 공식 샘플 전체(2,002프레임)를 실행하고 참조 구현과 비교합니다.
"""),
        predict("내 파이프라인과 참조 구현의 관절 오차(°)는 완전히 같을까요, 조금 다를까요? 왜?"),
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
        md("---\n## 📮 제출 카드 (10분)\n정답 비교 호출 횟수도 함께 출력됩니다. 횟수가 많다고 감점이 아니라, 판단 과정을 돌아보는 자료입니다."),
        code(REFLECTION),
        md("""
---
## 🚀 심화 미션 (정답 없음)

**A. 버그 민감도 지도** — 고친 파이프라인에 버그를 **하나씩만** 다시 넣은 변형 4개를 직접 만들어, 각각 몇 프레임 버티는지와 관절 오차를 표로 만드세요. 어떤 버그가 가장 '조용한' 버그인가요? 조용한 버그를 잡으려면 어떤 지표를 봐야 할까요?

**B. 무작위 상태 테스트** — 기저 회전·관절각·속도를 무작위로 수백 번 바꿔 가며 불변식을 검사하는 property-based 테스트로 확장하세요. 고정된 몇 개 상태만 쓰는 테스트가 놓칠 수 있는 버그를 예로 들어 보세요.

**C. 이력 초기화** — 시작할 때 이력 10프레임을 0으로 채우면 어떻게 될까요? 계약서 4절의 초기화 규칙과 비교하고, 그 차이가 처음 1초 동안 결과에 어떤 영향을 주는지 측정하세요.
"""),
        code('''
# 🚀 심화 작업 공간
'''),
    ]
    return name, cells


# ---------------------------------------------------------------------------
def day3():
    name = "day3_experiment.ipynb"
    cells = [md(f"""
# Day 3 · 실험 설계와 발표 — 기준 조건 위에서 질문하기
{badge(name)}

**오늘의 결과물**: 기준 실험 표 + 나만의 실험 1개(가설·지표·결과·한계) + 3분 발표 + 제출 카드

| 시간 | 내용 |
|---|---|
| 0:00–0:10 | 환경 준비 · 기준/통제 조건이란 |
| 0:10–0:35 | Mission 3-1 기준 실험 3조건 + 지표 추가 |
| 0:35–1:20 | Mission 3-2 자유 실험 |
| 1:20–1:30 | Mission 3-3 결과 영상 |
| 1:30–1:55 | 발표 (1인/1조 3분) |
| 1:55–2:00 | 📮 제출 카드 |

{AGENT_POLICY}
- 오늘은 추가로: 실험 카드의 **가설은 코드를 돌리기 전에 본인이** 씁니다. 해석·한계도 본인 판단으로 씁니다(AI 의견은 참고만).
"""),
        code(SETUP, hidden=True),
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
        predict("세 조건 중 어느 것이 넘어질까요? 모드 0과 모드 2 중 어느 쪽 관절 오차가 더 클까요? 이유는?"),
        code('''
N = 1000   # 20초. 시간이 남으면 None(전체 40초)
baseline = {f"cond:{m}": ref.run_physics(lab.sim, lab.policy, lab.motion, mode=m, max_frames=N) for m in ["reference_pd", 0, 2]}
display(analysis.table(baseline))
analysis.plot(baseline, lab);
'''),
        md(f"""
{MISSION_RULE}

1. **해석**: `reference_pd`가 먼저 넘어지는 이유를 AI와 토론하되, **근거가 되는 수치**(골반 높이 그래프 등)는 직접 찾으세요.
2. **지표 추가**: 관절 오차를 **부위별**(다리 12 / 허리 3 / 팔·손목 14, MuJoCo 순서 계약서 2절)로 나누는 함수를 만들고 모드 0과 2를 비교하세요.
"""),
        code('''
# ✍️ Mission 3-1 — 부위별 관절 오차 (deg)
def joint_error_by_group(arrays, lab):
    """반환 예: {"legs": 3.1, "waist": 1.2, "arms": 7.8}"""
    raise NotImplementedError

for label in ["cond:0", "cond:2"]:
    print(label, joint_error_by_group(baseline[label][1], lab))
'''),
        verify("예측과 실제가 달랐다면 무엇이 달랐나요? 부위별 오차에서 모드 2가 특히 나쁜 부위가 있다면, 입력 데이터의 어떤 한계와 관련 있을까요?"),
        md("""
---
## Mission 3-2 · 자유 실험 (45분)
아래에서 하나를 고르거나 직접 제안하세요. **실험 카드의 가설을 먼저 채우고** 코드를 작성합니다.

| 카드 | 질문 | 힌트 |
|---|---|---|
| A. 좌우 반전 | 반전 모션도 같은 품질로 추종할까? | `lab.load_motion(mirrored=True)` (공식 `_M` 샘플) |
| B. 외란 강건성 | 골반을 얼마나 세게·오래 밀면 넘어질까? | `run_physics(..., on_frame=fn)`에서 `sim.data.xfrc_applied[sim.pelvis_id, :3]` 설정. **힘을 끄는 것도 잊지 말 것.** 수백 N을 잠깐 미는 정도로는 버팁니다 — 크기와 지속시간을 함께 바꿔 경계를 찾으세요 |
| C. 입력 잡음·지연 | 사람 관절 위치에 잡음 σ cm, 또는 k프레임 지연이 생기면? | `dataclasses.replace(lab.motion, local24=...)` |
| D. 재생 속도 | 0.8배/1.2배 속도 걷기도 따라갈까? | 시간축 재보간: 위치는 선형, 회전은 SLERP (계약서 1절) |
| E. 손목 정보 제거 | 손목 각도를 0으로 바꾸면(근사 데이터의 한계)? | `replace(lab.motion, wrists_il=np.zeros_like(...))` |
| F. 내 데이터 | 다른 동작 데이터도 될까? | 강사가 OpenSim NPZ를 주면 `ref.load_opensim_npz(path)` (근사 변환) |

`run_physics`는 `(summary, arrays)`를 돌려줍니다. 여러 조건을 `{"라벨": 결과}`로 모아 `analysis.table`, `analysis.plot`에 넣으세요.
"""),
        code('''
# 실험 카드 — "가설"은 코드를 돌리기 전에 본인이 작성
card3 = {
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
check.experiment_card(card3)
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
1. 질문과 **실행 전 가설** → 2. 기준 vs 내 조건 (표 1개 + 그림/영상 1개) → 3. 해석과 한계 → 4. AI가 틀렸던 순간 1개

## 📮 제출 카드와 마무리 — `docs/03_checklist.md`
"""),
        code(REFLECTION),
        md("""
---
## 🚀 심화 미션 (정답 없음)

**A. 외란 경계 지도** — 힘(N) × 지속시간(s) 격자에서 넘어짐 여부를 지도로 그리세요. 이진 탐색으로 경계를 효율적으로 찾는 코드를 AI와 만들고, 몇 번의 시뮬레이션으로 찾았는지 기록하세요.

**B. 반전 × 모드** — 좌우 반전 샘플을 모드 0과 모드 2로 각각 돌려 원본과 비교하세요. 정책이 좌우 대칭적으로 학습되었다고 말할 수 있나요? 그 주장의 근거와 한계는?

**C. 재현성** — 같은 조건을 여러 번 돌리면 결과가 완전히 같나요? 스레드 수(`OMP_NUM_THREADS`)나 실행 순서가 바뀌면? 차이가 있다면 그 크기를 결과 해석에 어떻게 반영해야 할까요?
"""),
        code('''
# 🚀 심화 작업 공간
'''),
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
