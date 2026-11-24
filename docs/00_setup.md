# 시작하기

## 1. Colab에서 열기 (기본 경로)

README의 **Open in Colab** 배지를 누르고 맨 위 `0. 실습 환경 준비` 셀을 실행합니다.
이 셀이 저장소를 `/content/sonic-vibe-coding-lab`에 받고, 필요한 패키지를 설치하고, 공식 자산(약 100 MB)을 내려받습니다. 처음 1회 1–2분 걸립니다.

- **런타임 유형**: CPU로 충분합니다. 정책 추론(ONNX CPU)과 물리 시뮬레이션은 CPU만 씁니다. GPU 런타임은 Day 3 메쉬 영상 렌더링만 빨라집니다.
- **런타임이 끊기면** `/content`가 지워집니다. 0번 셀부터 다시 실행하세요(자산은 다시 받음).
- **Day 2 버그 사냥 파일**은 Colab 왼쪽 파일 탐색기에서 `sonic-vibe-coding-lab/exercises/buggy_pipeline.py`를 더블클릭해 편집합니다. 런타임이 끊기면 수정 내용도 사라지므로, 중간중간 노트북 셀에 복사해 두세요.

## 2. 바이브 코딩 도구 연결 — 세 가지 방식

### A. 웹 채팅형 (ChatGPT, Claude, Gemini 등)
1. `docs/01_contract.md` 전문을 대화 첫 메시지에 붙여 넣습니다.
2. 미션 셀의 설명과 함수 시그니처를 붙여 넣고 구현을 요청합니다.
3. 받은 코드를 `✍️` 셀에 붙여 실행 → 체크 셀 출력을 다시 대화에 붙여 넣습니다.

### B. Colab 내장 AI
셀의 AI 기능에 미션 설명과 "docs/01_contract.md의 규칙을 따른다"를 함께 적습니다. 계약서 내용을 직접 붙여 주는 편이 정확합니다.

### C. 에이전트형 (Claude Code, Codex, Cursor 등) — 로컬 편집
1. 이 저장소를 **Fork** → 로컬에 clone.
2. 에이전트는 저장소의 `AGENTS.md`/`CLAUDE.md`에서 규칙을 읽습니다(정답 파일을 읽지 않기 등).
3. 미션 코드는 `my_work/day1.py`, `day2.py`, `day3.py`에 작성합니다. 노트북의 `✍️` 셀에서 `from my_work.day1 import *` 줄의 주석을 해제하면 불러옵니다.
4. 실행 방법 둘 중 하나
   - **Colab**: Fork를 push하고, 노트북 0번 셀의 `REPO`를 내 Fork 주소로 바꿔 실행.
   - **로컬**(Ubuntu 22.04 기준): 아래 conda 환경에서 Jupyter/VS Code로 `notebooks/`를 실행. 코드가 CPU만 쓰므로 그대로 동작합니다.

```bash
conda env create -f environment.yml
conda activate sonic-lab
jupyter lab        # 또는 VS Code에서 notebooks/*.ipynb 열고 커널 sonic-lab 선택
```
메쉬 영상 렌더링을 로컬에서 하려면 `sudo apt-get install -y libosmesa6`(CPU) 또는 NVIDIA 드라이버의 EGL이 필요합니다. 없으면 `skeleton_video()`를 쓰면 됩니다.

## 3. 규칙
- `sonic_lab/reference.py`, `sonic_lab/check.py`에는 정답이 있습니다. 열지 않습니다.
- 체크 셀이 ❌여도 괜찮습니다. 메시지를 근거로 AI와 원인을 좁혀 가는 과정이 실습입니다.
- 막히면 강사가 `lab.policy` 등 참조 객체로 우회하는 방법을 알려 줍니다. 다음 미션은 참조 구현으로 이어서 진행할 수 있게 설계되어 있습니다.
