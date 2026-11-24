# 시작하기

이 과정은 **AI 에이전트를 쓰는 것을 전제로** 합니다. 코드는 어떤 도구로 써도 됩니다.
평가는 코드가 아니라 **당신의 판단**입니다. 매 Day의 🔮 예측(실행 전), 🧾 검증 기록(체크 후), 📮 제출 카드는 본인이 직접 씁니다.

## 1. Colab에서 열기 (기본 경로)

README의 **Open in Colab** 배지를 누르고 맨 위 `0. 실습 환경 준비` 셀을 실행합니다.
이 셀이 저장소를 `/content/sonic-vibe-coding-lab`에 받고, 필요한 패키지를 설치하고, 공식 자산(약 125 MB)을 내려받습니다. 처음 1회 1–2분 걸립니다.

- **런타임 유형**: CPU로 충분합니다. 정책 추론(ONNX CPU)과 물리 시뮬레이션은 CPU만 씁니다. GPU 런타임은 메쉬 영상 렌더링만 빨라집니다.
- **런타임이 끊기면** `/content`가 지워집니다. 0번 셀부터 다시 실행하세요.
- **Day 2 버그 사냥 파일**은 Colab 왼쪽 파일 탐색기에서 `sonic-vibe-coding-lab/exercises/buggy_pipeline.py`를 더블클릭해 편집합니다. 런타임이 끊기면 수정 내용도 사라지므로 수시로 노트북 메모 셀에 복사해 두세요.

## 2. AI 도구 연결 — 네 가지 방식

### A. 웹 채팅형 (ChatGPT, Claude, Gemini 등)
1. `docs/01_contract.md` 전문을 대화 첫 메시지에 붙여 넣습니다.
2. 미션 설명과 함수 시그니처를 붙여 넣고 구현을 요청합니다.
3. 받은 코드를 `✍️` 셀에 붙여 실행 → 체크 출력을 다시 대화에 붙여 넣습니다.

### B. Colab 내장 AI
셀의 AI 기능에 미션 설명과 "docs/01_contract.md의 규칙을 따른다"를 함께 적습니다. 계약서 내용을 직접 붙여 주는 편이 정확합니다.

### C. 에이전트가 Colab 노트북을 직접 조작 — Colab MCP 서버 (선택)
Google의 [Colab MCP 서버](https://github.com/googlecolab/colab-mcp)를 쓰면 **내 컴퓨터에서 실행 중인 에이전트**(Claude Code, Gemini CLI 등)가 브라우저의 Colab 노트북에 셀을 만들고 실행할 수 있습니다. 클라우드에서 도는 에이전트(웹 채팅 등)는 쓸 수 없습니다.

1. `uv` 설치: `pip install uv` (서버가 Python 3.13 이상을 요구하지만 `uvx`가 알아서 받습니다)
2. 첫 실행이 느리므로 미리 한 번 받아 두기: `uvx git+https://github.com/googlecolab/colab-mcp --help`
3. 에이전트에 등록
   - Claude Code: `claude mcp add colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp` → `claude mcp list`로 확인
   - Codex(공식 호환 목록에 없음, 동작 미보장): `~/.codex/config.toml`에
     ```toml
     [mcp_servers.colab-mcp]
     command = "uvx"
     args = ["git+https://github.com/googlecolab/colab-mcp"]
     startup_timeout_sec = 60
     ```
4. 에이전트에게 "Colab에 연결해 줘"라고 하면 `open_colab_browser_connection` 도구가 브라우저에 **새 빈 Colab 노트북**을 엽니다. 60초 안에 연결되면 노트북 편집·실행 도구가 나타납니다(한 번에 브라우저 탭 1개만 연결).
5. ⚠️ 연결되는 곳은 **새 빈 노트북**입니다. 이 과정의 노트북(day1 등)을 바로 조작하지 않습니다. 에이전트에게 Day 노트북의 `0. 실습 환경 준비` 코드를 그 노트북에서 실행하게 한 뒤 미션 코드를 맡기고, 🔮/🧾/📮은 과정 노트북(다른 탭)이나 `templates/`에 본인이 씁니다. 과정 노트북 자체에서 에이전트와 작업하려면 D의 VS Code + Colab 확장이 더 단순합니다.

### D. 에이전트형 도구로 로컬 편집 (Claude Code, Codex, Cursor 등)
1. 이 저장소를 **Fork** → 로컬에 clone. 에이전트는 `AGENTS.md`/`CLAUDE.md`에서 실습 규칙을 자동으로 읽습니다.
2. 미션 코드는 `my_work/day1.py`, `day2.py`, `day3.py`에 작성하고, 노트북 `✍️` 셀의 `from my_work.dayN import *` 주석을 해제해 불러옵니다.
3. 실행은 셋 중 하나
   - **VS Code + Colab 확장**: Google의 공식 Colab 확장을 설치하면 VS Code에서 연 노트북의 커널로 Colab 런타임을 고를 수 있습니다(`Select Kernel → Colab`).
   - **Colab 웹**: Fork를 push하고 노트북 0번 셀의 `REPO`를 내 Fork 주소로 바꿔 실행.
   - **로컬**(Ubuntu 22.04 기준): 아래 conda 환경에서 Jupyter/VS Code로 `notebooks/` 실행. CPU만 쓰므로 그대로 동작합니다.

```bash
conda env create -f environment.yml
conda activate sonic-lab
jupyter lab
```
로컬에서 메쉬 영상을 만들려면 `sudo apt-get install -y libosmesa6`(CPU) 또는 NVIDIA 드라이버의 EGL이 필요합니다. 없으면 `skeleton_video()`를 씁니다.

## 3. 규칙
- `sonic_lab/reference.py`, `sonic_lab/check.py`에는 정답이 있습니다. 본인도, 에이전트도 열지 않습니다.
- **🔮 예측은 실행 전에, 🧾 검증 기록과 📮 제출 카드는 본인이 직접** 씁니다. 이것이 평가 대상입니다.
- Day 2의 정답 비교(`check.bughunt`, `check.invariant_tests`)는 Mission 2-3에서만 부릅니다. 호출 횟수가 기록됩니다.
- 빨리 끝났다면 각 노트북 끝의 🚀 심화 미션으로 넘어갑니다. 정답이 없는 문제가 대부분입니다.
