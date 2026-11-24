# 시작하기 — 브라우저만으로 (공용 컴퓨터실 Windows · 개인 노트북)

설치할 것은 없습니다. **브라우저(Chrome 권장)** 하나로 Colab과 AI 에이전트를 씁니다.
이 과정은 AI 에이전트 사용을 전제로 하며, 평가는 코드가 아니라 **당신의 판단**입니다.
🔮 예측(실행 전), 🧾 검증 기록(체크 후), 📮 제출 카드는 본인이 직접 씁니다.

## 0. 준비물

| 계정 | 용도 | 필수 |
|---|---|---|
| Google | Colab 실행, Colab의 Gemini | ✅ |
| GitHub | 웹 코딩 에이전트(C 방식)를 쓸 때 Fork | C 방식만 |
| Claude(Pro 이상) 또는 ChatGPT(Codex 포함 요금제) | 웹 코딩 에이전트 | C 방식만 |

**공용 PC 주의**: Chrome **게스트 창**이나 시크릿 창에서 로그인하고, 수업이 끝나면 Google·GitHub·Claude/ChatGPT에서 모두 **로그아웃**하세요. 비밀번호 저장 팝업은 "저장 안 함".

## 1. Colab에서 노트북 열기 (모두 공통)

1. README의 **Open in Colab** 배지 클릭 → Google 로그인 → 상단 **"Drive에 사본 저장"**(내 기록이 남도록).
2. 맨 위 `0. 실습 환경 준비` 셀 실행(처음 1–2분). 저장소·패키지·공식 자산(약 125 MB)이 Colab 서버에 준비됩니다.
3. 런타임 유형은 **CPU**로 충분합니다. 런타임이 끊기면 0번 셀부터 다시 실행합니다.

모든 계산은 Colab 서버에서 하므로 PC 사양이나 운영체제(Windows/macOS)와 상관없습니다.

## 2. AI 에이전트 — 세 가지 방식 (설치 없음)

### A. Colab의 Gemini — 가장 간단, 노트북 안에서 바로
Colab 화면 아래쪽 **Gemini(✦) 버튼**으로 대화하면 Gemini가 노트북의 여러 셀을 읽고, 코드를 만들고, 오류를 고칩니다.
- 처음 지시 예: "이 노트북은 SONIC 실습이다. 저장소의 `docs/01_contract.md` 규칙만 근거로 하고, 🔮/🧾/📮 셀은 절대 작성하지 마."
- 계약서 내용을 대화에 직접 붙여 주면 더 정확합니다(왼쪽 📁 → `sonic-vibe-coding-lab/docs/01_contract.md`를 열어 복사).
- 계정 연령·국가에 따라 일부 기능이 보이지 않을 수 있습니다. 그때는 B로.

### B. 웹 채팅 (ChatGPT, Claude, Gemini) — 어디서나 동작
1. 다른 탭에서 채팅을 열고 `docs/01_contract.md` 전문을 첫 메시지로 붙여 넣습니다.
2. 미션 설명과 함수 시그니처를 붙여 넣고 구현을 요청 → 받은 코드를 `✍️` 셀에 붙여 실행 → 체크 출력을 다시 채팅에 붙여 넣습니다.

### C. 웹 코딩 에이전트 + GitHub 자동 리포트 (Claude Code on the web, Codex) — 가장 강력, 준비 10분
에이전트가 내 GitHub Fork의 브랜치에 코드를 push하면, GitHub이 **실제 모델로 자동 실행**해서 체크 표·그래프·로봇 GIF를 PR 댓글로 보여 주고,
**"Colab에서 내 브랜치로 열기"** 링크(REPO/BRANCH가 채워진 노트북)를 줍니다. 자세한 흐름: [`docs/04_web_agent_workflow.md`](04_web_agent_workflow.md)

> Colab MCP 서버(로컬 에이전트가 브라우저 Colab을 조작)는 **내 컴퓨터에 프로그램 설치가 필요**하므로 공용 컴퓨터실에서는 쓰지 않습니다. 개인 노트북에서 쓰려면 [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp)를 참고하세요.

## 3. Day 2 버그 파일 편집

- A·B 방식: Colab 왼쪽 📁 → `sonic-vibe-coding-lab/exercises/buggy_pipeline.py` 더블클릭 → 편집 후 저장. 런타임이 끊기면 수정이 사라지므로 수시로 노트북 메모 셀에 복사해 두세요.
- C 방식: 에이전트가 Fork의 `exercises/buggy_pipeline.py`를 고쳐 push → 자동 리포트 확인 → 리포트의 Colab 링크(또는 실행 중이면 `🔄 최신 코드 받기`).

## 4. 규칙

- `sonic_lab/reference.py`, `sonic_lab/check.py`에는 정답이 있습니다. 본인도 에이전트도 열지 않습니다.
- **🔮 예측은 실행 전에, 🧾 검증 기록과 📮 제출 카드는 본인이 직접** 씁니다. 이것이 평가 대상입니다.
- Day 2의 정답 비교(`check.bughunt`, `check.invariant_tests`)는 Mission 2-3에서만 부릅니다. 호출 횟수가 기록됩니다.
- 빨리 끝났다면 각 노트북 끝의 🚀 심화 미션으로 넘어갑니다.

## 부록. 개인 Linux/macOS 노트북에서 로컬 실행 (선택)

```bash
conda env create -f environment.yml && conda activate sonic-lab
jupyter lab          # notebooks/ 실행. CPU만 사용
```
메쉬 영상은 `libosmesa6`(Linux) 등 OpenGL이 필요하며, 없으면 `skeleton_video()`를 씁니다.
