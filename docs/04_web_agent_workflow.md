# 웹 코딩 에이전트 + GitHub 자동 리포트 (Claude Code on the web · Codex)

에이전트가 내 Fork에 코드를 push하면 GitHub이 **실제 NVIDIA 모델로 자동 실행**해서 결과(체크 표, 그래프, 로봇 영상)를 보여 줍니다.
나는 그 결과를 보고 판단하고, 링크 하나로 Colab을 열어 🔮/🧾/📮를 씁니다. 브라우저만 있으면 됩니다.

```
 ┌───────────────┐  push(브랜치 day1)  ┌──────────────────────────┐   PR 댓글·Summary
 │ 웹 코딩 에이전트 │ ─────────────────▶ │ GitHub Actions             │ ──────────────────▶ 나: 표·그래프·GIF 확인
 │ (클라우드)      │                    │ student-check (실제 모델)   │                       │
 └───────────────┘                    └──────────────────────────┘                       ▼
        ▲                                                       "Colab에서 내 브랜치로 열기" 링크
        └──────────── 다음 지시 (리포트 내용을 근거로) ◀────────── 🔮 예측 · 🧾 검증 기록 · 📮 제출 카드
```

## 1. 처음 한 번 (10분)

1. 이 저장소를 **Fork** (오른쪽 위 Fork 버튼).
2. 내 Fork의 **Actions 탭** → *"I understand my workflows, go ahead and enable them"* 클릭. (Fork는 기본으로 Actions가 꺼져 있습니다. 공개 저장소라 실행 시간은 무료입니다.)
3. 에이전트에 Fork 연결 — 쓰는 것 하나만
   - **Claude Code on the web** (`claude.ai/code`): 내 Fork 선택. 에이전트가 직접 실행까지 하게 하려면 환경(Environment)의 Network access를 **Custom**으로 바꾸고 `huggingface.co`, `*.huggingface.co`, `*.hf.co`, `media.githubusercontent.com`을 추가, Setup script에 `tools/agent_setup.sh` 내용을 붙여 넣습니다. (이 설정이 없어도 아래 자동 리포트는 GitHub에서 돌아갑니다.)
   - **Codex** (`chatgpt.com/codex`): 내 Fork로 환경을 만들고 Setup script에 `bash tools/agent_setup.sh`.

## 2. 매 미션 흐름

1. **에이전트에게 지시** (예시)
   ```
   AGENTS.md 규칙을 따른다. docs/01_contract.md만 근거로 my_work/day1.py의 Mission 1-1~1-3을 구현해.
   day1 브랜치에 커밋·push하고 main으로 PR을 열어. 🔮/🧾/📮 칸은 건드리지 마.
   ```
2. **자동 리포트 확인** (push 후 3–6분)
   - PR이 열려 있으면 **PR 댓글**로 옵니다. 없으면 Actions 탭 → `student-check` 실행 → **Summary**.
   - 내용: Day 1 미션별 ✅/❌/⬜, Day 2 *내* 불변식 테스트 결과와 폐루프 실행 표, Day 3 실험 표·그래프, 로봇 동작 GIF.
   - 원본 영상(mp4)은 실행 페이지의 **Artifacts**.
3. **판단하고 기록**: 리포트의 **"Colab에서 내 브랜치로 열기"** 링크 → REPO/BRANCH가 이미 채워져 있고 `my_work` 코드를 불러오는 노트북이 열립니다. 0번 셀 실행 후 🔮 예측 → 실행 → 🧾 검증 기록. ("Drive에 사본 저장"을 해야 기록이 남습니다.)
4. **다음 지시**: ❌의 출력 메시지를 근거로 에이전트에게 원인 후보와 확인 테스트를 먼저 요청 → push → 리포트 갱신(같은 PR 댓글이 업데이트됨).

## 3. Day별 작성 위치

| Day | 에이전트가 고칠 파일 | 리포트가 보여 주는 것 |
|---|---|---|
| 1 | `my_work/day1.py` | 미션 1-1~1-5, 🚀 A 체크 결과 |
| 2 | `my_work/day2.py`(불변식 테스트), `exercises/buggy_pipeline.py`(버그 수정·BUGLOG) | 내 테스트 통과 여부, BUGLOG, 5초·40초 폐루프 표, GIF — **정답 비교는 하지 않음** |
| 3 | `my_work/day3.py`의 `experiment()`, `joint_error_by_group()` | 조건별 표, 골반 높이·관절 오차 그래프, 마지막 조건 GIF, 부위별 오차 |

Day 2의 정답 비교(`check.bughunt`)는 리포트에 없습니다. Colab Mission 2-3에서 직접 실행합니다.

## 4. 잘 안 될 때

| 증상 | 확인 |
|---|---|
| Actions에 `student-check`가 안 보임 | Fork의 Actions 탭에서 활성화했는지 |
| 리포트가 PR 댓글로 안 옴 | PR이 *내 Fork 안*에서 `day1 → main`으로 열렸는지(원본 저장소로 연 PR에는 댓글을 달 수 없음). Summary에는 항상 있습니다 |
| 그림이 깨짐 | 1–2분 뒤 새로고침(raw 파일 캐시). Fork가 비공개면 그림이 보이지 않으니 공개로 두세요 |
| 리포트가 오래됨 | 같은 브랜치에 새로 push하면 이전 실행은 취소되고 새로 실행됩니다 |
