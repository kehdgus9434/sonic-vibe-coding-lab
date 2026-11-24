# SONIC 바이브 코딩 랩 — 휴머노이드 전신 제어 모델을 AI와 함께 해부·디버깅·실험하기

[![real-model-check](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/real-model-check.yml/badge.svg)](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/real-model-check.yml) [![offline-tests](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/test.yml/badge.svg)](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/test.yml)

NVIDIA **GEAR-SONIC** 공개 모델(사람/로봇 모션 → 64차원 토큰 → Unitree G1 29관절 제어)을 Google Colab에서 실행하며,
**AI 코딩 도구가 만든 연구 코드를 어떻게 믿을 수 있게 만드는지** 3일(2시간 × 3회) 동안 실습합니다.

- 대상: 코딩 경험이 있는 대학원생 · 이론 강의 최소화 · 과제 없음
- 환경: **브라우저만** — 공용 컴퓨터실(Windows)·개인 노트북 어디서나, 설치 없음
- AI 도구: **자유, 에이전트 사용 전제** — Colab의 Gemini, 웹 채팅(ChatGPT·Claude·Gemini), 웹 코딩 에이전트(Claude Code on the web, Codex)
- 평가: 코드가 아니라 **판단** — 실행 전 🔮 예측, 체크 후 🧾 검증 기록, 매 Day 📮 제출 카드 (본인 작성)
- 실행 환경: **Google Colab CPU 런타임**으로 충분 (GPU는 영상 렌더링만 빨라짐)

## 3일 구성

| | 노트북 | 핵심 질문 | 결과물 |
|---|---|---|---|
| **Day 1** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehdgus9434/sonic-vibe-coding-lab/blob/main/notebooks/day1_anatomy.ipynb) `day1_anatomy` | 모델과 데이터의 '계약'을 AI와 함께 정확히 읽을 수 있나? | 내가 만든 Encoder 입력 → 토큰 |
| **Day 2** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehdgus9434/sonic-vibe-coding-lab/blob/main/notebooks/day2_bughunt.ipynb) `day2_bughunt` | 정답 코드 없이 버그를 잡을 수 있나? (테스트 먼저 → 수정 → 마지막에 정답 비교) | 불변식 테스트 + 버그 4개 수정 + 40초 완주 |
| **Day 3** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehdgus9434/sonic-vibe-coding-lab/blob/main/notebooks/day3_experiment.ipynb) `day3_experiment` | 기준 조건 위에서 나만의 질문을 실험할 수 있나? | 실험 1개 + 결과 영상 |

각 노트북 맨 위 `0. 실습 환경 준비` 셀만 실행하면 저장소·패키지·공식 자산(약 125 MB)이 자동으로 준비됩니다. 빨리 끝난 사람을 위해 각 노트북 끝에 🚀 심화 미션(대부분 정답 없음)이 있습니다.

## 먼저 읽을 문서

| 문서 | 내용 |
|---|---|
| [`docs/00_setup.md`](docs/00_setup.md) | 브라우저만으로 시작하기: Colab, AI 도구 3가지 방식(Colab Gemini / 웹 채팅 / 웹 코딩 에이전트), 공용 PC 주의 |
| [`docs/01_contract.md`](docs/01_contract.md) | **입출력 계약서** — AI 코드가 맞는지 판단하는 유일한 기준. 바이브 코딩 도구에 통째로 넣으세요 |
| [`docs/02_vibe_coding_guide.md`](docs/02_vibe_coding_guide.md) | 연구 코드 바이브 코딩 원칙, 프롬프트 템플릿, 이 자료가 만들어진 실제 과정(01→08) |
| [`docs/03_checklist.md`](docs/03_checklist.md) | AI가 만든 연구 코드 검증 체크리스트 |
| [`docs/04_web_agent_workflow.md`](docs/04_web_agent_workflow.md) | 웹 코딩 에이전트(Claude Code on the web·Codex) → push → **GitHub이 실제 모델로 자동 실행해 PR 댓글로 결과(표·그래프·로봇 GIF)와 Colab 링크 제공** |

## 저장소 구조

```
notebooks/            Day 1–3 실습 노트북 (Colab)
docs/                 계약서 · 가이드 · 체크리스트
exercises/            Day 2 버그 사냥 대상 (buggy_pipeline.py)
my_work/              에이전트형 도구 사용자의 미션 작성 파일
templates/            프롬프트 로그
sonic_lab/            실습 지원 패키지 (자산 다운로드, 참조 구현, 체크포인트, 렌더링)
  ├─ reference.py     ⚠️ 정답 구현 — 실습 중에는 열지 않기
  └─ check.py         ⚠️ 채점 로직 — 실습 중에는 열지 않기
AGENTS.md, CLAUDE.md  에이전트형 도구가 자동으로 읽는 실습 규칙
tests/                관리자용 오프라인 테스트 (가짜 모델)
tools/                노트북 생성기, 학생 자동 리포트(student_report.py), 실제 모델 점검, 웹 에이전트 환경 준비(agent_setup.sh)
```

## 관리

저장소를 옮기거나 Fork해서 쓸 때는 `python tools/set_repo.py <ID>/<저장소>`로 Colab 배지·클론 주소를 한 번에 바꿉니다.
코드가 바뀌면 GitHub Actions의 `real-model-check`가 실제 NVIDIA 모델로 기준 3조건·버그 재현·Day 3 예시·렌더링·노트북 3개 실행을 자동 점검합니다(결과는 Actions 실행 페이지 Summary).

## 출처와 라이선스

- 모델·샘플: [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) (NVIDIA Open Model License) — 실행 시 고정 리비전 `6733128a`에서 내려받으며 이 저장소에 포함하지 않습니다.
- G1 모델·정책 파라미터: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) 고정 리비전 `4f5e118a`에서 실행 시 내려받습니다.
- `sonic_lab/reference.py`는 AI인력양성사업 교육자료 07번 노트북(Colab에서 공식 샘플 2,002프레임 완주 확인)의 구현을 교육용으로 정리한 것입니다.
- ⚠️ 이 저장소의 출력은 시뮬레이션 전용입니다. **실물 로봇에 보내지 마세요.** 완주는 "넘어지지 않고 실행됨"이며 동작 품질 검증이 아닙니다.
