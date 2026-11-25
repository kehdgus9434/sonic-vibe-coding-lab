# SONIC 실습 — 휴머노이드 전신 제어 모델을 Colab에서 해부하고 실험하기

[![real-model-check](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/real-model-check.yml/badge.svg)](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/real-model-check.yml) [![offline-tests](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/test.yml/badge.svg)](https://github.com/kehdgus9434/sonic-vibe-coding-lab/actions/workflows/test.yml)

NVIDIA **GEAR-SONIC** 공개 모델(사람/로봇 동작 → 64차원 토큰 → Unitree G1 29관절 제어)을 Google Colab에서 실행하며,
모델의 입출력 구조 → 폐루프 제어 → 실험까지 3개의 Lab(각 2시간)으로 따라갑니다.

- **설치 없음**: 브라우저와 Google 계정만 있으면 됩니다. CPU 런타임으로 충분합니다(공용 PC·개인 노트북 모두 가능).
- **빈칸 채우기 방식**: 대부분의 코드는 작성되어 있고, `TODO` 셀의 `########### Answer #############` 사이 빈칸(`None`, `_____`)만 채웁니다.
- **바로 확인**: 각 TODO 아래의 `check.…` / `assert`가 ✅/❌로 알려 줍니다. 별도 제출이나 채점은 없습니다.

## Lab 구성

| Lab | 노트북 | 내용 | TODO |
|---|---|---|---|
| 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehdgus9434/sonic-vibe-coding-lab/blob/main/notebooks/lab1_sonic_encoder.ipynb) `lab1_sonic_encoder` | SONIC 구조, ONNX 입출력, Encoder 입력 1762칸, 쿼터니언 순서, 30→50 Hz 정렬, 토큰 시각화 | 7 |
| 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehdgus9434/sonic-vibe-coding-lab/blob/main/notebooks/lab2_closed_loop.ipynb) `lab2_closed_loop` | 관측 · Decoder 입력 · 액션 변환으로 G1 40초 걷기, 흔한 실수 4가지의 영향 | 6 |
| 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehdgus9434/sonic-vibe-coding-lab/blob/main/notebooks/lab3_experiments.ipynb) `lab3_experiments` | 기준 3조건, 부위별 오차, 좌우 반전 · 골반 밀기 · 손목 0 · 재생 속도 실험 | 5 |

각 노트북 맨 위의 **실습 환경 준비** 셀을 실행하면 이 저장소와 공식 모델·로봇·걷기 샘플(약 125 MB)이 자동으로 준비됩니다.
Lab마다 끝에 **해석 질문**이 있습니다(제출용이 아니라 스스로 정리하기 위한 질문).

## 참고 문서

| 문서 | 내용 |
|---|---|
| [`docs/00_setup.md`](docs/00_setup.md) | Colab에서 시작하기, AI 도구 활용 팁, 공용 PC 주의 |
| [`docs/01_contract.md`](docs/01_contract.md) | 입출력 계약서 — 관절 순서표, 좌표계·쿼터니언 규칙, Encoder/Decoder 블록 구성 |

## 저장소 구조

```
notebooks/        Lab 1–3 노트북 (Colab)
docs/             시작하기 · 입출력 계약서
sonic_lab/        실습 지원 패키지 (공식 자산 다운로드, 참조 구현, 자가 확인, 렌더링)
tools/            실제 모델 점검 스크립트 (GitHub Actions)
tests/            오프라인 테스트
```

## 이전 버전

AI 에이전트로 전체를 구현하고 예측·검증 기록으로 평가하던 이전 구성은 [`v1-agent-judgment`](https://github.com/kehdgus9434/sonic-vibe-coding-lab/tree/v1-agent-judgment) 브랜치에 있습니다.

## 출처와 라이선스

- 모델·샘플: [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) (NVIDIA Open Model License) — 실행 시 고정 리비전 `6733128a`에서 내려받으며 이 저장소에 포함하지 않습니다.
- G1 모델·정책 파라미터: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) 고정 리비전 `4f5e118a`에서 실행 시 내려받습니다.
- `sonic_lab/reference.py`는 AI인력양성사업 교육자료 07번 노트북(공식 샘플 2,002프레임 완주 확인)의 구현을 교육용으로 정리한 것입니다.
- ⚠️ 출력은 시뮬레이션 전용입니다. **실물 로봇에 보내지 마세요.** 완주는 "넘어지지 않고 실행됨"이며 동작 품질 검증이 아닙니다.
