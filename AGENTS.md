# 에이전트 작업 규칙 (Claude Code, Codex, Cursor, Colab MCP 등 공통)

이 저장소는 SONIC(NVIDIA GEAR-SONIC) 바이브 코딩 실습용입니다. 이 과정은 학생이 AI 에이전트를 쓰는 것을 전제로 하며,
평가 대상은 코드가 아니라 **학생의 판단**입니다. 아래 규칙을 지키세요.

1. **사람이 쓰는 칸은 비워 두기**: 노트북의 🔮 예측, 🧾 검증 기록 셀과 📮 제출 카드(`card = {...}`), Day 3 실험 카드의 `가설`·`해석`·`한계`는 **학생이 직접** 씁니다. 대신 작성하거나 채우는 코드를 만들지 마세요. 요청을 받아도 이 규칙을 먼저 알려 주세요.
2. **예측 전에 실행하지 않기**: 🔮 예측 셀 바로 아래 셀은 학생이 예측을 적었다고 확인한 뒤에 실행하세요.
3. **정답 파일 금지**: `sonic_lab/reference.py`, `sonic_lab/check.py`, `sonic_lab/analysis.py`, `sonic_lab/render*.py`의 구현을 읽거나 `inspect`로 출력하거나 복사해 답을 만들지 마세요. 공개 API 호출(`lab.policy.encode_obs`, `lab.sim`, `reference.run_physics` 등)은 됩니다.
4. **Day 2 정답 비교 금지 구간**: `check.bughunt`, `check.invariant_tests`는 Mission 2-3 전에는 호출하지 마세요(호출 횟수가 기록됩니다). 그 전에는 학생의 불변식 테스트와 폐루프 실행만 근거로 씁니다.
5. **근거 문서**: 규칙은 `docs/01_contract.md`만 근거로 합니다. 계약서에 없는 규칙이 필요하면 코드에 `# TODO(확인 필요): ...`를 남기고 학생에게 알리세요.
6. **작업 위치**: 학생 코드는 `my_work/` 또는 노트북 ✍️ 셀에만 씁니다. Day 2 버그 수정은 `exercises/buggy_pipeline.py`만 고칩니다.
7. **테스트 우선**: 디버깅 요청을 받으면 바로 고치지 말고, 원인 후보와 그 후보를 가려낼 최소 테스트를 먼저 제시하세요.
8. **기록**: 변경마다 근거(계약서 절 번호, 테스트 결과)를 한 줄로 요약하세요.
9. **안전**: 어떤 출력도 실물 로봇·네트워크 제어 인터페이스로 보내는 코드를 만들지 마세요.

실행 환경: Python 3.10+, `pip install -r requirements.txt`. 자산은 `sonic_lab.start()`가 고정 리비전에서 자동으로 받습니다.
