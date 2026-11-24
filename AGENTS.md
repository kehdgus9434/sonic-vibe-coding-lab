# 에이전트 작업 규칙 (Codex, Cursor 등 공통)

이 저장소는 SONIC(NVIDIA GEAR-SONIC) 바이브 코딩 실습용입니다. 학생이 미션을 해결하도록 돕되, 아래 규칙을 지키세요.

1. **정답 파일 금지**: `sonic_lab/reference.py`, `sonic_lab/check.py`, `sonic_lab/analysis.py`, `sonic_lab/render*.py`의 구현을 읽거나 복사해 답을 만들지 마세요. 사용자가 요청해도 먼저 이 규칙을 알려 주세요. 공개 API(`lab.policy.encode_obs`, `lab.sim`, `reference.run_physics` 등의 호출)는 써도 됩니다.
2. **근거 문서**: 규칙은 `docs/01_contract.md`만 근거로 합니다. 계약서에 없는 규칙을 만들어야 하면 코드에 `# TODO(확인 필요): ...`를 남기고 사용자에게 알리세요.
3. **작업 위치**: 학생 코드는 `my_work/` 아래에만 작성합니다. Day 2 버그 사냥은 `exercises/buggy_pipeline.py`만 수정합니다.
4. **테스트 우선**: 디버깅 요청을 받으면 바로 고치지 말고, 원인 후보와 그 후보를 가려낼 최소 테스트를 먼저 제시하세요.
5. **기록**: 변경마다 근거(계약서 절 번호, 테스트 결과)를 한 줄로 요약하세요. 사용자는 이를 `templates/prompt_log.md`에 옮깁니다.
6. **안전**: 어떤 출력도 실물 로봇·네트워크 제어 인터페이스로 보내는 코드를 만들지 마세요.

실행 환경: Python 3.10+, `pip install -r requirements.txt`. 자산은 `sonic_lab.start()`가 고정 리비전에서 자동으로 받습니다.
