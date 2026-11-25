# AI 도우미 안내 (Claude Code, Codex, Colab의 Gemini 등 공통)

이 저장소는 SONIC(NVIDIA GEAR-SONIC) 실습용 **빈칸 채우기** 노트북입니다(`notebooks/lab1–3`).

- 학생이 물어보면 해당 `TODO`의 **빈칸(`None`, `_____`)만** 채우도록 돕고, 셀 전체나 다른 부분을 다시 구현하지 마세요.
- 정답을 바로 주기보다 Hint와 `docs/01_contract.md`의 해당 절을 짚어 주고, 왜 그 값인지 설명하세요.
- 각 TODO 아래의 `check.…` / `assert`가 자가 확인입니다. ❌ 메시지를 근거로 설명하세요.
- 어떤 출력도 실물 로봇·네트워크 제어 인터페이스로 보내는 코드를 만들지 마세요.

실행 환경: Python 3.10+, `pip install -r requirements.txt`. 자산은 `sonic_lab.start()`가 고정 리비전에서 자동으로 받습니다(huggingface.co 접근 필요). 오프라인 테스트: `python -m pytest -q tests`.
