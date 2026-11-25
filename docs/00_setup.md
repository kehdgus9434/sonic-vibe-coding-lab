# 시작하기 — 브라우저만으로

설치할 것은 없습니다. 브라우저(Chrome 권장)와 Google 계정만 있으면 됩니다. 공용 컴퓨터실(Windows)·개인 노트북 모두 같습니다.

## 1. 노트북 열기
1. README의 **Open in Colab** 배지 클릭 → Google 로그인
2. 상단 **"Drive에 사본 저장"** — 내가 채운 빈칸이 내 Drive에 남습니다.
3. 맨 위 **실습 환경 준비** 셀 실행 (처음 1–2분). 런타임 유형은 **CPU**로 충분합니다.
4. 위에서부터 차례로 실행하며 `TODO` 셀의 빈칸을 채웁니다.

런타임이 끊기면(오래 쉬거나 창을 닫으면) 실습 환경 준비 셀부터 다시 실행하세요. Drive 사본에 채운 내용은 남아 있습니다.

## 2. 빈칸 채우기 규칙
- `########### Answer #############` 사이의 `None` 또는 `_____`만 바꿉니다.
- 채우지 않고 실행하면 `AssertionError: TODO …` 또는 `NameError: name '_____' is not defined`가 납니다. 정상입니다.
- 채운 뒤 셀 아래의 ✅/❌를 확인합니다. ❌이면 함께 나오는 메시지(어느 블록이 어떻게 다른지)를 읽고 다시 채웁니다.
- 막히면 [`01_contract.md`](01_contract.md)의 해당 절(관절 순서표, 쿼터니언 규칙, 블록 구성)을 보세요.

## 3. AI 도구 활용 (선택)
ChatGPT·Claude·Gemini 웹이나 Colab의 Gemini(화면 아래 ✦ 버튼)에 **빈칸이 있는 셀 하나**와 계약서의 해당 절을 함께 주고 물어보면 됩니다.
예: "이 셀의 TODO 3 빈칸에 들어갈 인덱스가 왜 [1, 2, 3, 0]인지 설명해 줘."
정답만 받아 적기보다 **왜 그 값인지** 묻는 것이 해석 질문에 답하는 데 도움이 됩니다.

## 4. 공용 PC 주의
Chrome 게스트 창이나 시크릿 창을 쓰고, 수업이 끝나면 Google(과 AI 서비스)에서 로그아웃하세요.

## 부록. 로컬 실행 (선택, Linux/macOS)
```bash
conda env create -f environment.yml && conda activate sonic-lab
jupyter lab          # notebooks/ 실행. CPU만 사용
```
메쉬 영상은 OpenGL(Linux는 `libosmesa6`)이 필요하며, 없으면 골격 영상으로 대신 보여 줍니다.
