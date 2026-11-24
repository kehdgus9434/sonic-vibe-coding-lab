#!/bin/bash
# 클라우드 코딩 에이전트(Claude Code on the web, Codex cloud) 환경의 "Setup script"에 붙여 넣는 준비 스크립트.
# 패키지를 설치하고 공식 자산(약 125 MB)을 미리 받아 둡니다. 저장소 위치와 상관없이 동작합니다.
#   필요한 네트워크: pypi.org, github.com, raw/media.githubusercontent.com, huggingface.co(+ *.hf.co)
set -e
REPO_URL="${SONIC_LAB_REPO:-https://github.com/kehdgus9434/sonic-vibe-coding-lab}"
pip install -q "sonic-vibe-coding-lab @ git+${REPO_URL}"
python - <<'PY'
from sonic_lab import assets
a = assets.prepare()
print("SONIC lab assets ready:", a.root)
PY
