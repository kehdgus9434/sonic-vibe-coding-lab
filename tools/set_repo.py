"""GitHub에 올린 뒤 한 번 실행: 노트북·README의 저장소 주소를 내 저장소로 바꿉니다.

    python tools/set_repo.py <GitHub ID 또는 조직>/<저장소 이름>
    예) python tools/set_repo.py hlab-kr/sonic-vibe-coding-lab
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "YOUR_GITHUB_ID/sonic-vibe-coding-lab"

slug = sys.argv[1] if len(sys.argv) > 1 else ""
if not re.fullmatch(r"[\w.-]+/[\w.-]+", slug):
    sys.exit(__doc__)
for path in [ROOT / "README.md", ROOT / "tools/build_notebooks.py", *sorted((ROOT / "notebooks").glob("*.ipynb"))]:
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        path.write_text(text.replace(PLACEHOLDER, slug), encoding="utf-8")
        print("updated", path.relative_to(ROOT))
