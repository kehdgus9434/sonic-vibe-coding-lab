"""학생 작업 자동 리포트 — 웹 코딩 에이전트가 push하면 GitHub Actions(student-check)가 실행합니다.

    python tools/student_report.py --out outputs/report [--days 1 2 3] [--repo OWNER/REPO --branch BRANCH]

- Day 1: my_work/day1.py의 미션 함수를 체크(✅/❌/⬜ 미작성)
- Day 2: exercises/buggy_pipeline.py에 **학생 자신의** 불변식 테스트(my_work/day2.py)와 폐루프 실행만 적용.
         정답 비교(check.bughunt / invariant_tests)는 하지 않습니다 — Colab Mission 2-3에서만.
- Day 3: my_work/day3.py의 experiment(lab) 결과 표·그래프·영상
- Colab에서 이 브랜치를 바로 여는 노트북(REPO/BRANCH가 채워지고 my_work를 불러오게 설정)을 만듭니다.
결과: <out>/report.md, 그림(png/gif), 영상(mp4), colab/*.ipynb
"""
import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = "https://github.com/kehdgus9434/sonic-vibe-coding-lab"


def fresh_import(name):
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    return importlib.import_module(name)


def run_step(fn):
    """(상태, 출력) — 상태: ok / fail / todo / error"""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = fn()
        status = "ok" if result is not False else "fail"
    except NotImplementedError:
        status = "todo"
    except AssertionError as error:
        buffer.write(f"AssertionError: {error}\n")
        status = "fail"
    except Exception as error:  # noqa: BLE001
        buffer.write("".join(traceback.format_exception_only(type(error), error)))
        status = "error"
    return status, buffer.getvalue().strip()


ICON = {"ok": "✅", "fail": "❌", "todo": "⬜ 미작성", "error": "💥 오류"}


def details(text, limit=1500):
    if not text:
        return ""
    text = text if len(text) < limit else text[:limit] + "\n…"
    return f"<details><summary>출력</summary>\n\n```\n{text}\n```\n</details>"


def to_gif(mp4, gif, width=360, fps=10):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-loglevel", "error", "-y", "-i", str(mp4), "-vf", f"fps={fps},scale={width}:-1:flags=lanczos", str(gif)],
                   check=True)
    return gif


def video(lab, arrays, out, label):
    from sonic_lab import render
    mp4 = render.mesh_video(lab, arrays, out.with_suffix(".mp4"), label=label, seconds=6, width=480, height=360)
    if mp4 is None:
        mp4 = render.skeleton_video(lab, arrays, out.with_suffix(".mp4"), title=label, seconds=6)
    return to_gif(mp4, out.with_suffix(".gif"))


def run_table(results):
    keys = ["simulated_frames", "requested_frames", "completed", "stop_reason", "min_root_z_m", "joint_rmse_deg"]
    rows = ["| 조건 | 프레임 | 완주 | 중단 이유 | 최저 골반 높이(m) | 관절 RMSE(°) |", "|---|---|---|---|---|---|"]
    for label, (s, _) in results.items():
        v = {k: s.get(k) for k in keys}
        rmse = "" if v["joint_rmse_deg"] is None else f'{v["joint_rmse_deg"]:.2f}'
        z = "" if v["min_root_z_m"] is None else f'{v["min_root_z_m"]:.3f}'
        rows.append(f'| {label} | {v["simulated_frames"]}/{v["requested_frames"]} | {"✅" if v["completed"] else "❌"} | '
                    f'{v["stop_reason"]} | {z} | {rmse} |')
    return "\n".join(rows)


# ---------------------------------------------------------------------------

def day1(lab, out):
    import numpy as np  # noqa: F401
    from sonic_lab import check
    from sonic_lab.assets import SAMPLE
    from sonic_lab.reference import load_published_pickle
    m = fresh_import("my_work.day1")
    robot = load_published_pickle(lab.assets.sample_root / "robot_filtered/210531" / SAMPLE)
    human = load_published_pickle(lab.assets.sample_root / "smpl_filtered" / SAMPLE)
    missions = [
        ("1-1 ONNX 입출력", lambda: check.onnx_io(*m.inspect_onnx(lab.assets.model_dir))),
        ("1-2 Encoder 블록 표", lambda: check.encoder_slices(lab, m.build_encoder_slices(lab.assets.model_dir / "observation_config.yaml"))),
        ("1-3 쿼터니언 순서", lambda: check.root_quat_order(m.detect_root_quat_order(robot))),
        ("1-4 50 Hz 정렬", lambda: check.resampled_motion(lab, *m.resample_g1(robot, human))),
        ("1-5 Mode-2 Encoder 입력", lambda: check.encoder_obs(lab, m.build_encoder_obs_mode2)),
        ("🚀 A. Mode-0 Encoder 입력", lambda: check.encoder_obs_mode0(lab, m.build_encoder_obs_mode0)),
    ]
    lines = ["## Day 1 — `my_work/day1.py`", "", "| 미션 | 결과 |", "|---|---|"]
    notes = []
    for title, fn in missions:
        status, text = run_step(fn)
        lines.append(f"| {title} | {ICON[status]} |")
        if status in ("fail", "error"):
            notes.append(f"**{title}**\n{details(text)}")
    return "\n".join(lines + [""] + notes), {}


def day2(lab, out):
    from sonic_lab import reference as ref
    spec = importlib.util.spec_from_file_location("buggy_pipeline", ROOT / "exercises/buggy_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipeline = module.Pipeline()
    m = fresh_import("my_work.day2")
    lines = ["## Day 2 — `exercises/buggy_pipeline.py` + `my_work/day2.py`", "",
             "> 정답 비교는 하지 않습니다. 아래는 **내 불변식 테스트**와 **물리 실행 결과**뿐입니다. 최종 판정은 Colab Mission 2-3에서.", ""]
    status, text = run_step(lambda: m.invariant_tests(pipeline, lab))
    lab.sim.reset(lab.motion)
    label = {"ok": "✅ 모두 통과", "fail": "❌ 실패", "todo": "⬜ 미작성", "error": "💥 오류"}[status]
    lines.append(f"- 내 불변식 테스트: {label}")
    if text:
        lines.append(details(text))
    buglog = re.search(r"BUGLOG.*?\n((?:\d+\..*\n?)+)", module.__doc__ or "")
    if buglog:
        entries = [e for e in buglog.group(1).strip().splitlines() if e.strip().rstrip(".").strip() not in "1234"]
        lines.append(f"- BUGLOG 기록: {len(entries)}개" + ("" if not entries else "\n" + "\n".join(f"  - {e.strip()}" for e in entries)))
    runs = {}
    status, text = run_step(lambda: runs.__setitem__("5초", ref.run_physics(lab.sim, lab.policy, lab.motion, 2, max_frames=250, pipeline=pipeline)))
    if status != "ok":
        lines.append(f"- 폐루프 실행 {ICON[status]}\n{details(text)}")
        return "\n".join(lines), {}
    if runs["5초"][0]["completed"]:
        runs["40초 전체"] = ref.run_physics(lab.sim, lab.policy, lab.motion, 2, pipeline=pipeline)
    lines += ["", run_table(runs)]
    last = list(runs.values())[-1]
    gif = video(lab, last[1], out / "day2_pipeline", "my pipeline / mode 2")
    return "\n".join(lines), {"day2_pipeline.gif": gif}


def day3(lab, out):
    from sonic_lab import analysis
    m = fresh_import("my_work.day3")
    lines = ["## Day 3 — `my_work/day3.py`", ""]
    holder = {}
    status, text = run_step(lambda: holder.__setitem__("results", m.experiment(lab)))
    if status != "ok":
        lines.append(f"- experiment(): {ICON[status]}\n{details(text)}")
        return "\n".join(lines), {}
    results = holder["results"]
    lines += [run_table(results), ""]
    for label, (_, arrays) in results.items():
        st, txt = run_step(lambda: print(m.joint_error_by_group(arrays, lab)))
        if st == "ok":
            lines.append(f"- 부위별 관절 오차 · {label}: `{txt}`")
        elif st != "todo":
            lines.append(f"- 부위별 관절 오차 · {label}: {ICON[st]}\n{details(txt)}")
    fig = analysis.plot(results, lab)
    png = out / "day3_plot.png"
    fig.savefig(png, dpi=110)
    label = list(results)[-1]
    gif = video(lab, results[label][1], out / "day3_last_condition", label)
    return "\n".join(lines), {"day3_plot.png": png, "day3_last_condition.gif": gif}


# ---------------------------------------------------------------------------

def colab_notebooks(out, repo, branch):
    """REPO/BRANCH를 채우고 my_work import 주석을 해제한 노트북 사본."""
    made = []
    target = out / "colab"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "notebooks").glob("day*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            src = re.sub(r'REPO = "[^"]*"', f'REPO = "https://github.com/{repo}"', src)
            src = re.sub(r'BRANCH = "[^"]*"', f'BRANCH = "{branch}"', src)
            src = re.sub(r"^# (from my_work\.day\d import \*)", r"\1", src, flags=re.M)
            cell["source"] = src
        (target / path.name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        made.append(path.name)
    return made


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "outputs/report")
    p.add_argument("--days", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--repo", default="")
    p.add_argument("--branch", default="")
    p.add_argument("--assets-url", default="", help="그림·노트북이 올라갈 위치(raw URL 접두사). 없으면 상대 경로")
    p.add_argument("--colab-prefix", default="", help="Colab 링크 접두사 (예: https://colab.research.google.com/github/OWNER/REPO/blob/reports/BRANCH)")
    p.add_argument("--cache-key", default="", help="이미지 URL 캐시 무효화용 값(커밋 SHA 등)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import sonic_lab
    with contextlib.redirect_stdout(io.StringIO()):
        lab = sonic_lab.start(quiet=True)
    begin = time.time()
    sections, files = [], {}
    for day, fn in [(1, day1), (2, day2), (3, day3)]:
        if day not in a.days:
            continue
        print(f"== Day {day}", flush=True)
        status, text = run_step(lambda: sections.append(fn(lab, a.out)))
        if status != "ok":
            sections.append((f"## Day {day}\n{ICON[status]}\n{details(text)}", {}))
    body = ["# 🤖 SONIC 실습 자동 리포트"]
    if a.repo and a.branch:
        body.append(f"`{a.repo}` · 브랜치 `{a.branch}` · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    notebooks = colab_notebooks(a.out, a.repo or "kehdgus9434/sonic-vibe-coding-lab", a.branch or "main")
    if a.colab_prefix:
        links = " · ".join(f"[{n.split('_')[0].upper()}]({a.colab_prefix}/colab/{n})" for n in notebooks)
        body += ["", f"**▶ Colab에서 내 브랜치로 열기**: {links}",
                 "> 열면 0번 셀의 REPO/BRANCH가 이 브랜치로 채워져 있고 `my_work` 코드를 불러옵니다. "
                 "🔮 예측·🧾 검증 기록·📮 제출 카드는 거기서 **직접** 작성하세요."]
    body += ["", "> 이 리포트는 코드가 실행되는지와 체크 결과를 보여 줄 뿐입니다. ✅가 '맞는 이유'로 나왔는지 판단하는 것은 당신의 몫입니다."]
    for text, made in sections:
        body += ["", text]
        for name, path in made.items():
            files[name] = path
            url = f"{a.assets_url}/{name}" if a.assets_url else name
            if a.cache_key:
                url += f"?v={a.cache_key}"
            body.append(f"\n![{name}]({url})")
    body += ["", f"<sub>실행 시간 {time.time() - begin:.0f}s · 영상(mp4)은 Actions 실행 페이지의 Artifacts에 있습니다.</sub>"]
    (a.out / "report.md").write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))


if __name__ == "__main__":
    main()
