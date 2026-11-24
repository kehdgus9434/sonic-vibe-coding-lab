"""실제 NVIDIA 모델 점검 (관리자·CI용). GitHub Actions의 real-model-check 워크플로가 실행합니다.

    python tools/real_model_check.py --out outputs/real_check

확인 항목
1. 기준 3조건(reference_pd / 모드 0 / 모드 2) 전체 길이 실행과 소요 시간
2. Day 2 버그 파이프라인 + 버그 1개씩 넣은 변형이 실제로 넘어지는지, 올바른 파이프라인은 완주하는지
3. Day 3 카드 예시(좌우 반전, 밀기, 손목 0)
4. 메쉬 렌더링(OSMesa/EGL) · 골격 영상
5. 노트북 3개를 처음부터 끝까지 실행 — 학생이 채울 ✍️ 셀의 NotImplementedError 외의 오류가 없어야 함
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location("buggy_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Pipeline()


def brief(summary):
    keys = ["simulated_frames", "requested_frames", "completed", "stop_reason", "runtime_s",
            "min_root_z_m", "joint_rmse_deg", "mean_abs_joint_error_deg"]
    return {k: summary.get(k) for k in keys}


def run_notebooks(out):
    import nbformat
    from nbclient import NotebookClient
    report = {}
    for path in sorted((ROOT / "notebooks").glob("day*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        begin = time.time()
        NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=True,
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
        errors = []
        for i, cell in enumerate(nb.cells):
            for o in cell.get("outputs", []):
                if o.get("output_type") == "error":
                    errors.append({"cell": i, "ename": o["ename"], "evalue": o["evalue"][:200]})
        unexpected = [e for e in errors if e["ename"] != "NotImplementedError"]
        report[path.name] = {"seconds": round(time.time() - begin), "expected_stub_errors": len(errors) - len(unexpected),
                             "unexpected_errors": unexpected}
        nbformat.write(nb, out / f"executed_{path.name}")
        print(f"notebook {path.name}: {report[path.name]}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "outputs/real_check")
    p.add_argument("--skip-notebooks", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    import numpy as np
    import dataclasses
    import sonic_lab
    from sonic_lab import check, reference as ref, render

    t0 = time.time()
    lab = sonic_lab.start()
    result = {"start_s": round(time.time() - t0), "official": lab.motion.metadata}

    print("== 1. 기준 3조건 (전체 길이)", flush=True)
    base = {}
    for mode in ["reference_pd", 0, 2]:
        s, arr = ref.run_physics(lab.sim, lab.policy, lab.motion, mode)
        base[str(mode)] = (s, arr)
        print(mode, brief(s), flush=True)
    result["baseline"] = {k: brief(v[0]) for k, v in base.items()}

    print("== 2. Day 2 버그 사냥", flush=True)
    bug = {"buggy_all4": ref.run_physics(lab.sim, lab.policy, lab.motion, 2, max_frames=500,
                                         pipeline=load_pipeline(ROOT / "exercises/buggy_pipeline.py"))[0]}
    for i, mutant in enumerate([check._M1, check._M2, check._M3, check._M4], 1):
        bug[f"only_bug{i}"] = ref.run_physics(lab.sim, lab.policy, lab.motion, 2, max_frames=500, pipeline=mutant())[0]
    bug["correct_pipeline_full"] = ref.run_physics(lab.sim, lab.policy, lab.motion, 2, pipeline=check._Correct())[0]
    for k, v in bug.items():
        print(k, brief(v), flush=True)
    result["bughunt"] = {k: brief(v) for k, v in bug.items()}
    result["bughunt_check_on_buggy"] = brief(check.bughunt(lab, load_pipeline(ROOT / "exercises/buggy_pipeline.py")))

    print("== 3. Day 3 카드 예시", flush=True)

    def push(force, start=250, duration=10):
        def on_frame(frame, sim):
            sim.data.xfrc_applied[sim.pelvis_id, :3] = [force, 0, 0] if start <= frame < start + duration else 0
        return on_frame

    d3 = {"mirrored": ref.run_physics(lab.sim, lab.policy, lab.load_motion(mirrored=True), 2, max_frames=1000)[0]}
    for force in [100, 200, 400]:
        d3[f"push_{force}N"] = ref.run_physics(lab.sim, lab.policy, lab.motion, 2, max_frames=1000, on_frame=push(force))[0]
    no_wrist = dataclasses.replace(lab.motion, wrists_il=np.zeros_like(lab.motion.wrists_il))
    d3["wrists_zero"] = ref.run_physics(lab.sim, lab.policy, no_wrist, 2, max_frames=1000)[0]
    for k, v in d3.items():
        print(k, brief(v), flush=True)
    result["day3_cards"] = {k: brief(v) for k, v in d3.items()}

    print("== 4. 렌더링", flush=True)
    t0 = time.time()
    video = render.mesh_video(lab, base["2"][1], a.out / "mode2_mesh.mp4", label="real model / mode 2", seconds=6)
    result["mesh_video"] = {"ok": video is not None, "seconds": round(time.time() - t0)}
    t0 = time.time()
    render.skeleton_video(lab, base["2"][1], a.out / "mode2_skeleton.mp4", title="mode 2", seconds=6)
    result["skeleton_video"] = {"seconds": round(time.time() - t0)}

    if not a.skip_notebooks:
        print("== 5. 노트북 실행", flush=True)
        result["notebooks"] = run_notebooks(a.out)

    failures = []
    for mode in ["0", "2"]:
        if not result["baseline"][mode]["completed"]:
            failures.append(f"공식 샘플 모드 {mode}가 완주하지 못함")
    if not result["bughunt"]["correct_pipeline_full"]["completed"]:
        failures.append("올바른 파이프라인이 완주하지 못함")
    if result["bughunt"]["buggy_all4"]["completed"]:
        failures.append("버그 파이프라인이 넘어지지 않음 (Day 2 증상이 재현되지 않음)")
    for name, nb in result.get("notebooks", {}).items():
        if nb["unexpected_errors"]:
            failures.append(f"{name}: 예상하지 못한 오류 {nb['unexpected_errors']}")
    result["failures"] = failures
    (a.out / "real_model_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    lines = ["# 실제 모델 점검 결과", "", "| 항목 | 프레임 | 완주 | 중단 이유 | 시간(s) | 관절 RMSE(°) |", "|---|---|---|---|---|---|"]
    for group in ["baseline", "bughunt", "day3_cards"]:
        for k, v in result[group].items():
            rmse = v.get("joint_rmse_deg")
            lines.append(f"| {group}/{k} | {v['simulated_frames']}/{v['requested_frames']} | {v['completed']} | {v['stop_reason']} | "
                         f"{v['runtime_s']} | {'' if rmse is None else round(rmse, 2)} |")
    lines += ["", f"- start(): {result['start_s']} s · mesh video: {result['mesh_video']} · skeleton: {result['skeleton_video']}"]
    for name, nb in result.get("notebooks", {}).items():
        lines.append(f"- {name}: {nb['seconds']} s, 학생 미작성 셀 오류 {nb['expected_stub_errors']}개, 예상 외 오류 {len(nb['unexpected_errors'])}개")
    lines += ["", "**실패 항목**: " + ("없음" if not failures else "; ".join(failures))]
    text = "\n".join(lines)
    (a.out / "summary.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(text)
    print(text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
