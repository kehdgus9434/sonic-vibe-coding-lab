"""실제 NVIDIA 모델 점검 (관리자·CI용). GitHub Actions의 real-model-check 워크플로가 실행합니다.

    python tools/real_model_check.py --out outputs/real_check

확인 항목
1. 기준 3조건(reference_pd / 모드 0 / 모드 2) 전체 길이 실행과 소요 시간
2. Lab 2 '흔한 실수' 4가지가 각각 어떤 결과를 내는지
3. 메쉬 렌더링(OSMesa/EGL) · 골격 영상
4. 학생용 노트북 3개를 빈칸 상태로 끝까지 실행 — 빈칸 때문에 생기는 오류(AssertionError, NameError) 외의 오류가 없어야 함
   (빈칸을 모범답안으로 채운 실행은 강사용 비공개 저장소의 run-answers가 담당)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_STUDENT_ERRORS = {"AssertionError", "NameError"}


def brief(summary):
    keys = ["simulated_frames", "requested_frames", "completed", "stop_reason", "runtime_s", "min_root_z_m", "joint_rmse_deg"]
    return {k: summary.get(k) for k in keys}


def run_notebooks(out):
    import nbformat
    from nbclient import NotebookClient
    report = {}
    for path in sorted((ROOT / "notebooks").glob("lab*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        begin = time.time()
        NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=True,
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
        errors = [{"cell": i, "ename": o["ename"], "evalue": o["evalue"][:160]}
                  for i, cell in enumerate(nb.cells) for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        unexpected = [e for e in errors if e["ename"] not in EXPECTED_STUDENT_ERRORS]
        report[path.name] = {"seconds": round(time.time() - begin), "blank_errors": len(errors) - len(unexpected),
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
    from scipy.spatial.transform import Rotation as R
    import sonic_lab
    from sonic_lab import reference as ref, render

    t0 = time.time()
    lab = sonic_lab.start()
    result = {"start_s": round(time.time() - t0)}

    print("== 1. 기준 3조건 (전체 길이)", flush=True)
    base = {str(m): ref.run_physics(lab.sim, lab.policy, lab.motion, m) for m in ["reference_pd", 0, 2]}
    result["baseline"] = {k: brief(v[0]) for k, v in base.items()}

    print("== 2. 흔한 실수 4가지 (500프레임)", flush=True)
    c = lab.contract

    class Base:
        def observation(self, sim, prev): return sim.observation(prev)
        def decoder_obs(self, token, history): return ref.Policy.decoder_obs(token, history)
        def action_to_target(self, action, contract): return ref.action_to_target(action, contract)

    class NoOffset(Base):
        def observation(self, sim, prev):
            return dict(sim.observation(prev), q=sim.data.qpos[sim.qaddr][c.mujoco_to_isaaclab])

    class QuatOrder(Base):
        def observation(self, sim, prev):
            return dict(sim.observation(prev), gravity=R.from_quat(sim.data.qpos[3:7]).inv().apply([0, 0, -1]))

    class History(Base):
        def decoder_obs(self, token, history): return ref.Policy.decoder_obs(token, list(reversed(list(history))))

    class Remap(Base):
        def action_to_target(self, action, contract):
            return contract.default + np.asarray(action)[contract.mujoco_to_isaaclab] * contract.scale

    mistakes = {cls.__name__: ref.run_physics(lab.sim, lab.policy, lab.motion, 2, max_frames=500, pipeline=cls())[0]
                for cls in [Base, NoOffset, QuatOrder, History, Remap]}
    result["mistakes"] = {k: brief(v) for k, v in mistakes.items()}

    print("== 3. 렌더링", flush=True)
    t0 = time.time()
    video = render.mesh_video(lab, base["2"][1], a.out / "mode2_mesh.mp4", label="real model / mode 2", seconds=6)
    result["mesh_video"] = {"ok": video is not None, "seconds": round(time.time() - t0)}
    render.skeleton_video(lab, base["2"][1], a.out / "mode2_skeleton.mp4", title="mode 2", seconds=6)

    if not a.skip_notebooks:
        print("== 4. 학생용 노트북 (빈칸 상태)", flush=True)
        result["notebooks"] = run_notebooks(a.out)

    failures = []
    for mode in ["0", "2"]:
        if not result["baseline"][mode]["completed"]:
            failures.append(f"공식 샘플 모드 {mode}가 완주하지 못함")
    if not result["mistakes"]["Base"]["completed"]:
        failures.append("올바른 파이프라인이 500프레임을 완주하지 못함")
    if not result["mesh_video"]["ok"]:
        failures.append("메쉬 렌더링 실패")
    for name, nb in result.get("notebooks", {}).items():
        if nb["unexpected_errors"]:
            failures.append(f"{name}: 빈칸과 무관한 오류 {nb['unexpected_errors']}")
    result["failures"] = failures
    (a.out / "real_model_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    lines = ["# 실제 모델 점검 결과", "", "| 항목 | 프레임 | 완주 | 중단 이유 | 시간(s) | 관절 RMSE(°) |", "|---|---|---|---|---|---|"]
    for group in ["baseline", "mistakes"]:
        for k, v in result[group].items():
            rmse = v.get("joint_rmse_deg")
            lines.append(f"| {group}/{k} | {v['simulated_frames']}/{v['requested_frames']} | {v['completed']} | {v['stop_reason']} | "
                         f"{v['runtime_s']} | {'' if rmse is None else round(rmse, 2)} |")
    lines += ["", f"- start(): {result['start_s']} s · mesh video: {result['mesh_video']}"]
    for name, nb in result.get("notebooks", {}).items():
        lines.append(f"- {name}: {nb['seconds']} s, 빈칸 오류 {nb['blank_errors']}개, 예상 외 오류 {len(nb['unexpected_errors'])}개")
    lines += ["", "**실패 항목**: " + ("없음" if not failures else "; ".join(failures))]
    text = "\n".join(lines)
    (a.out / "summary.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(text)
    print(text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
