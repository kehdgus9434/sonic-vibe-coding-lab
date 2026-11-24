"""결과 시각화: MuJoCo 메쉬 영상(EGL/OSMesa) 또는 GL 없는 골격 영상."""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _has_gpu():
    if not shutil.which("nvidia-smi"):
        return False
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _ensure_osmesa():
    if ctypes.util.find_library("OSMesa") is None and Path("/content").exists() and shutil.which("apt-get"):
        print("libosmesa6 설치 중(Colab, 1회)...", flush=True)
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"], check=True)


def mesh_video(lab, arrays, out, label="", seconds=6.0, fps=25, width=640, height=480, backend="auto"):
    """run_physics()의 arrays(qpos)를 G1 메쉬 영상으로. GPU 런타임은 EGL, CPU 런타임은 OSMesa."""
    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "states.npz"
        np.savez(npz, qpos=arrays["qpos"], times_s=arrays["times_s"])
        engines = [backend] if backend != "auto" else (["egl", "osmesa"] if _has_gpu() else ["osmesa", "egl"])
        for engine in engines:
            if engine == "osmesa":
                _ensure_osmesa()
            env = dict(os.environ, MUJOCO_GL=engine, PYOPENGL_PLATFORM=engine)
            if engine == "osmesa":
                env.update(LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe")
            cmd = [sys.executable, "-u", "-m", "sonic_lab.render_worker", "--scene", str(lab.assets.scene_xml),
                   "--npz", str(npz), "--out", str(out), "--label", label, "--fps", str(fps),
                   "--width", str(width), "--height", str(height)]
            if seconds is not None:
                cmd += ["--seconds", str(seconds)]
            print(f"렌더링: {engine} ...", flush=True)
            r = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parents[1]),
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(r.stdout.strip().splitlines()[-1])
                return out
            print(f"  {engine} 실패:", (r.stderr or r.stdout).strip().splitlines()[-1:])
    print("메쉬 렌더링 실패 → skeleton_video()로 대체하세요.")
    return None


def skeleton_video(lab, arrays, out, title="", seconds=6.0, fps=25):
    """OpenGL 없이 matplotlib로 G1 링크 골격(순기구학)을 그립니다. 어디서나 동작."""
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import mujoco
    model = lab.sim.model
    state = mujoco.MjData(model)
    qpos = arrays["qpos"]
    n = min(len(qpos), int(seconds * 50)) if seconds else len(qpos)
    idx = np.arange(0, n, max(1, round(50 / fps)))
    visible = [i for i in range(1, model.nbody) if "hand_" not in (model.body(i).name or "")]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111, projection="3d")
    with imageio.get_writer(out, fps=fps, codec="libx264", macro_block_size=2) as writer:
        for i in idx:
            state.qpos[:] = qpos[i]
            mujoco.mj_forward(model, state)
            xyz = state.xpos.copy()
            xyz[:, :2] -= state.qpos[:2]
            ax.clear()
            ax.set(xlim=(-.8, .8), ylim=(-.8, .8), zlim=(0, 1.8))
            ax.view_init(elev=12, azim=-65)
            ax.set_box_aspect([1.6, 1.6, 1.8])
            for b in visible:
                p = model.body_parentid[b]
                if p > 0:
                    ax.plot(*xyz[[p, b]].T, color="#15803d", linewidth=2)
            ax.set_title(f"{title} t={(i + 1) / 50:.2f}s", fontsize=9)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    print(f"saved {out} ({len(idx)} frames)")
    return out


def show(path, width=480):
    """Colab/Jupyter에서 mp4 표시."""
    from IPython.display import HTML, display
    import base64
    data = base64.b64encode(Path(path).read_bytes()).decode()
    display(HTML(f'<video width="{width}" controls loop autoplay muted src="data:video/mp4;base64,{data}"></video>'))
