"""MuJoCo 오프스크린 렌더러 (별도 프로세스에서 실행). 08번 노트북 렌더러를 정리한 것입니다.

저장된 qpos를 `mj_forward`로 복원해 그리기만 합니다. 정책·물리를 다시 실행하지 않습니다.
python -m sonic_lab.render_worker --scene ... --npz ... --out ... [--fps 25 --seconds 6]
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    model.vis.quality.offsamples = 0
    model.vis.quality.shadowsize = 512
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.height)
    state = mujoco.MjData(model)
    with np.load(a.npz, allow_pickle=False) as z:
        qpos, times = z["qpos"], z["times_s"]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos shape {qpos.shape} does not match model nq={model.nq}")
    duration = len(qpos) / 50 if a.seconds is None else min(a.seconds, len(qpos) / 50)
    count = max(1, int(np.ceil(duration * a.fps - 1e-9)))
    idx = np.minimum(np.floor(np.arange(count) * 50 / a.fps).astype(int), len(qpos) - 1)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -15
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, width=a.width, height=a.height) as renderer, \
            imageio.get_writer(a.out, fps=a.fps, codec="libx264", macro_block_size=2,
                               ffmpeg_params=["-crf", "23", "-preset", "veryfast"]) as writer:
        for n, i in enumerate(idx):
            state.qpos[:] = qpos[i]
            mujoco.mj_forward(model, state)
            camera.lookat[:] = state.qpos[:3] + [0, 0, .05]
            renderer.update_scene(state, camera=camera, scene_option=opt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, a.width, 28), fill=(13, 24, 38))
            draw.text((10, 5), f"{a.label}  t={times[i]:.2f}s  (MuJoCo recorded physics)", font=font, fill="white")
            writer.append_data(np.asarray(frame))
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(idx)} frames", flush=True)
    print(f"saved {a.out} ({len(idx)} frames)", flush=True)


if __name__ == "__main__":
    main()
