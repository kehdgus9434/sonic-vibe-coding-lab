"""Day 3 실험 결과 정리 도우미."""

from __future__ import annotations

import numpy as np

COLUMNS = ["label", "simulated_frames", "requested_frames", "completed", "stop_reason",
           "min_root_z_m", "min_pelvis_up_cosine", "joint_rmse_deg", "mean_abs_joint_error_deg", "runtime_s"]


def table(results: dict):
    """results: {라벨: (summary, arrays)} → pandas DataFrame."""
    import pandas as pd
    rows = []
    for label, (summary, _) in results.items():
        rows.append({k: summary.get(k) for k in COLUMNS[1:]} | {"label": label})
    df = pd.DataFrame(rows)[COLUMNS]
    return df.round(3)


_FONT_READY = False


def korean_font():
    """그래프의 한글이 □로 깨지지 않게 한글 글꼴을 설정합니다(없으면 Colab/CI에서 fonts-nanum 설치 시도)."""
    global _FONT_READY
    if _FONT_READY:
        return
    import os
    import shutil
    import subprocess
    from matplotlib import font_manager, rcParams
    wanted = ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "Noto Sans CJK JP")

    def find():
        names = {f.name for f in font_manager.fontManager.ttflist}
        return next((w for w in wanted if w in names), None)

    name = find()
    if name is None and shutil.which("apt-get") and hasattr(os, "geteuid") and os.geteuid() == 0:
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-nanum"], capture_output=True)
        for path in font_manager.findSystemFonts():
            if "Nanum" in path:
                font_manager.fontManager.addfont(path)
        name = find()
    if name:
        rcParams["font.family"] = name
    rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def plot(results: dict, lab=None):
    """골반 높이와 (참조가 있으면) 관절 오차 시계열."""
    import matplotlib.pyplot as plt
    korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
    for label, (summary, arrays) in results.items():
        if not len(arrays["qpos"]):
            continue
        t = arrays["times_s"]
        axes[0].plot(t, arrays["qpos"][:, 2], label=label)
        if lab is not None and lab.motion.g1_q is not None and summary["name"] == lab.motion.name:
            ids = np.minimum(np.arange(len(t)) + 1, len(lab.motion.times) - 1)
            err = np.degrees(np.abs(arrays["qpos"][:, lab.sim.qaddr] - lab.motion.g1_q[ids]).mean(axis=1))
            axes[1].plot(t, err, label=label)
    axes[0].axhline(.35, color="#c2410c", ls="--", lw=1)
    axes[0].set(title="pelvis height (m) — 0.35 m 아래면 넘어짐 판정", xlabel="time (s)")
    axes[1].set(title="mean |joint - reference| (deg)", xlabel="time (s)")
    for ax in axes:
        ax.legend(fontsize=8)
    return fig
