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


def plot(results: dict, lab=None):
    """골반 높이와 (참조가 있으면) 관절 오차 시계열."""
    import matplotlib.pyplot as plt
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
    axes[1].set(title="mean |joint − reference| (deg)", xlabel="time (s)")
    for ax in axes:
        ax.legend(fontsize=8)
    return fig
