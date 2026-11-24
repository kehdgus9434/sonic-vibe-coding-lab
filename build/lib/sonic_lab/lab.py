"""실습 시작 도우미: 환경 변수, 자산 준비, 참조 객체 생성."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import sys


def _limit_threads(threads: int):
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, str(threads))
    os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass
class Lab:
    assets: object
    contract: object
    policy: object
    sim: object
    motion: object          # 공식 걷기 샘플 (50 Hz)

    def load_motion(self, mirrored: bool = False):
        from . import reference as ref
        from .assets import SAMPLE, SAMPLE_MIRRORED
        return ref.load_official(self.assets, self.contract, SAMPLE_MIRRORED if mirrored else SAMPLE)


def start(threads: int = 1, quiet: bool = False) -> Lab:
    """자산을 내려받고(처음 1회 약 1–2분) 참조 계약·정책·시뮬레이터·공식 샘플을 준비합니다."""
    _limit_threads(threads)
    from . import assets as assets_mod
    from . import reference as ref
    import numpy as np, onnxruntime, mujoco  # noqa: E401

    a = assets_mod.prepare(quiet=quiet)
    contract = ref.load_contract(a)
    policy = ref.Policy(a, contract)
    sim = ref.Simulator(a, contract)
    motion = ref.load_official(a, contract)
    if not quiet:
        print(f"Python {sys.version.split()[0]} | {platform.platform()} | numpy {np.__version__} | "
              f"onnxruntime {onnxruntime.__version__} | mujoco {mujoco.__version__}")
        print(f"공식 샘플: {motion.name} | {len(motion.times)} frames @ 50 Hz ({motion.times[-1]:.1f} s)")
        print("준비 완료 → lab.assets / lab.contract / lab.policy / lab.sim / lab.motion")
    return Lab(a, contract, policy, sim, motion)
