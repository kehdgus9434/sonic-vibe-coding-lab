"""Day 3 미션 작성 파일. 노트북에서 `from my_work.day3 import *`로 불러오고, PR 자동 리포트도 이 파일을 실행합니다."""
import dataclasses

import numpy as np

from sonic_lab import reference as ref


def joint_error_by_group(arrays, lab):
    """Mission 3-1: 부위별 평균 관절 오차(deg). 반환 예: {"legs": 3.1, "waist": 1.2, "arms": 7.8}"""
    raise NotImplementedError


def experiment(lab, N=1000):
    """Mission 3-2: {"라벨": ref.run_physics(...) 결과, ...}. 첫 항목은 기준 조건."""
    results = {"기준(mode 2)": ref.run_physics(lab.sim, lab.policy, lab.motion, mode=2, max_frames=N)}
    # results["내 조건"] = ...
    return results
