"""Day 1 미션 작성 파일 (에이전트형 도구 사용자용). 노트북에서 `from my_work.day1 import *`로 불러옵니다."""
import numpy as np


def inspect_onnx(model_dir):
    """Mission 1-1: (encoder 입력 길이, encoder 출력 길이, decoder 입력 길이, decoder 출력 길이)"""
    raise NotImplementedError


def build_encoder_slices(yaml_path):
    """Mission 1-2: {encoder 관측 이름: (start, stop)} — YAML의 enabled 순서, 총 1762"""
    raise NotImplementedError


def detect_root_quat_order(robot_record):
    """Mission 1-3: 공식 G1 샘플 root_rot의 성분 순서 → 'xyzw' 또는 'wxyz'"""
    raise NotImplementedError


def resample_g1(robot_record, human_record, hz=50):
    """Mission 1-4: (times (T,), g1_q50 (T,29)) — 두 데이터 중 짧은 길이에 맞춘 50 Hz"""
    raise NotImplementedError


def build_encoder_obs_mode2(motion, frame, robot_xyzw, alignment):
    """Mission 1-5: Mode-2 encoder 입력 (1762,) float32"""
    raise NotImplementedError
