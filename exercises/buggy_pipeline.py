"""Day 2 버그 사냥 대상: AI가 작성한 'SONIC 폐루프 파이프라인'.

이 코드는 오류 없이 실행되고 그럴듯한 출력을 냅니다. 그런데 G1은 몇 초 안에 넘어집니다.
공식 입출력 계약(docs/01_contract.md)과 다른 부분이 모두 4곳 있습니다.

규칙
- 이 파일만 수정합니다. sonic_lab/reference.py, sonic_lab/check.py는 열지 않습니다(정답이 들어 있음).
- 바이브 코딩 도구에게 계약서와 check 출력 메시지를 근거로 주고, '추측'이 아니라 '테스트'로 원인을 찾게 하세요.
- 고친 곳마다 아래 BUGLOG에 근거를 한 줄씩 남깁니다.

BUGLOG (발견 순서대로)
1.
2.
3.
4.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

HISTORY_KEYS = ["omega", "q", "dq", "action", "gravity"]


class Pipeline:
    """run_physics(..., pipeline=Pipeline())에 꽂아 쓰는 세 단계."""

    def observation(self, sim, previous_action):
        """현재 MuJoCo 상태 → 정책용 고유수용 관측 1프레임.

        omega   : 기저 각속도 (3,)  — free joint qvel[3:6], 기저 좌표계
        q       : 29관절 각도 (29,) — IsaacLab 순서
        dq      : 29관절 속도 (29,) — IsaacLab 순서
        action  : 직전 정책 출력 (29,)
        gravity : 기저 좌표계에서 본 중력 방향 (3,)
        """
        c, d = sim.contract, sim.data
        base_rotation = R.from_quat(d.qpos[3:7])           # 기저 쿼터니언
        return {
            "omega": d.qvel[3:6].copy(),
            "q": d.qpos[sim.qaddr][c.mujoco_to_isaaclab],              # MuJoCo → IsaacLab 재배열
            "dq": d.qvel[sim.vaddr][c.mujoco_to_isaaclab].copy(),
            "action": np.asarray(previous_action).copy(),
            "gravity": base_rotation.inv().apply([0, 0, -1]),
        }

    def decoder_obs(self, token, history):
        """64차원 토큰 + 최근 10프레임 이력 → Decoder 입력 994차원.

        history는 deque(maxlen=10)이며 append로 쌓였으므로 history[-1]이 최신입니다.
        """
        recent_first = list(reversed(history))               # 최신 프레임을 앞에 둔다
        parts = [np.asarray(token, np.float32).reshape(64)]
        for key in HISTORY_KEYS:
            parts.append(np.asarray([frame[key] for frame in recent_first]).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def action_to_target(self, action, contract):
        """정책 출력(IsaacLab 순서) → PD 목표 관절각(MuJoCo 순서, rad)."""
        action_mj = np.asarray(action)[contract.mujoco_to_isaaclab]    # IsaacLab → MuJoCo
        return contract.default + action_mj * contract.scale
