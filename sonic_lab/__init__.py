"""SONIC 바이브 코딩 실습 패키지.

    from sonic_lab import start, check
    lab = start()
"""

from .lab import start  # noqa: F401
from . import check  # noqa: F401

__version__ = "1.0.0"
