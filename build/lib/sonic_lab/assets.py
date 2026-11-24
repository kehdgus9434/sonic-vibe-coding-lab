"""공식 자산 다운로드 (고정 리비전).

- NVlabs/GR00T-WholeBodyControl: 정책 파라미터 헤더, 시뮬레이터 설정, G1 MJCF/메쉬
- nvidia/GEAR-SONIC (Hugging Face): Encoder/Decoder ONNX, observation_config.yaml, 걷기 샘플

리비전은 07번 노트북이 Colab에서 검증한 값과 같습니다. 바꾸면 결과 재현성이 깨질 수 있습니다.
라이선스: 모델은 NVIDIA Open Model License, 코드/자산은 각 저장소 라이선스를 따릅니다.
이 저장소는 자산을 재배포하지 않고 실행 시 원본에서 내려받습니다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

GIT_REVISION = "4f5e118ae82e0448f2b2f441024d59fe74e0a4d9"
HF_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
SAMPLE = "walk_forward_amateur_001__A001.pkl"
SAMPLE_MIRRORED = "walk_forward_amateur_001__A001_M.pkl"
MODEL_REL = Path("gear_sonic/data/robot_model/model_data/g1")
HEADER_REL = Path("gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp")
SIM_CONFIG_REL = Path("gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml")

RAW = f"https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/{GIT_REVISION}"
MEDIA = f"https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{GIT_REVISION}"
HF = f"https://huggingface.co/nvidia/GEAR-SONIC/resolve/{HF_REVISION}"


@dataclass
class Assets:
    root: Path
    repo: Path         # GR00T-WholeBodyControl 일부 사본
    model_dir: Path    # ONNX + observation_config.yaml
    sample_root: Path  # smpl_filtered/, robot_filtered/210531/

    @property
    def scene_xml(self) -> Path:
        return self.repo / MODEL_REL / "scene_43dof.xml"


def default_root() -> Path:
    if os.environ.get("SONIC_LAB_CACHE"):
        return Path(os.environ["SONIC_LAB_CACHE"])
    if Path("/content").exists():
        return Path("/content/sonic_assets")
    return Path.home() / ".cache" / "sonic_lab"


def download(url: str, path: Path) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 100:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "sonic-vibe-coding-lab"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"Git LFS pointer instead of actual asset: {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def prepare(root: Path | str | None = None, meshes: bool = True, mirrored: bool = True, quiet: bool = False) -> Assets:
    """필요한 공식 자산만 받습니다(ONNX 약 91 MB + 샘플 약 3 MB + G1 메쉬). 이미 있으면 재사용."""
    root = Path(root) if root is not None else default_root()
    repo, model_dir, sample_root = root / "official_source", root / "models", root / "sample_data"
    for relative in [HEADER_REL, SIM_CONFIG_REL, MODEL_REL / "scene_43dof.xml", MODEL_REL / "g1_29dof_with_hand.xml"]:
        download(f"{RAW}/{relative.as_posix()}", repo / relative)
    if meshes:
        tree = ET.parse(repo / MODEL_REL / "g1_29dof_with_hand.xml")
        files = []
        for node in tree.findall("./asset/mesh"):
            name = node.attrib["file"]
            if Path(name).name != name:
                raise ValueError(f"Unexpected mesh filename: {name}")
            files.append(MODEL_REL / "meshes" / name)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda rel: download(f"{MEDIA}/{rel.as_posix()}", repo / rel), files))
    for filename in ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]:
        download(f"{HF}/{filename}", model_dir / filename)
    samples = [SAMPLE] + ([SAMPLE_MIRRORED] if mirrored else [])
    for sample in samples:
        for relative in [Path("smpl_filtered") / sample, Path("robot_filtered/210531") / sample]:
            download(f"{HF}/sample_data/{relative.as_posix()}", sample_root / relative)
    assets = Assets(root, repo, model_dir, sample_root)
    if not quiet:
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6
        print(f"공식 자산 준비 완료: {root} ({size:.0f} MB) | GitHub {GIT_REVISION[:8]} | HF {HF_REVISION[:8]}")
    return assets
