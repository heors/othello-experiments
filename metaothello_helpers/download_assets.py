"""Download the classic assets used by the continuation experiment."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

LOGGER = logging.getLogger(__name__)

HF_DATA_REPO = "aviralchawla/metaothello"
HF_MODEL_REPO = "aviralchawla/metaothello"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _move_if_missing(src: Path, dest: Path) -> None:
    if dest.exists():
        LOGGER.info("Skipping %s; already exists.", dest.relative_to(REPO_ROOT))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    LOGGER.info("Saved %s", dest.relative_to(REPO_ROOT))


def download_classic_data(data_name: str = "train_classic_20M.zarr") -> None:
    """Download the classic training Zarr store into ``data/classic``."""
    dest = DATA_DIR / "classic" / data_name
    if dest.exists():
        LOGGER.info("Skipping %s; already exists.", dest.relative_to(REPO_ROOT))
        return

    LOGGER.info("Downloading classic data store %s", data_name)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=[f"{data_name}/**"],
            local_dir=tmp,
        )
        src = Path(tmp) / data_name
        if not src.exists():
            raise FileNotFoundError(f"Could not find {data_name} in downloaded dataset snapshot")
        _move_if_missing(src, dest)


def download_classic_checkpoint(epoch: int = 50) -> None:
    """Download the classic checkpoint for the requested epoch."""
    ckpt_name = f"epoch_{epoch}.ckpt"
    dest = DATA_DIR / "classic" / "ckpts" / ckpt_name
    if dest.exists():
        LOGGER.info("Skipping %s; already exists.", dest.relative_to(REPO_ROOT))
        return

    LOGGER.info("Downloading classic checkpoint %s", ckpt_name)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            allow_patterns=[f"classic/{ckpt_name}"],
            local_dir=tmp,
        )
        src = Path(tmp) / "classic" / ckpt_name
        if not src.exists():
            raise FileNotFoundError(f"Could not find classic/{ckpt_name} in downloaded model snapshot")
        _move_if_missing(src, dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download classic MetaOthello continuation assets.")
    parser.add_argument("--epoch", type=int, default=50, help="Classic checkpoint epoch to download.")
    parser.add_argument(
        "--data_name",
        type=str,
        default="train_classic_20M.zarr",
        help="Classic Zarr store name in the HuggingFace dataset repo.",
    )
    parser.add_argument("--skip_ckpt", action="store_true", help="Do not download the checkpoint.")
    parser.add_argument("--skip_data", action="store_true", help="Do not download the Zarr data.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    args = parse_args()
    if not args.skip_data:
        download_classic_data(args.data_name)
    if not args.skip_ckpt:
        download_classic_checkpoint(args.epoch)


if __name__ == "__main__":
    main()
