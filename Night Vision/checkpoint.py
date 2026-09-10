from __future__ import annotations

import logging
import os
import urllib.request

logger = logging.getLogger("netra.night_vision.download")

CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/"
    "main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth"
)


def download_checkpoint(dest_path: str = "weights/zero_dce_pp_epoch99.pth", force: bool = False) -> str:
    if os.path.exists(dest_path) and not force:
        logger.info("Checkpoint already present at %s, skipping download.", dest_path)
        return dest_path

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    logger.info("Downloading Zero-DCE++ checkpoint from %s", CHECKPOINT_URL)
    urllib.request.urlretrieve(CHECKPOINT_URL, dest_path)
    logger.info("Saved to %s", dest_path)
    return dest_path


if __name__ == "__main__":
    download_checkpoint()