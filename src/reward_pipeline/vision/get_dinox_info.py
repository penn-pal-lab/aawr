"""
Process episodes with DINO-X object detection.

This module handles running DINO-X object detection on episode frames,
saving the masks and detection info for each frame.
"""
from __future__ import annotations
import json, logging, os
from pathlib import Path
import cv2
import  numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List


from dinox_detector import DINOX
from dinox_utils import _rle2mask, _string2rle, _decode_and_save_mask
from dinox_utils import atomic_write_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


import time
import functools
import shutil

def timer(func):
    """A decorator thdinoxat prints the execution time of the function it decorates."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer



# --- atomic writers ---------------------------------------------------------
@atomic_write_files
def _write_json(content, fp):
    with open(fp, "w") as f:
        json.dump(content, f, indent=2)


@atomic_write_files
def _write_npy(arr: np.ndarray, fp):
    np.save(fp, arr)


def build_frame_list(episode_root: Path, indices: list[int]) -> tuple[list[int], dict]:
    """
    Returns
    -------
    todo   : list[int]      # Frames that need DINO-X processing
    dinfo  : dict           # Existing dinox_info (empty dict if none exists)
    """
    dinox_json = episode_root / "dinox_info.json"
    dinfo = json.load(open(dinox_json)) if dinox_json.exists() else {}

    done = set(int(k) for k, v in dinfo.items() if v and v[0]["mask_file"])
    # Can also check vis/ directory
    vis_dir = episode_root / "vis"
    if vis_dir.exists():
        for jpg in vis_dir.glob("*_bbox_mask.jpg"):
            done.add(int(jpg.stem.split("_")[0]))

    todo = [i for i in indices if i not in done]
    print(f"todo: {todo}")
    return todo, dinfo


@timer
def _process_episode(
    h5_path: Path,
    frames_dir: Path,
    dinox: DINOX,
    prompt: str | None = None,
    use_seek: bool = False,
    skip_completed: bool = False,
    num_workers: int = 4,                    
    object_name: str = "yellow pineapple toy"
) -> bool:
    """
    Each slow DINO-X request is submitted to a thread pool.
    Every worker writes its own mask.npy via `safe_write_npy`
    so no two threads can collide.
    """
    episode_root   = h5_path.parent
    completion_flag = episode_root / "dinox_done.md"
    if skip_completed and completion_flag.exists():
        logger.warning("%s already done - skip", episode_root.name)
        return True

    # ---------- load whitelist ----------
    wl_file = episode_root / "white_list.json"
    indices: List[int] = json.load(open(wl_file))
    if not indices:
        logger.warning("empty whitelist - skip")
        return False

    # ---------- I/O paths ----------
    mask_dir = episode_root / "masks"
    vis_dir  = episode_root / "vis"
    mask_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # ---------- per-frame worker ----------
    def _worker(fidx: int) -> Dict:
        img_path = frames_dir / f"{fidx:05d}.jpg"
        if not img_path.exists():
            return {"frame": fidx, "entries": []}

        pred = dinox.get_dinox(str(img_path), prompt) if not use_seek \
               else dinox.get_dinox_seek(str(img_path), prompt)

        frame_info = []
        # BUG FIX: if no pred, the subfunction will write a "Nothing detected" image
        dinox.visualize_bbox_and_mask(pred, img_path, vis_dir, f"{fidx:05d}")
        if pred:
            for j, obj in enumerate(pred):
                if obj.get("category") != object_name:
                    # TODO: this is a hack to avoid processing other objects
                    # TODO: remove this once we have a more robust way to handle this
                    continue
                mask_file = f"{fidx:05d}_pineapple_{j}.npy"
                arr = _rle2mask(_string2rle(obj["mask"]["counts"]), obj["mask"]["size"]).astype(np.uint8)
                safe_name = f"{fidx:05d}_pineapple_{j}.npy"
                _write_npy(safe_name, arr, mask_dir)

                frame_info.append({
                    "bbox": obj["bbox"],
                    "mask_file": mask_file,
                    "category": obj["category"],
                    "score": round(obj["score"], 5),
                })
        if not frame_info:   # no pineapple → empty stub
            frame_info = [{"bbox": [], "mask_file": "", "category": "", "score": 0.0}]
        
        return {"frame": fidx, "entries": frame_info}

    todo, dinox_info = build_frame_list(episode_root, indices)
    # if not todo:
    #     logger.warning("  nothing left to do - skip API calls"); return True

    logger.info("  %d / %d frames remain", len(todo), len(indices))

    # ---------- submit to pool ----------
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        fut2idx = {pool.submit(_worker, idx): idx for idx in todo}
        for fut in as_completed(fut2idx):
            res = fut.result()
            dinox_info[str(res["frame"])] = res["entries"]

    # ---------- atomic episode-level JSON ----------
    _write_json("dinox_info.json", dinox_info, episode_root)

    completion_flag.touch()      # light-weight flag
    logger.info("done %s - %d frames processed", episode_root.name, len(indices))
    return True


# ───────────────────────── High-level driver ───────────────────────── #
@timer
def dinox_dataset(
    data_dir: Path,
    stop_after: int | None = None,
    prompt: str | None = None,
    use_seek: bool = False,
    skip_completed: bool = False,
    num_workers: int = 4,
    object_name: str = "yellow pineapple toy"
):
    """
    Traverse *data_dir* recursively, run the full privileged-info extraction
    for every `trajectory.h5` found.
    
    Args:
        data_dir: Root directory containing trajectory data
        stop_after: Maximum number of episodes to process
        prompt: Text prompt for DINO-X
        use_seek: Whether to use seek mode in DINO-X
        skip_completed: Skip folders with .dinox_completed flag
        num_workers: Number of workers for multiprocessing
    """
    dinox = DINOX()
    processed = 0
    successful = 0

    for h5_path in sorted(data_dir.rglob("trajectory.h5")):
        rel = h5_path.relative_to(data_dir)
        logger.info("Episode: %s", rel.parent)

        frames_dir = h5_path.parent / "recordings" / "frames" / "hand_camera"
        if not frames_dir.exists():
            logger.warning("  frames dir missing → skip episode")
            continue

        success = _process_episode(
            h5_path, frames_dir, dinox, 
            prompt=prompt, use_seek=use_seek,
            skip_completed=skip_completed,
            num_workers=num_workers,
            object_name=object_name
        )
        
        if success:
            successful += 1

        processed += 1
        if stop_after and processed >= stop_after:
            break

    logger.info("Finished. Processed %d episode(s), %d successful.", processed, successful)
    return processed, successful


def test():
    NUM_WORKERS = 2
    DATA_DIR = Path("/home/edward/projects/fowm/pi0_data/test/")
    print(f"Processing directory: {DATA_DIR}")
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    dinox_dataset(DATA_DIR, prompt="yellow pineapple toy. toy", use_seek=False, skip_completed=False, num_workers=NUM_WORKERS)
   

def reset_test_dir():
    DATA_DIR = Path("/home/edward/projects/fowm/pi0_data/test/")
    TEST_COPY_DIR = Path("/home/edward/projects/fowm/pi0_data/test_copy/")
    
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    
    if TEST_COPY_DIR.exists():
        shutil.copytree(TEST_COPY_DIR, DATA_DIR)
    else:
        logger.warning(f"Test copy directory {TEST_COPY_DIR} does not exist. Creating empty test directory.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

@timer
def main():
    import sys
    DATA_ROOT = Path(sys.argv[1])  
    NUM_WORKERS = int(sys.argv[2])
    print(f"RUNNING DINOX ON {NUM_WORKERS} WORKERS")
    OBJECT_NAME = sys.argv[3]
    print(f"OBJECT NAME: {OBJECT_NAME}")
    input("WARNING: Are you sure you want to run DINOX on this object? Press Enter to continue")
    for outcome in ['success', 'failure']:
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            continue
        for date_dir in outcome_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            for h5_path in date_dir.rglob("trajectory.h5"):
                data_dir = h5_path.parent
                logger.info(f"Processing directory: {data_dir}")

                dinox_dataset(
                    data_dir, 
                    prompt=f"{OBJECT_NAME}. toy . box. book. bowl", 
                    use_seek=False, 
                    skip_completed=False,
                    num_workers=NUM_WORKERS,
                    object_name=OBJECT_NAME
                )

if __name__ == "__main__":
    main()
