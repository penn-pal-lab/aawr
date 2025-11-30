import os
import random
import argparse
from typing import List, Tuple

import sys
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation

def find_traj_files(base_dir: str) -> List[str]:
    traj_files: List[str] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "trajectory.h5":
                traj_files.append(os.path.join(root, file))
    return traj_files


def load_cartesian_positions(file_path: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        data = f["observation/robot_state/cartesian_position"][:]  # (T, 6)
        return data[:, :3]  # xyz


def sample_trajectories(files: List[str], num: int, rng: random.Random) -> List[np.ndarray]:
    if not files:
        return []
    num_to_take = min(len(files), max(0, num))
    chosen = rng.sample(files, num_to_take)
    trajs: List[np.ndarray] = []
    for fpath in chosen:
        try:
            trajs.append(load_cartesian_positions(fpath))
        except Exception as e:
            print(f"[warn] Skipping invalid HDF5 {fpath}: {e}")
    return trajs


def get_success_cmap() -> LinearSegmentedColormap:
    # Success as blue gradient (start->finish): dark -> medium -> light blue
    return LinearSegmentedColormap.from_list(
        "success_blue",
        [(0.0, "#0D47A1"), (0.5, "#42A5F5"), (1.0, "#BBDEFB")]
    )


def get_failure_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "failure_red",
        [(0.0, "#B71C1C"), (0.5, "#EF5350"), (1.0, "#FFCDD2")]
    )


def compute_axis_limits(traj_groups: List[List[np.ndarray]], margin: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    chunks: List[np.ndarray] = []
    for group in traj_groups:
        if group:
            chunks.append(np.concatenate(group, axis=0))
    if not chunks:
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    all_xyz = np.concatenate(chunks, axis=0)
    x_min, y_min, z_min = np.min(all_xyz, axis=0)
    x_max, y_max, z_max = np.max(all_xyz, axis=0)
    return (x_min - margin, x_max + margin), (y_min - margin, y_max + margin), (z_min - margin, z_max + margin)


def apply_theme(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, dark_bg: bool) -> None:
    bg = "black" if dark_bg else "white"
    fg = "white" if dark_bg else "black"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlabel("X", fontsize=14, color=fg)
    ax.set_ylabel("Y", fontsize=14, color=fg)
    ax.set_zlabel("Z", fontsize=14, color=fg)
    ax.tick_params(axis="x", colors=fg)
    ax.tick_params(axis="y", colors=fg)
    ax.tick_params(axis="z", colors=fg)
    ax.grid(False)
    # Make 3D panes transparent (remove gray plane)
    try:
        subtle_gray = (0.6, 0.6, 0.6, 0.12)
        ax.xaxis.set_pane_color(subtle_gray)
        ax.yaxis.set_pane_color(subtle_gray)
        ax.zaxis.set_pane_color(subtle_gray)
    except Exception:
        pass
    # Emphasize axis lines at edges (white on dark bg, black on light bg)
    try:
        axisline_color = (1, 1, 1, 1) if dark_bg else (0, 0, 0, 1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            if hasattr(axis, "_axinfo"):
                axis._axinfo["axisline"]["color"] = axisline_color
                # Slightly thicker axis lines
                axis._axinfo["axisline"]["linewidth"] = 1.2
    except Exception:
        try:
            # Older mpl_toolkits API
            ax.w_xaxis.line.set_color("white" if dark_bg else "black")
            ax.w_yaxis.line.set_color("white" if dark_bg else "black")
            ax.w_zaxis.line.set_color("white" if dark_bg else "black")
        except Exception:
            pass

def get_common_start_point(success_trajs: List[np.ndarray], failure_trajs: List[np.ndarray]) -> np.ndarray:
    for group in (success_trajs, failure_trajs):
        for traj in group:
            if traj is not None and traj.shape[0] >= 1:
                return traj[0]
    return None

def draw_start_sphere(ax: matplotlib.axes.Axes, center: np.ndarray, limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], dark_bg: bool) -> None:
    if center is None:
        return
    (x_l, x_r), (y_l, y_r), (z_l, z_r) = limits
    lx = max(1e-6, x_r - x_l)
    ly = max(1e-6, y_r - y_l)
    lz = max(1e-6, z_r - z_l)
    radius = 0.02 * (lx + ly + lz) / 3.0
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    color = "white" if dark_bg else "black"
    try:
        ax.plot_surface(xs, ys, zs, color=color, edgecolor="none", linewidth=0, alpha=1.0, shade=True, antialiased=True)
    except Exception:
        ax.scatter(center[0], center[1], center[2], c=color, marker="o", s=300)

def plot_static_success_failure(success_trajs: List[np.ndarray], failure_trajs: List[np.ndarray], save_path: str, dark_bg: bool) -> None:
    print(f"[static] Plotting static image -> {save_path}")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    (x_l, x_r), (y_l, y_r), (z_l, z_r) = compute_axis_limits([success_trajs, failure_trajs], margin=0.05)
    ax.set_xlim([x_l, x_r])
    ax.set_ylim([y_l, y_r])
    ax.set_zlim([z_l, z_r])

    apply_theme(fig, ax, dark_bg)
    fg_color = "white" if dark_bg else "black"

    ax.view_init(elev=0, azim=-90)

    common_start = get_common_start_point(success_trajs, failure_trajs)
    draw_start_sphere(ax, common_start, ((x_l, x_r), (y_l, y_r), (z_l, z_r)), dark_bg)

    for traj in success_trajs:
        if traj.shape[0] < 2:
            continue
        segments = np.stack([traj[:-1], traj[1:]], axis=1)
        # Use warm colormap across trajectory (start->finish)
        norm_local = Normalize(vmin=0, vmax=traj.shape[0])
        lc = Line3DCollection(segments, cmap=get_success_cmap(), norm=norm_local, alpha=1.0, linewidth=2.4)
        lc.set_array(np.linspace(0, traj.shape[0], max(1, traj.shape[0] - 1)))
        try:
            lc.set_depthshade(False)
            lc.set_zsort("none")
        except Exception:
            pass
        ax.add_collection3d(lc)
        # end (star for success)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c=fg_color, marker="*", s=200, edgecolors=fg_color)

    for traj in failure_trajs:
        if traj.shape[0] < 2:
            continue
        segments = np.stack([traj[:-1], traj[1:]], axis=1)
        norm_local = Normalize(vmin=0, vmax=traj.shape[0])
        lc = Line3DCollection(segments, cmap=get_failure_cmap(), norm=norm_local, alpha=1.0, linewidth=2.4)
        lc.set_array(np.linspace(0, traj.shape[0], max(1, traj.shape[0] - 1)))
        try:
            lc.set_depthshade(False)
            lc.set_zsort("none")
        except Exception:
            pass
        ax.add_collection3d(lc)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c=fg_color, marker="X", s=110, edgecolors=fg_color, linewidths=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[static] Saved {save_path}")


def plot_rotating_success_failure_gif(success_trajs: List[np.ndarray], failure_trajs: List[np.ndarray], save_path: str, dark_bg: bool) -> None:
    print(f"[gif] Plotting rotating video -> {save_path}")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    (x_l, x_r), (y_l, y_r), (z_l, z_r) = compute_axis_limits([success_trajs, failure_trajs], margin=0.05)
    ax.set_xlim([x_l, x_r])
    ax.set_ylim([y_l, y_r])
    ax.set_zlim([z_l, z_r])

    apply_theme(fig, ax, dark_bg)
    fg_color = "white" if dark_bg else "black"

    ax.view_init(elev=15, azim=-90)

    success_cmap = get_success_cmap()
    failure_cmap = get_failure_cmap()

    max_steps = 1
    if success_trajs:
        max_steps = max(max_steps, max(tr.shape[0] for tr in success_trajs))
    if failure_trajs:
        max_steps = max(max_steps, max(tr.shape[0] for tr in failure_trajs))

    lines: List[Tuple[np.ndarray, Line3DCollection]] = []
    end_pts: List[np.ndarray] = []
    traj_lengths: List[int] = []
    traj_is_failure: List[bool] = []

    for traj in success_trajs:
        if traj.shape[0] < 2:
            continue
        segments = np.stack([traj[:-1], traj[1:]], axis=1)
        norm_local = Normalize(vmin=0, vmax=traj.shape[0])
        lc = Line3DCollection(segments, cmap=success_cmap, norm=norm_local, alpha=1.0, linewidth=2.2)
        lc.set_array(np.linspace(0, traj.shape[0], max(1, traj.shape[0] - 1)))
        try:
            lc.set_depthshade(False)
            lc.set_zsort("none")
        except Exception:
            pass
        ax.add_collection3d(lc)
        lines.append((traj, lc))
        end_pts.append(traj[-1])
        traj_lengths.append(traj.shape[0])
        traj_is_failure.append(False)

    for traj in failure_trajs:
        if traj.shape[0] < 2:
            continue
        segments = np.stack([traj[:-1], traj[1:]], axis=1)
        norm_local = Normalize(vmin=0, vmax=traj.shape[0])
        lc = Line3DCollection(segments, cmap=failure_cmap, norm=norm_local, alpha=1.0, linewidth=2.2)
        lc.set_array(np.linspace(0, traj.shape[0], max(1, traj.shape[0] - 1)))
        try:
            lc.set_depthshade(False)
            lc.set_zsort("none")
        except Exception:
            pass
        ax.add_collection3d(lc)
        lines.append((traj, lc))
        end_pts.append(traj[-1])
        traj_lengths.append(traj.shape[0])
        traj_is_failure.append(True)

    common_start = get_common_start_point(success_trajs, failure_trajs)
    draw_start_sphere(ax, common_start, ((x_l, x_r), (y_l, y_r), (z_l, z_r)), dark_bg)
    end_scats = []
    for is_fail in traj_is_failure:
        if is_fail:
            sc = ax.scatter([np.nan], [np.nan], [np.nan], c=fg_color, marker="X", s=110, edgecolors=fg_color, linewidths=0.8)
        else:
            sc = ax.scatter([np.nan], [np.nan], [np.nan], c=fg_color, marker="*", s=160, edgecolors=fg_color)
        end_scats.append(sc)

    pause_frames = 40
    total_frames = max_steps + pause_frames
    print(f"[gif] Frames: max_steps={max_steps}, pause={pause_frames}, total={total_frames}")

    def update(frame: int):
        draw_frame = min(frame, max_steps - 1)

        for traj, lc in lines:
            num_points = traj.shape[0]
            current_points = traj if draw_frame >= num_points else traj[:draw_frame]
            if current_points.shape[0] >= 2:
                segments = np.stack([current_points[:-1], current_points[1:]], axis=1)
                lc.set_segments(segments)
                lc.set_array(np.linspace(0, current_points.shape[0], max(1, current_points.shape[0] - 1)))

        for sc, (end_pt, traj_len, is_fail) in zip(end_scats, zip(end_pts, traj_lengths, traj_is_failure)):
            if draw_frame >= traj_len - 1:
                grow_frames = 10
                if is_fail:
                    base = 100
                    step = 5
                    final_cap = 170
                else:
                    base = 160
                    step = 8
                    final_cap = 240
                if draw_frame < traj_len - 1 + grow_frames:
                    size = base + step * (draw_frame - (traj_len - 1))
                else:
                    size = final_cap
                sc._offsets3d = ([end_pt[0]], [end_pt[1]], [end_pt[2]])
                try:
                    sc.set_sizes([size])
                except Exception:
                    pass
            else:
                sc._offsets3d = ([np.nan], [np.nan], [np.nan])

        azim = -90 + frame * 0.3
        ax.view_init(elev=15, azim=azim)
        return [lc for _, lc in lines] + end_scats

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)

    # Always save as MP4 for reliability.
    out_path = save_path
    if out_path.lower().endswith(".gif"):
        out_path = os.path.splitext(out_path)[0] + ".mp4"
        print(f"[gif] Saving as MP4 instead of GIF: {out_path}")
    elif not out_path.lower().endswith(".mp4"):
        out_path = f"{out_path}.mp4"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print("[gif] Saving MP4...")

    bar_width = 30

    def progress_callback(frame_idx: int, total: int):
        t = total if total and total > 0 else total_frames
        pct = (frame_idx + 1) / t
        filled = int(bar_width * pct)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r[gif] Saving: |{bar}| {frame_idx + 1}/{t} ({int(pct * 100)}%)")
        sys.stdout.flush()
        if frame_idx + 1 >= t:
            sys.stdout.write("\n")

    fps = 10
    try:
        if animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            try:
                anim.save(out_path, writer=writer, progress_callback=progress_callback)
            except TypeError:
                # Older Matplotlib versions may not support progress_callback.
                anim.save(out_path, writer=writer)
        else:
            raise RuntimeError("FFmpeg not available. Please install FFmpeg for fast MP4 saving.")
    finally:
        print(f"[gif] Saved {out_path}")
        plt.close(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize success/failure trajectories (static + rotating gif).")
    parser.add_argument("--input_dir", type=str, default="/mnt/sda/edward/projects/aawr/data/offline_data/bookshelf_d/data_vert_duck_valid")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualizations. Defaults to input_dir/3D_traj")
    parser.add_argument("--num_success", type=int, default=5)
    parser.add_argument("--num_failure", type=int, default=5)
    parser.add_argument("--light_bg", action="store_true", help="Use light background (white). Default is dark (black).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dark_bg = not args.light_bg

    print(f"[info] Scanning for trajectories...")
    args.success_dir = os.path.join(args.input_dir, "success")
    args.failure_dir = os.path.join(args.input_dir, "failure")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "3D_traj")
    os.makedirs(args.output_dir, exist_ok=True)
    succ_files = find_traj_files(args.success_dir) if os.path.isdir(args.success_dir) else []
    fail_files = find_traj_files(args.failure_dir) if os.path.isdir(args.failure_dir) else []
    print(f"[info] Found {len(succ_files)} success files, {len(fail_files)} failure files.")

    success_trajs = sample_trajectories(succ_files, args.num_success, rng)
    failure_trajs = sample_trajectories(fail_files, args.num_failure, rng)
    print(f"[info] Using {len(success_trajs)} success trajs and {len(failure_trajs)} failure trajs.")

    if not success_trajs and not failure_trajs:
        print("[error] No trajectories to visualize. Exiting.")
        return

    static_path = os.path.join(args.output_dir, f"traj_static_{args.num_success}_{args.num_failure}.png")
    video_path = os.path.join(args.output_dir, f"traj_rotate_{args.num_success}_{args.num_failure}.mp4")

    plot_static_success_failure(success_trajs, failure_trajs, static_path, dark_bg)
    plot_rotating_success_failure_gif(success_trajs, failure_trajs, video_path, dark_bg)

    print(f"[done] Wrote:\n - {static_path}\n - {video_path}")


if __name__ == "__main__":
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    main()


