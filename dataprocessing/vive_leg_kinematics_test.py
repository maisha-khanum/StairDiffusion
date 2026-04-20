import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))
from vive_leg_kinematics import compute_flexion
from vive_extraction_tools import find_flexion_axis


def plot_tracker_positions_3d(results: list[tuple[str, dict]], title: str = '') -> None:
    """
    3D trajectory of tracker positions coloured by time.

    Args:
        results: list of (label, result_dict) pairs where result_dict comes from
                 compute_flexion.  Plots proximal_pos from the first entry and
                 distal_pos from every entry, labelled accordingly.
        title:   Figure title.
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Collect all positions to set consistent axis limits
    all_pos = []

    for _, r in results:
        for pos in [r['proximal_pos'], r['distal_pos']]:
            t      = r['t_s']
            points = pos                                              # Nx3
            segs   = np.stack([points[:-1], points[1:]], axis=1)     # (N-1)x2x3
            t_norm = (t[:-1] - t[0]) / (t[-1] - t[0])

            lc = Line3DCollection(segs, cmap='plasma', linewidth=0.8, alpha=0.85)
            lc.set_array(t_norm)
            ax.add_collection3d(lc)

            ax.scatter(*pos[0],  color='green', s=30, zorder=5)
            ax.scatter(*pos[-1], color='red',   s=30, zorder=5)
            all_pos.append(pos)

    all_pos = np.vstack(all_pos)
    ax.set_xlim(all_pos[:, 0].min(), all_pos[:, 0].max())
    ax.set_ylim(all_pos[:, 1].min(), all_pos[:, 1].max())
    ax.set_zlim(all_pos[:, 2].min(), all_pos[:, 2].max())
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title or 'Tracker positions')

    # Single colourbar for time
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, label='Time (normalised)')

    # Legend: green = start, red = end
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=8, label='end'),
    ], loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()


def compute_windowed_flexion_axes(result: dict, stride: int = 200,
                                  window: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the PCA flexion axis in rolling windows across the recording.

    Args:
        result: Output of compute_flexion (must contain 'joint_quat' and 't_s').
        stride: Step between window centres in samples.
        window: Number of samples per window.

    Returns:
        t_centres: 1-D array of timestamps at each window centre.
        axes:      Nx3 array of unit flexion-axis vectors, one per window.
    """
    quats_wxyz = result['joint_quat']   # Nx4 (w,x,y,z)
    t          = result['t_s']
    n          = len(quats_wxyz)

    centres, axes = [], []
    for start in range(0, n - window, stride):
        end     = start + window
        window_q = R.from_quat(quats_wxyz[start:end, [1, 2, 3, 0]])  # (x,y,z,w) for scipy
        axis    = find_flexion_axis(window_q)
        centres.append(t[start + window // 2])
        axes.append(axis)

    return np.array(centres), np.array(axes)


def plot_flexion_axis_evolution_3d(t_centres: np.ndarray, axes: np.ndarray,
                                   title: str = 'Flexion axis over time') -> None:
    """
    3D quiver plot of the flexion axis at each window, coloured by time.
    All arrows originate from the origin — only direction matters.
    """
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    t_norm = (t_centres - t_centres[0]) / (t_centres[-1] - t_centres[0])
    cmap   = plt.cm.plasma

    for axis, tn in zip(axes, t_norm):
        color = cmap(tn)
        ax.quiver(0, 0, 0, *axis, length=0.8, normalize=True,
                  color=color, linewidth=1.2, arrow_length_ratio=0.15)

    # Unit-sphere wireframe for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.04, color='gray')

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_centres[0], t_centres[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, label='Time (s)')

    plt.tight_layout()
    plt.show()


def animate_tracker_frames(knee: dict, ankle: dict,
                           step: int = 5,
                           arrow_scale: float = 0.08,
                           title: str = '') -> None:
    """
    3D animation of tracker coordinate frames over time.

    Each tracker is drawn as a position dot plus three orthogonal arrows:
        red   = local X axis
        green = local Y axis
        blue  = local Z axis

    Trackers shown:
        waist       — from knee['proximal']
        right_knee  — from knee['distal']  (== ankle['proximal'])
        right_foot  — from ankle['distal']

    Args:
        knee:        Output of compute_flexion for knee joint.
        ankle:       Output of compute_flexion for ankle joint.
        step:        Subsample every N frames (default 5 → 100 Hz from 500 Hz).
        arrow_scale: Length of each axis arrow in metres (default 0.08 m).
        title:       Figure / window title.
    """
    from matplotlib.animation import FuncAnimation
    from scipy.spatial.transform import Rotation as Rot

    # --- collect trackers: (label, positions Nx3, quaternions Nx4 w,x,y,z) ---
    trackers = [
        ('waist',      knee['proximal_pos'],  knee['proximal_quat']),
        ('right_knee', knee['distal_pos'],     knee['distal_quat']),
        ('right_foot', ankle['distal_pos'],    ankle['distal_quat']),
    ]
    tracker_colors = ['tab:purple', 'tab:orange', 'tab:cyan']
    axis_colors    = ['red', 'green', 'blue']   # X, Y, Z

    t   = knee['t_s']
    idx = np.arange(0, len(t), step)
    n_frames = len(idx)

    # --- pre-compute rotation matrices for all trackers at subsampled frames ---
    rot_mats = []   # list of (n_frames x 3 x 3) per tracker
    for _, _, quats in trackers:
        # quats is (w,x,y,z) → scipy wants (x,y,z,w)
        rots = Rot.from_quat(quats[idx][:, [1, 2, 3, 0]])
        rot_mats.append(rots.as_matrix())   # n_frames x 3 x 3

    # --- set up figure ---
    all_pos = np.vstack([pos for _, pos, _ in trackers])
    margin  = 0.1
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    time_text = ax.set_title('')

    # --- create persistent artists for each tracker ---
    dots  = []   # one scatter per tracker
    lines = []   # 3 lines per tracker (X/Y/Z axes)

    for tc, (label, *_) in zip(tracker_colors, trackers):
        dot, = ax.plot([], [], [], 'o', color=tc, markersize=6, label=label)
        dots.append(dot)
        tracker_lines = []
        for ac in axis_colors:
            ln, = ax.plot([], [], [], '-', color=ac, linewidth=1.5)
            tracker_lines.append(ln)
        lines.append(tracker_lines)

    ax.legend(loc='upper left', fontsize=8)

    def update(frame_i):
        for t_idx, (_, pos, _) in enumerate(trackers):
            p = pos[idx[frame_i]]
            R = rot_mats[t_idx][frame_i]   # 3x3

            dots[t_idx].set_data_3d([p[0]], [p[1]], [p[2]])

            for axis_i in range(3):
                tip = p + arrow_scale * R[:, axis_i]
                lines[t_idx][axis_i].set_data_3d(
                    [p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]]
                )

        time_text.set_text(f'{title}  t = {t[idx[frame_i]]:.2f} s')
        return [time_text] + dots + [ln for tl in lines for ln in tl]

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000 * step / 500,   # matches real time at 500 Hz
                        blit=False)

    plt.tight_layout()
    plt.show()
    return ani   # keep reference alive


def plot_joint_flexion(knee: dict, ankle: dict, title: str = '') -> None:
    """Two-panel plot: knee flexion (top) and ankle flexion (bottom), shared time axis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title)

    axes[0].plot(knee['t_s'], knee['flexion_deg'], color='steelblue', linewidth=1.0)
    axes[0].axhline(0, color='gray', linewidth=0.6, linestyle='--')
    axes[0].set_ylabel('Flexion (deg)')
    axes[0].set_title('Knee (waist → right_knee)')

    axes[1].plot(ankle['t_s'], ankle['flexion_deg'], color='darkorange', linewidth=1.0)
    axes[1].axhline(0, color='gray', linewidth=0.6, linestyle='--')
    axes[1].set_ylabel('Flexion (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Ankle (right_knee → right_foot)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    DATASET = '/home/maisha/StairDiffusion/npz/walk_full_vid'
    
    npz = os.path.join(DATASET, os.path.basename(DATASET) + '.npz')

    knee  = compute_flexion(npz, proximal='waist',      distal='right_knee')
    ankle = compute_flexion(npz, proximal='right_knee', distal='right_foot')

    for label, r in [('Knee', knee), ('Ankle', ankle)]:
        ang = r['flexion_deg']
        print(f'{label}: [{ang.min():.1f}, {ang.max():.1f}] deg  '
              f'peak={ang.max():.1f} at t={r["t_s"][np.argmax(ang)]:.2f}s')

    plot_joint_flexion(knee, ankle, title=os.path.basename(DATASET))

    animate_tracker_frames(knee, ankle, title=os.path.basename(DATASET))


    # t_centres, axes = compute_windowed_flexion_axes(knee, stride=200, window=200)
    # plot_flexion_axis_evolution_3d(t_centres, axes, title=f'Knee flexion axis — {os.path.basename(DATASET)}')


    # t_centres, axes = compute_windowed_flexion_axes(ankle, stride=200, window=200)
    # plot_flexion_axis_evolution_3d(t_centres, axes, title=f'Ankle flexion axis — {os.path.basename(DATASET)}')
