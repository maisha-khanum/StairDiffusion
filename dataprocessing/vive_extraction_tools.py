"""
Hosts tools for extracting kinematic data from the HTC Vive system.
This includes functions for parsing the raw data files and converting them into a usable format for analysis.

NPZ file format:
  - tracker_names: array of tracker role name strings (e.g. ['right_foot', 'right_knee'])
  - num_trackers: scalar int
  - {tracker_name}_positions:    Nx3 float32, (x, y, z) in meters
  - {tracker_name}_quaternions:  Nx4 float32, (w, x, y, z) unit quaternions
  - {tracker_name}_unix_time_ns: N int64, Unix timestamps in nanoseconds
  - {tracker_name}_xr_time_ns:   N int64, OpenXR performance-counter timestamps in nanoseconds

Tracker-to-body-segment mapping (matches get_kinematics.py):
  thigh -> right_knee / left_knee / right_thigh / left_thigh
  shank -> right_ankle / left_ankle / right_shank / left_shank
  foot  -> right_foot / left_foot
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tracker-to-segment mapping (keep in sync with get_kinematics.py)
# ---------------------------------------------------------------------------
TRACKER_MAPPING = {
    'thigh': ['waist'],
    'shank': ['left_knee', 'right_knee'],
    'foot':  ['left_foot', 'right_foot'],
}


def get_vive_info(npz_path: str) -> dict:
    """
    Return metadata about a Vive NPZ recording without loading the full arrays.

    Args:
        npz_path: Path to an NPZ file produced by the Vive tracker system.

    Returns:
        dict with keys:
            'tracker_names' : list[str]
            'num_trackers'  : int
            'sample_counts' : dict {tracker_name: int}
            'duration_s'    : dict {tracker_name: float}  (derived from unix timestamps)
            'sample_rates'  : dict {tracker_name: float}  (Hz, derived from unix timestamps)
            'segments'      : dict {segment: tracker_name} for matched thigh/shank/foot
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    tracker_names = list(data['tracker_names'])
    num_trackers = int(data['num_trackers'])

    sample_counts = {}
    duration_s = {}
    sample_rates = {}
    for name in tracker_names:
        t = data[f'{name}_unix_time_ns']
        n = len(t)
        sample_counts[name] = n
        dur = (t[-1] - t[0]) / 1e9 if n > 1 else 0.0
        duration_s[name] = dur
        sample_rates[name] = (n - 1) / dur if dur > 0 else float('nan')

    # Identify which segment each tracker maps to
    segments = {}
    for segment, candidates in TRACKER_MAPPING.items():
        for candidate in candidates:
            if candidate in tracker_names:
                segments[segment] = candidate
                break

    return {
        'tracker_names': tracker_names,
        'num_trackers': num_trackers,
        'sample_counts': sample_counts,
        'duration_s': duration_s,
        'sample_rates': sample_rates,
        'segments': segments,
    }


def get_vive_data(npz_path: str, trackers: list = None) -> dict:
    """
    Load position and quaternion data from a Vive NPZ file.

    Args:
        npz_path: Path to the NPZ file.
        trackers: Optional list of tracker names to load. If None, all trackers
                  present in the file are loaded.

    Returns:
        dict keyed by tracker name, each value is a dict:
            'positions'    : Nx3 float32 array (x, y, z) in metres
            'quaternions'  : Nx4 float32 array (w, x, y, z)
            'unix_time_ns' : N int64 array of Unix timestamps in nanoseconds
            'xr_time_ns'   : N int64 array of OpenXR timestamps in nanoseconds
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    available = list(data['tracker_names'])

    if trackers is None:
        trackers = available
    else:
        missing = [t for t in trackers if t not in available]
        if missing:
            raise ValueError(f"Requested trackers not in file: {missing}. Available: {available}")

    result = {}
    for name in trackers:
        result[name] = {
            'positions':    data[f'{name}_positions'],
            'quaternions':  data[f'{name}_quaternions'],
            'unix_time_ns': data[f'{name}_unix_time_ns'],
            'xr_time_ns':   data[f'{name}_xr_time_ns'],
        }

    return result


def get_thigh_frame(npz_path: str, duration: float = 2.0) -> np.ndarray:
    """
    Estimate a stable reference orientation for the thigh tracker.

    The thigh is assumed to be approximately still during the first `duration`
    seconds of the recording.  The reference frame is the mean rotation over
    that window, returned as a 3x3 rotation matrix.

    Args:
        npz_path: Path to the NPZ file.
        duration: Length of the initial still window in seconds (default 2 s).

    Returns:
        3x3 numpy array — the mean rotation matrix of the thigh tracker over
        the still window, to be used as a reference (neutral) frame.

    Raises:
        ValueError if no thigh tracker is found in the file.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    available = list(data['tracker_names'])

    # Find the thigh tracker
    thigh_name = None
    for candidate in TRACKER_MAPPING['thigh']:
        if candidate in available:
            thigh_name = candidate
            break

    if thigh_name is None:
        raise ValueError(f"No thigh tracker found in file. Available trackers: {available}")

    t_ns = data[f'{thigh_name}_unix_time_ns']
    quats = data[f'{thigh_name}_quaternions']  # Nx4, (w, x, y, z)

    # Select samples within the first `duration` seconds
    t_s = (t_ns - t_ns[0]) / 1e9
    mask = t_s <= duration
    still_quats = quats[mask]

    if len(still_quats) == 0:
        still_quats = quats[:1]

    # scipy Rotation expects (x, y, z, w); our data is (w, x, y, z)
    still_quats_xyzw = still_quats[:, [1, 2, 3, 0]]
    rotations = R.from_quat(still_quats_xyzw)
    mean_rotation = rotations.mean()

    return mean_rotation.as_matrix()


def compute_euler_angles(quaternions: np.ndarray, sequence: str = 'XYZ',
                         degrees: bool = True) -> np.ndarray:
    """
    Convert an array of quaternions to Euler angles.

    Args:
        quaternions: Nx4 array in (w, x, y, z) convention.
        sequence:    Euler angle sequence passed to scipy (default 'XYZ').
        degrees:     If True, output is in degrees; otherwise radians.

    Returns:
        Nx3 array of Euler angles.
    """
    # scipy expects (x, y, z, w)
    quats_xyzw = quaternions[:, [1, 2, 3, 0]]
    rotations = R.from_quat(quats_xyzw)
    return rotations.as_euler(sequence, degrees=degrees)


def get_reference_quaternions(npz_path: str, duration: float = 2.0) -> dict:
    """
    Compute a mean reference quaternion for every tracker in a recording by
    averaging over the first `duration` seconds (the subject should be still
    in a known neutral pose during this window).

    Each tracker is strapped on at a slightly different orientation each
    session, so storing these reference quaternions lets downstream code
    cancel out that mounting offset when computing joint angles.

    Args:
        npz_path: Path to the NPZ file.
        duration: Length of the initial still window in seconds (default 2 s).

    Returns:
        dict {tracker_name: np.ndarray shape (4,)} — mean quaternion in
        (w, x, y, z) convention, one per tracker in the file.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    tracker_names = list(data['tracker_names'])

    refs = {}
    for name in tracker_names:
        t_ns = data[f'{name}_unix_time_ns']
        quats = data[f'{name}_quaternions']  # Nx4, (w, x, y, z)

        t_s = (t_ns - t_ns[0]) / 1e9
        mask = t_s <= duration
        still_quats = quats[mask] if mask.any() else quats[:1]

        # scipy expects (x, y, z, w)
        still_xyzw = still_quats[:, [1, 2, 3, 0]]
        mean_rot = R.from_quat(still_xyzw).mean()

        # Store back as (w, x, y, z) to stay consistent with the file format
        xyzw = mean_rot.as_quat()
        refs[name] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)

    return refs


def compute_relative_rotation(quat_proximal: np.ndarray,
                              quat_distal: np.ndarray,
                              ref_proximal: np.ndarray = None,
                              ref_distal: np.ndarray = None,
                              degrees: bool = True) -> np.ndarray:
    """
    Compute the rotation of a distal segment relative to a proximal segment,
    optionally corrected for per-session tracker mounting offsets.

    Because trackers are physically re-attached at different orientations each
    recording, raw world-frame quaternions carry an arbitrary mounting offset.
    Passing reference quaternions (captured in a neutral standing pose at the
    start of the session) removes that offset so the result reflects the true
    anatomical joint angle change.

    Without references (raw):
        R_joint(t) = R_prox(t)^-1 · R_dist(t)

    With references (calibrated):
        ΔR_prox(t) = R_prox_ref^-1 · R_prox(t)   # change from neutral for each tracker
        ΔR_dist(t) = R_dist_ref^-1 · R_dist(t)
        R_joint(t) = ΔR_prox(t)^-1 · ΔR_dist(t)  # mounting offset cancels out

    Args:
        quat_proximal: Nx4 array (w, x, y, z) for the proximal segment.
        quat_distal:   Nx4 array (w, x, y, z) for the distal segment.
                       Must be at the same timestamps as quat_proximal (interpolate first).
        ref_proximal:  (4,) reference quaternion (w, x, y, z) for the proximal tracker
                       in the neutral pose.  Use get_reference_quaternions() to obtain.
                       If None, no calibration is applied.
        ref_distal:    (4,) reference quaternion (w, x, y, z) for the distal tracker.
                       If None, no calibration is applied.
        degrees:       If True, output Euler angles in degrees.

    Returns:
        Nx3 array of joint Euler angles (XYZ sequence: flexion, ab/adduction, rotation).
        Values represent change from the neutral reference pose when refs are provided,
        or raw relative orientation when refs are omitted.
    """
    if (ref_proximal is None) != (ref_distal is None):
        raise ValueError("Provide both ref_proximal and ref_distal, or neither.")

    prox_xyzw = quat_proximal[:, [1, 2, 3, 0]]
    dist_xyzw = quat_distal[:, [1, 2, 3, 0]]

    r_prox = R.from_quat(prox_xyzw)
    r_dist = R.from_quat(dist_xyzw)

    if ref_proximal is not None:
        # Convert references from (w,x,y,z) to (x,y,z,w) for scipy
        r_prox_ref = R.from_quat(ref_proximal[[1, 2, 3, 0]])
        r_dist_ref = R.from_quat(ref_distal[[1, 2, 3, 0]])

        # Express each tracker's motion relative to its own neutral orientation
        delta_prox = r_prox_ref.inv() * r_prox
        delta_dist = r_dist_ref.inv() * r_dist

        r_joint = delta_prox.inv() * delta_dist
    else:
        r_joint = r_prox.inv() * r_dist

    return r_joint.as_euler('XYZ', degrees=degrees)


def compute_joint_rotation(quat_proximal: np.ndarray,
                           quat_distal: np.ndarray,
                           ref_proximal: np.ndarray = None,
                           ref_distal: np.ndarray = None) -> R:
    """
    Compute the calibrated relative rotation between two segments and return the
    raw scipy Rotation object (not decomposed into Euler angles).

    Same calibration logic as compute_relative_rotation; use this when you need
    the rotation in a representation other than Euler angles (e.g. rotation
    vectors for PCA, or swing-twist decomposition for hinge-joint angles).

    Args:
        quat_proximal: Nx4 (w, x, y, z)
        quat_distal:   Nx4 (w, x, y, z)
        ref_proximal:  (4,) reference quaternion (w, x, y, z), or None
        ref_distal:    (4,) reference quaternion (w, x, y, z), or None

    Returns:
        scipy Rotation of length N representing the joint rotation at each frame.
    """
    if (ref_proximal is None) != (ref_distal is None):
        raise ValueError("Provide both ref_proximal and ref_distal, or neither.")

    r_prox = R.from_quat(quat_proximal[:, [1, 2, 3, 0]])
    r_dist = R.from_quat(quat_distal[:, [1, 2, 3, 0]])

    if ref_proximal is not None:
        r_prox_ref = R.from_quat(ref_proximal[[1, 2, 3, 0]])
        r_dist_ref = R.from_quat(ref_distal[[1, 2, 3, 0]])
        delta_prox = r_prox_ref.inv() * r_prox
        delta_dist = r_dist_ref.inv() * r_dist
        return delta_prox.inv() * delta_dist
    else:
        return r_prox.inv() * r_dist


def find_flexion_axis(r_joint: R) -> np.ndarray:
    """
    Find the dominant rotation axis of a joint from a series of relative
    rotations, using PCA on the rotation vectors.

    For a hinge joint (knee, ankle) the relative rotation is approximately
    confined to one axis regardless of how the trackers are mounted.  PCA on
    the rotation vectors recovers that axis from the data.

    Only frames with rotation above the median angle are used so that noisy
    near-zero frames don't pollute the estimate.

    Args:
        r_joint: scipy Rotation of length N (output of compute_joint_rotation).

    Returns:
        (3,) unit vector — the flexion axis in the joint's local frame.
        Oriented so that the largest observed rotation has a positive projection
        (i.e. peak flexion is positive).
    """
    rotvecs = r_joint.as_rotvec()                    # Nx3, axis * angle (rad)
    angles  = np.linalg.norm(rotvecs, axis=1)        # N

    # Use only frames with meaningful rotation to avoid noise dominating PCA
    threshold = np.median(angles)
    mask = angles > threshold
    if mask.sum() < 3:
        mask = np.ones(len(rotvecs), dtype=bool)

    _, _, Vt = np.linalg.svd(rotvecs[mask], full_matrices=False)
    axis = Vt[0]  # first right singular vector = direction of maximum variance

    # Orient so peak-flexion frame has positive projection
    peak_idx = np.argmax(angles)
    if np.dot(rotvecs[peak_idx], axis) < 0:
        axis = -axis

    return axis


def compute_hinge_angle(r_joint: R, flexion_axis: np.ndarray,
                        degrees: bool = True) -> np.ndarray:
    """
    Extract the signed rotation angle about a single hinge axis from a series
    of joint rotations (swing-twist decomposition).

    For each frame the full 3-D relative rotation is decomposed into:
      - Twist: rotation about `flexion_axis` — this is the flexion angle
      - Swing: rotation perpendicular to `flexion_axis` — discarded

    The twist angle is derived directly from the quaternion without converting
    to Euler angles, so it is free from gimbal lock and axis-alignment
    assumptions.

    Formula: given q = (w, x, y, z) and unit axis a:
        proj  = dot((x,y,z), a)          # signed projection of vector part
        angle = 2 * arctan2(proj, w)     # signed twist angle

    Args:
        r_joint:       scipy Rotation of length N.
        flexion_axis:  (3,) unit vector defining the flexion axis, in the same
                       frame as r_joint (output of find_flexion_axis).
        degrees:       If True, return degrees; otherwise radians.

    Returns:
        (N,) array of signed flexion angles.
        Positive = flexion direction of `flexion_axis`.
    """
    flexion_axis = flexion_axis / np.linalg.norm(flexion_axis)  # ensure unit

    # scipy stores as (x,y,z,w) internally; as_quat() returns (x,y,z,w)
    quat_xyzw = r_joint.as_quat()  # Nx4
    w   = quat_xyzw[:, 3].copy()
    vec = quat_xyzw[:, :3].copy()  # (x,y,z)

    # q and -q represent the same rotation, but arctan2 gives different values
    # for each.  Enforce w >= 0 (canonical form) so angles stay in (-180, 180).
    neg = w < 0
    w[neg]   = -w[neg]
    vec[neg] = -vec[neg]

    proj   = vec @ flexion_axis    # signed projection onto flexion axis
    angles = 2.0 * np.arctan2(proj, w)

    return np.degrees(angles) if degrees else angles


def get_axis_in_world(quaternions: np.ndarray, local_axis: np.ndarray) -> np.ndarray:
    """
    Rotate a fixed local axis into world space for each frame.

    Args:
        quaternions: Nx4 array in (w, x, y, z) convention.
        local_axis:  (3,) vector in the tracker's local frame (need not be unit).

    Returns:
        Nx3 array of unit vectors — the local axis expressed in world space at
        each frame.
    """
    local_axis = np.asarray(local_axis, dtype=float)
    local_axis = local_axis / np.linalg.norm(local_axis)

    rots = R.from_quat(quaternions[:, [1, 2, 3, 0]])   # (x,y,z,w) for scipy
    return rots.apply(local_axis)                        # Nx3


def compute_axis_flexion(quat_proximal: np.ndarray,
                         local_axis_prox: np.ndarray,
                         quat_distal: np.ndarray,
                         local_axis_dist: np.ndarray,
                         ref_quat_prox: np.ndarray,
                         ref_quat_dist: np.ndarray,
                         degrees: bool = True) -> np.ndarray:
    """
    Compute the flexion angle as the change in the angle between a chosen local
    axis on each tracker, relative to the angle measured during the still
    calibration window.

    Method
    ------
    1. For every frame, rotate each tracker's chosen local axis into world space.
    2. Compute the angle between the two world-space axes via arccos of their dot
       product.  This gives a value in [0°, 180°].
    3. Subtract the reference angle (same calculation applied to the calibration
       quaternions) so that the output is 0° when the subject is in the neutral
       standing pose.

    This approach is independent of the arbitrary mounting orientation of each
    tracker because only the *chosen* local axis — e.g. the axis that points
    roughly along the limb segment — is used.

    Args:
        quat_proximal:   Nx4 (w, x, y, z) — world-frame quaternions of the
                         proximal tracker at every frame.
        local_axis_prox: (3,) vector in the proximal tracker's local frame
                         (e.g. (0,0,1) for Z axis).
        quat_distal:     Nx4 (w, x, y, z) — world-frame quaternions of the
                         distal tracker at every frame.
        local_axis_dist: (3,) vector in the distal tracker's local frame
                         (e.g. (0,0,1) or (-1,0,0)).
        ref_quat_prox:   (4,) mean quaternion (w, x, y, z) for the proximal
                         tracker during the still calibration window.
        ref_quat_dist:   (4,) mean quaternion (w, x, y, z) for the distal
                         tracker during the still calibration window.
        degrees:         If True, return degrees; otherwise radians.

    Returns:
        (N,) array — flexion angle change from neutral pose (positive = opening
        angle between the two axes increasing from the reference).
    """
    local_axis_prox = np.asarray(local_axis_prox, dtype=float)
    local_axis_dist = np.asarray(local_axis_dist, dtype=float)
    local_axis_prox = local_axis_prox / np.linalg.norm(local_axis_prox)
    local_axis_dist = local_axis_dist / np.linalg.norm(local_axis_dist)

    # World-space axes at every frame
    prox_axes = get_axis_in_world(quat_proximal, local_axis_prox)   # Nx3
    dist_axes = get_axis_in_world(quat_distal,   local_axis_dist)   # Nx3

    dots   = np.clip(np.einsum('ij,ij->i', prox_axes, dist_axes), -1.0, 1.0)
    angles = np.arccos(dots)                                         # N, radians

    # Reference angle from the still calibration quaternions
    ref_prox_world = R.from_quat(ref_quat_prox[[1, 2, 3, 0]]).apply(local_axis_prox)
    ref_dist_world = R.from_quat(ref_quat_dist[[1, 2, 3, 0]]).apply(local_axis_dist)
    ref_angle      = np.arccos(np.clip(np.dot(ref_prox_world, ref_dist_world), -1.0, 1.0))

    flexion = angles - ref_angle
    return np.degrees(flexion) if degrees else flexion


def plot_tracker_positions(npz_path: str, tracker_name: str = None,
                           axes: tuple = (0, 1, 2)) -> None:
    """
    Quick-look plot of tracker position over time.

    Args:
        npz_path:     Path to the NPZ file.
        tracker_name: Name of the tracker to plot. If None, plots the first one.
        axes:         Which position axes to plot (0=x, 1=y, 2=z).
    """
    data = get_vive_data(npz_path)
    if tracker_name is None:
        tracker_name = list(data.keys())[0]

    t_s = (data[tracker_name]['unix_time_ns'] - data[tracker_name]['unix_time_ns'][0]) / 1e9
    pos = data[tracker_name]['positions']
    labels = ['x', 'y', 'z']

    plt.figure(figsize=(10, 4))
    for ax_idx in axes:
        plt.plot(t_s, pos[:, ax_idx], label=labels[ax_idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'{tracker_name} position — {os.path.basename(npz_path)}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tracker_orientations(npz_path: str, tracker_name: str = None,
                              sequence: str = 'XYZ', degrees: bool = True) -> None:
    """
    Quick-look plot of tracker Euler angles over time.

    Args:
        npz_path:     Path to the NPZ file.
        tracker_name: Name of the tracker to plot. If None, plots the first one.
        sequence:     Euler sequence for scipy (default 'XYZ').
        degrees:      If True, plot in degrees.
    """
    data = get_vive_data(npz_path)
    if tracker_name is None:
        tracker_name = list(data.keys())[0]

    t_s = (data[tracker_name]['unix_time_ns'] - data[tracker_name]['unix_time_ns'][0]) / 1e9
    angles = compute_euler_angles(data[tracker_name]['quaternions'],
                                  sequence=sequence, degrees=degrees)
    unit = 'deg' if degrees else 'rad'

    plt.figure(figsize=(10, 4))
    for i, label in enumerate(list(sequence)):
        plt.plot(t_s, angles[:, i], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Angle ({unit})')
    plt.title(f'{tracker_name} orientation ({sequence}) — {os.path.basename(npz_path)}')
    plt.legend()
    plt.tight_layout()
    plt.show()
