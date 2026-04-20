# loads vive data with thigh and shank markers, and calculates the knee angles using the method in vive_extraction_tools.py

import os
import sys
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(__file__))
from vive_extraction_tools import (
    get_vive_data,
    get_reference_quaternions,
    compute_joint_rotation,
    find_flexion_axis,
    compute_hinge_angle,
)

# Calibration: stand still then perform pure flexion
STILL_DURATION_S = 0.1   # seconds standing still  → neutral reference quaternions
FLEX_DURATION_S  = 3.0   # seconds of pure flexion → anatomical flexion axis


def interpolate_to_common_time(data: dict, target_hz: float = 500.0) -> dict:
    """
    Resample all trackers in `data` to a shared uniform timeline.

    The shared timeline spans the overlap of all trackers' recording windows,
    sampled at `target_hz`.

    Args:
        data:       Output of get_vive_data — {tracker_name: {positions, quaternions,
                    unix_time_ns, xr_time_ns}}.
        target_hz:  Output sample rate in Hz (default 500 Hz).

    Returns:
        dict {tracker_name: {'positions': Nx3, 'quaternions': Nx4, 't_s': N}}
        where t_s is the shared timeline in seconds.
    """
    # Common time window: overlap of all trackers
    t_starts = [d['unix_time_ns'][0] for d in data.values()]
    t_ends   = [d['unix_time_ns'][-1] for d in data.values()]
    t_start_s = max(t_starts) / 1e9
    t_end_s   = min(t_ends)   / 1e9

    t_common = np.arange(t_start_s, t_end_s, 1.0 / target_hz)

    result = {}
    for name, d in data.items():
        t_s = d['unix_time_ns'] / 1e9

        # Drop duplicate timestamps — they cause divide-by-zero in interp1d and
        # produce NaN values that make scipy's Rotation reject the quaternions.
        _, unique_idx = np.unique(t_s, return_index=True)
        t_s      = t_s[unique_idx]
        pos_u    = d['positions'][unique_idx]
        quat_u   = d['quaternions'][unique_idx].copy()

        # Ensure consistent quaternion hemisphere before interpolating.
        # Linear interp between q and -q (same rotation, opposite sign) passes
        # through zero, producing a near-180° spurious rotation after renorm.
        # Flip each quaternion so it stays on the same hemisphere as its predecessor.
        for i in range(1, len(quat_u)):
            if np.dot(quat_u[i], quat_u[i - 1]) < 0:
                quat_u[i] = -quat_u[i]

        interp_pos  = interp1d(t_s, pos_u,   axis=0, kind='linear',
                               bounds_error=False, fill_value=(pos_u[0],  pos_u[-1]))
        interp_quat = interp1d(t_s, quat_u, axis=0, kind='linear',
                               bounds_error=False, fill_value=(quat_u[0], quat_u[-1]))

        pos_interp  = interp_pos(t_common)
        quat_interp = interp_quat(t_common)

        # Re-normalise quaternions after linear interpolation
        norms = np.linalg.norm(quat_interp, axis=1, keepdims=True)
        quat_interp = quat_interp / norms

        result[name] = {
            'positions':   pos_interp.astype(np.float32),
            'quaternions': quat_interp.astype(np.float32),
            't_s':         t_common,
        }

    return result


def compute_flexion(npz_path: str,
                    proximal: str,
                    distal: str,
                    still_duration: float = STILL_DURATION_S,
                    flex_duration: float = FLEX_DURATION_S,
                    target_hz: float = 500.0) -> dict:
    """
    Compute the flexion angle at any joint defined by a proximal and distal
    tracker.  Works for knee (waist → right_knee), ankle (right_knee → right_foot),
    or any other hinge pair present in the NPZ file.

    Recording protocol at the start:
        [0, still_duration)              — stand still → neutral reference quaternions
        [still_duration, still_duration + flex_duration) — pure flexion → anatomical axis

    Args:
        npz_path:       Path to the NPZ file.
        proximal:       Tracker name for the proximal segment (e.g. 'waist').
        distal:         Tracker name for the distal segment (e.g. 'right_knee').
        still_duration: Seconds to stand still at recording start (default 0.1 s).
        flex_duration:  Seconds of pure flexion after the still phase (default 3 s).
        target_hz:      Resample rate for the common timeline (Hz).

    Returns:
        dict with keys:
            't_s'          : 1-D array, time in seconds from recording start
            'flexion_deg'  : 1-D array, flexion angle in degrees
                             (positive = flexion, negative = hyperextension)
            'flexion_axis' : (3,) unit vector — anatomical flexion axis from calib window
    """
    data = get_vive_data(npz_path, trackers=[proximal, distal])
    refs = get_reference_quaternions(npz_path, duration=still_duration)
    interp_data = interpolate_to_common_time(data, target_hz=target_hz)

    t_common = interp_data[proximal]['t_s']
    t_s = t_common - t_common[0]

    r_joint = compute_joint_rotation(
        interp_data[proximal]['quaternions'],
        interp_data[distal]['quaternions'],
        ref_proximal=refs[proximal],
        ref_distal=refs[distal],
    )

    # Flexion axis from the pure-flexion window only.
    # The subject stands still for still_duration, then performs a clean
    # single-plane flexion for flex_duration. PCA on that window gives
    # the anatomical axis, which is then held fixed for the full recording.
    n_still      = int(np.searchsorted(t_s, still_duration))
    n_flex_end   = int(np.searchsorted(t_s, still_duration + flex_duration))
    flexion_axis = find_flexion_axis(r_joint[n_still:n_flex_end])
    flexion_deg  = compute_hinge_angle(r_joint, flexion_axis, degrees=True)

    # Store joint rotation as (w,x,y,z) array for downstream windowed analysis
    quat_xyzw     = r_joint.as_quat()                             # Nx4 (x,y,z,w)
    joint_quat    = quat_xyzw[:, [3, 0, 1, 2]]                   # Nx4 (w,x,y,z)

    return {
        't_s':              t_s,
        'flexion_deg':      flexion_deg,
        'flexion_axis':     flexion_axis,
        'joint_quat':       joint_quat,                            # Nx4 (w,x,y,z) calibrated relative rotation
        'proximal_pos':     interp_data[proximal]['positions'],    # Nx3
        'proximal_quat':    interp_data[proximal]['quaternions'],  # Nx4 (w,x,y,z) world-frame
        'distal_pos':       interp_data[distal]['positions'],      # Nx3
        'distal_quat':      interp_data[distal]['quaternions'],    # Nx4 (w,x,y,z) world-frame
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    DATASET = '/home/maisha/StairDiffusion/npz/walk_corner'
    
    npz = os.path.join(DATASET, os.path.basename(DATASET) + '.npz')

    knee  = compute_flexion(npz, proximal='waist',      distal='right_knee') # get flexion axis here, make it a 
    ankle = compute_flexion(npz, proximal='right_knee', distal='right_foot')

    for label, r in [('Knee', knee), ('Ankle', ankle)]:
        ang = r['flexion_deg']
        print(f'{label}: range=[{ang.min():.1f}, {ang.max():.1f}] deg  '
              f'peak={ang.max():.1f} deg at t={r["t_s"][np.argmax(ang)]:.2f}s')
