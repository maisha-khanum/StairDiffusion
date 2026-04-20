# Class intended to manage kinematics data from Vive Ultimate tracker data
import numpy as np
from scipy.interpolate import interp1d
import os


class ViveKinematics:
    def __init__(self):
        # Raw tracker data (positions: Nx3, quaternions: Nx4)
        self.thigh_data = {
            'positions': None,
            'quaternions': None,
            'unix_time_ns': None,
            'xr_time_ns': None
        }
        self.shank_data = {
            'positions': None,
            'quaternions': None,
            'unix_time_ns': None,
            'xr_time_ns': None
        }
        self.foot_data = {
            'positions': None,
            'quaternions': None,
            'unix_time_ns': None,
            'xr_time_ns': None
        }
        
        # Interpolated data (will be populated after interpolate_timestamps)
        self.thigh_interp = None
        self.shank_interp = None
        self.foot_interp = None
        
        # Available tracker names in the NPZ file
        self.available_trackers = None
        
    def load_from_NPZ(self, npz_path: str):
        """
        Load tracker data from a single NPZ file.
        
        Args:
            npz_path: Path to the NPZ file containing all tracker data
        
        Expected NPZ structure:
            - tracker_names: array of tracker role names
            - {tracker_name}_positions: Nx3 array
            - {tracker_name}_quaternions: Nx4 array (w, x, y, z)
            - {tracker_name}_unix_time_ns: N array of Unix timestamps
            - {tracker_name}_xr_time_ns: N array of XR timestamps
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        # Load the NPZ file
        data = np.load(npz_path, allow_pickle=True)
        
        # Get available tracker names
        self.available_trackers = data['tracker_names']
        print(f"Available trackers: {self.available_trackers}")
        
        # Mapping of body parts to possible tracker role names
        tracker_mapping = {
            'thigh': ['left_knee', 'right_knee', 'left_thigh', 'right_thigh'],
            'shank': ['left_ankle', 'right_ankle', 'left_shank', 'right_shank'],
            'foot': ['left_foot', 'right_foot']
        }
        
        # Load thigh data
        for tracker_name in tracker_mapping['thigh']:
            if tracker_name in self.available_trackers:
                self.thigh_data['positions'] = data[f'{tracker_name}_positions']
                self.thigh_data['quaternions'] = data[f'{tracker_name}_quaternions']
                self.thigh_data['unix_time_ns'] = data[f'{tracker_name}_unix_time_ns']
                self.thigh_data['xr_time_ns'] = data[f'{tracker_name}_xr_time_ns']
                print(f"Loaded thigh data from: {tracker_name} ({len(self.thigh_data['positions'])} samples)")
                break
        
        # Load shank data
        for tracker_name in tracker_mapping['shank']:
            if tracker_name in self.available_trackers:
                self.shank_data['positions'] = data[f'{tracker_name}_positions']
                self.shank_data['quaternions'] = data[f'{tracker_name}_quaternions']
                self.shank_data['unix_time_ns'] = data[f'{tracker_name}_unix_time_ns']
                self.shank_data['xr_time_ns'] = data[f'{tracker_name}_xr_time_ns']
                print(f"Loaded shank data from: {tracker_name} ({len(self.shank_data['positions'])} samples)")
                break
        
        # Load foot data
        for tracker_name in tracker_mapping['foot']:
            if tracker_name in self.available_trackers:
                self.foot_data['positions'] = data[f'{tracker_name}_positions']
                self.foot_data['quaternions'] = data[f'{tracker_name}_quaternions']
                self.foot_data['unix_time_ns'] = data[f'{tracker_name}_unix_time_ns']
                self.foot_data['xr_time_ns'] = data[f'{tracker_name}_xr_time_ns']
                print(f"Loaded foot data from: {tracker_name} ({len(self.foot_data['positions'])} samples)")
                break
        
        # Check what was loaded
        loaded = []
        if self.thigh_data['positions'] is not None:
            loaded.append('thigh')
        if self.shank_data['positions'] is not None:
            loaded.append('shank')
        if self.foot_data['positions'] is not None:
            loaded.append('foot')
        
        print(f"Successfully loaded: {loaded}")
        
        if not loaded:
            print("Warning: No matching trackers found for thigh, shank, or foot!")
    
    # --------------------- DATA PROCESSING ------------------------- #
    def interpolate_timestamps(self, rgb_t: np.ndarray):
        """
        Interpolate Vive tracker data to match RGB timestamps.
        
        Args:
            rgb_t: Target timestamps (Unix time in nanoseconds or seconds)
                   Shape: (M,) where M is number of RGB frames
        
        Returns:
            dict: Interpolated data for each body part at RGB timestamps
        """
        # Convert rgb_t to seconds if it's in nanoseconds (assuming values > 1e12 are nanoseconds)
        if np.max(rgb_t) > 1e12:
            rgb_t_s = rgb_t / 1e9
        else:
            rgb_t_s = rgb_t
        
        interpolated_data = {}
        
        # Interpolate thigh data
        if self.thigh_data['positions'] is not None:
            t_thigh_s = self.thigh_data['unix_time_ns'] / 1e9
            
            # Interpolate positions (Nx3)
            interp_pos = interp1d(t_thigh_s, self.thigh_data['positions'], 
                                 axis=0, kind='linear', fill_value='extrapolate')
            
            # Interpolate quaternions (Nx4)
            interp_quat = interp1d(t_thigh_s, self.thigh_data['quaternions'], 
                                  axis=0, kind='linear', fill_value='extrapolate')
            
            self.thigh_interp = {
                'positions': interp_pos(rgb_t_s),
                'quaternions': interp_quat(rgb_t_s),
                'timestamps': rgb_t_s
            }
            interpolated_data['thigh'] = self.thigh_interp
            print(f"Interpolated thigh data: {len(self.thigh_interp['positions'])} samples")
        
        # Interpolate shank data
        if self.shank_data['positions'] is not None:
            t_shank_s = self.shank_data['unix_time_ns'] / 1e9
            
            interp_pos = interp1d(t_shank_s, self.shank_data['positions'], 
                                 axis=0, kind='linear', fill_value='extrapolate')
            interp_quat = interp1d(t_shank_s, self.shank_data['quaternions'], 
                                  axis=0, kind='linear', fill_value='extrapolate')
            
            self.shank_interp = {
                'positions': interp_pos(rgb_t_s),
                'quaternions': interp_quat(rgb_t_s),
                'timestamps': rgb_t_s
            }
            interpolated_data['shank'] = self.shank_interp
            print(f"Interpolated shank data: {len(self.shank_interp['positions'])} samples")
        
        # Interpolate foot data
        if self.foot_data['positions'] is not None:
            t_foot_s = self.foot_data['unix_time_ns'] / 1e9
            
            interp_pos = interp1d(t_foot_s, self.foot_data['positions'], 
                                 axis=0, kind='linear', fill_value='extrapolate')
            interp_quat = interp1d(t_foot_s, self.foot_data['quaternions'], 
                                  axis=0, kind='linear', fill_value='extrapolate')
            
            self.foot_interp = {
                'positions': interp_pos(rgb_t_s),
                'quaternions': interp_quat(rgb_t_s),
                'timestamps': rgb_t_s
            }
            interpolated_data['foot'] = self.foot_interp
            print(f"Interpolated foot data: {len(self.foot_interp['positions'])} samples")
        
        return interpolated_data
    
    # --------------------- KINEMATICS CALCULATIONS ------------------------- #
    def calculate_knee_angle(self):
        """
        Calculate knee flexion/extension angle from thigh and shank tracker data.
        """
        pass
    
    def calculate_foot_angles(self):
        """
        Calculate foot dorsiflexion/plantarflexion angles.
        """
        pass