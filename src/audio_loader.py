"""
Audio Loading and Feature Extraction Module
============================================

This module handles:
1. Loading audio files with librosa at 16 kHz
2. Extracting MFCC features with configurable parameters
3. Organizing data by machine type, noise condition, machine ID, and normal/abnormal labels
4. Support for all MIMII machine types: fan, pump, valve, slider

Key improvements:
- machine_type parameter: supports 'fan', 'pump', 'valve', 'slider'
- All three noise conditions: -6_dB, 0_dB, 6_dB SNR
- Dynamic data discovery and loading
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioLoader:
    """
    Load and manage audio files from the MIMII dataset.
    
    Supports all MIMII machine types and noise conditions.
    
    Attributes:
        data_root (Path): Root directory containing the data folders
        sr (int): Sampling rate (default 16000 Hz as per MIMII spec)
        n_mfcc (int): Number of MFCC coefficients to extract (default 13)
        machine_type (str): Type of machine ('fan', 'pump', 'valve', 'slider')
    """
    
    VALID_MACHINE_TYPES = ['fan', 'pump', 'valve', 'slider']
    VALID_CONDITIONS = ['-6_dB', '0_dB', '6_dB']
    
    def __init__(self, data_root: str = "./data", sr: int = 16000, n_mfcc: int = 13, 
                 machine_type: str = "fan"):
        """
        Initialize the AudioLoader.
        
        Args:
            data_root: Root directory of the dataset
            sr: Sampling rate in Hz (MIMII is 16 kHz)
            n_mfcc: Number of MFCC coefficients (typical: 13-40)
            machine_type: Type of machine ('fan', 'pump', 'valve', 'slider')
        
        Raises:
            ValueError: If machine_type is not valid
            FileNotFoundError: If data_root does not exist
        """
        self.data_root = Path(data_root)
        self.sr = sr
        self.n_mfcc = n_mfcc
        
        # Validate machine type
        if machine_type not in self.VALID_MACHINE_TYPES:
            raise ValueError(
                f"Invalid machine_type '{machine_type}'. "
                f"Must be one of: {self.VALID_MACHINE_TYPES}"
            )
        self.machine_type = machine_type
        
        # Validate that data_root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        print(f"✓ AudioLoader initialized")
        print(f"  - Data root: {self.data_root}")
        print(f"  - Machine type: {self.machine_type}")
        print(f"  - Sampling rate: {self.sr} Hz")
        print(f"  - MFCC coefficients: {self.n_mfcc}")
    
    def list_available_conditions(self) -> List[str]:
        """
        List all available noise conditions in the dataset.
        
        Returns:
            List of condition directory names (e.g., ['-6_dB', '0_dB', '6_dB'])
        """
        conditions = []
        for d in self.data_root.iterdir():
            if d.is_dir() and 'dB' in d.name:
                # Extract condition (e.g., '0_dB' from '0_dB_fan')
                parts = d.name.split('_dB_')
                if len(parts) == 2:
                    cond = parts[0] + '_dB'
                    if cond not in conditions and cond in self.VALID_CONDITIONS:
                        conditions.append(cond)
        
        conditions.sort(key=lambda x: (int(x.split('_')[0]), x))  # Sort by SNR value
        return conditions
    
    def list_machine_ids(self, condition: str) -> List[str]:
        """
        List all available machine IDs for a given noise condition.
        
        Args:
            condition: Noise condition name (e.g., '0_dB', not '0_dB_fan')
        
        Returns:
            List of machine IDs (e.g., ['id_00', 'id_02', 'id_04', 'id_06'])
        
        Raises:
            FileNotFoundError: If condition path does not exist
        """
        # Build path: data/0_dB_fan/fan/id_XX/...
        condition_dir = f"{condition}_{self.machine_type}"
        condition_path = self.data_root / condition_dir / self.machine_type
        
        if not condition_path.exists():
            raise FileNotFoundError(
                f"Condition path not found: {condition_path}\n"
                f"Looking for: data/{condition_dir}/{self.machine_type}/id_XX/..."
            )
        
        machine_ids = [d.name for d in condition_path.iterdir() if d.is_dir()]
        machine_ids.sort()
        return machine_ids
    
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load a single audio file using librosa.
        
        Args:
            file_path: Path to the WAV file
        
        Returns:
            Tuple of (audio_data, sampling_rate)
            - audio_data: numpy array of shape (n_samples,)
            - sampling_rate: int, the sampling rate
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio data.
        
        Args:
            audio_data: Audio time-series (numpy array from librosa.load)
        
        Returns:
            MFCC feature matrix of shape (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        return mfcc
    
    def aggregate_mfcc(self, mfcc: np.ndarray, method: str = 'mean') -> np.ndarray:
        """
        Aggregate MFCC across time dimension to get fixed-size feature vector.
        
        Args:
            mfcc: MFCC matrix of shape (n_mfcc, time_steps)
            method: Aggregation method ('mean', 'std', 'mean_std')
        
        Returns:
            Fixed-size feature vector
        """
        if method == 'mean':
            return np.mean(mfcc, axis=1)
        elif method == 'std':
            return np.std(mfcc, axis=1)
        elif method == 'mean_std':
            # Concatenate mean and std: (n_mfcc,) + (n_mfcc,) = (2*n_mfcc,)
            return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def load_all_clips_from_directory(self, 
                                     directory: str,
                                     label: str = None,
                                     aggregate_method: str = 'mean') -> Tuple[List[np.ndarray], List[str]]:
        """
        Load all WAV files from a directory and extract MFCC features.
        
        Args:
            directory: Path to directory containing WAV files
            label: Optional label for the clips (e.g., 'normal' or 'abnormal')
            aggregate_method: How to aggregate MFCC ('mean', 'std', 'mean_std')
        
        Returns:
            Tuple of (features_list, filenames_list)
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        features_list = []
        filenames_list = []
        failed_count = 0
        
        wav_files = sorted(dir_path.glob("*.wav"))
        
        for wav_file in wav_files:
            y, sr = self.load_audio_file(str(wav_file))
            
            if y is None:
                failed_count += 1
                continue
            
            mfcc = self.extract_mfcc(y)
            mfcc_aggregated = self.aggregate_mfcc(mfcc, method=aggregate_method)
            
            features_list.append(mfcc_aggregated)
            filenames_list.append(wav_file.name)
        
        if failed_count > 0:
            print(f"  ⚠ Failed to load {failed_count} files")
        
        return features_list, filenames_list
    
    def load_condition_dataset(self, 
                              condition: str,
                              machine_ids: List[str] = None,
                              aggregate_method: str = 'mean') -> Dict:
        """
        Load entire dataset for a given noise condition and machine type.
        
        Args:
            condition: Noise condition name (e.g., '0_dB', not '0_dB_fan')
            machine_ids: List of machine IDs to load (e.g., ['id_00', 'id_02', 'id_04'])
                        If None, loads all available IDs
            aggregate_method: MFCC aggregation method
        
        Returns:
            Dictionary with structure:
            {
                'condition': '0_dB',
                'machine_type': 'fan',
                'machine_ids': {
                    'id_00': {
                        'normal': {'features': [...], 'filenames': [...]},
                        'abnormal': {'features': [...], 'filenames': [...]}
                    },
                    ...
                }
            }
        """
        if machine_ids is None:
            machine_ids = self.list_machine_ids(condition)
        
        dataset = {
            'condition': condition,
            'machine_type': self.machine_type,
            'machine_ids': {}
        }
        
        print(f"\nLoading condition: {condition} ({self.machine_type})")
        
        for machine_id in machine_ids:
            dataset['machine_ids'][machine_id] = {}
            
            # Load normal sounds
            normal_dir = self.data_root / f"{condition}_{self.machine_type}" / self.machine_type / machine_id / "normal"
            print(f"  Loading {machine_id} normal sounds...", end=" ")
            normal_features, normal_filenames = self.load_all_clips_from_directory(
                str(normal_dir),
                label='normal',
                aggregate_method=aggregate_method
            )
            dataset['machine_ids'][machine_id]['normal'] = {
                'features': normal_features,
                'filenames': normal_filenames
            }
            print(f"✓ {len(normal_features)} clips")
            
            # Load abnormal sounds
            abnormal_dir = self.data_root / f"{condition}_{self.machine_type}" / self.machine_type / machine_id / "abnormal"
            print(f"  Loading {machine_id} abnormal sounds...", end=" ")
            abnormal_features, abnormal_filenames = self.load_all_clips_from_directory(
                str(abnormal_dir),
                label='abnormal',
                aggregate_method=aggregate_method
            )
            dataset['machine_ids'][machine_id]['abnormal'] = {
                'features': abnormal_features,
                'filenames': abnormal_filenames
            }
            print(f"✓ {len(abnormal_features)} clips")
        
        return dataset


if __name__ == "__main__":
    print("Audio loader module ready! Import and use AudioLoader class.")
    print(f"\nSupported machine types: {AudioLoader.VALID_MACHINE_TYPES}")
    print(f"Supported conditions: {AudioLoader.VALID_CONDITIONS}")
