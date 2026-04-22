"""
Preprocessing Modifications Module
===================================

Implements two preprocessing modifications to address the domain shift problem:

1. Per-Machine-ID Normalization: Remove machine-specific acoustic identity
2. Variance-Weighted MFCC Selection: Downweight features encoding machine identity
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import pickle


class PreprocessingModifications:
    """
    Implements preprocessing modifications to reduce domain shift in anomaly detection.
    
    The domain shift problem: Models trained on one machine fail on another of the same type
    because each physical machine has a slightly different acoustic signature even when healthy.
    
    Our approach:
    1. Per-Machine-ID Normalization: Subtract mean MFCC of each machine from all its clips
    2. Variance-Weighted MFCC Selection: Downweight coefficients with high between-ID variance
    """
    
    def __init__(self):
        """Initialize the preprocessing modifier"""
        self.machine_means = {}  # Store means for each machine ID
        self.variance_weights = None  # Store weights for each MFCC coefficient
        self.is_fitted = False
        
        print("✓ PreprocessingModifications initialized")
    
    # =========================================================================
    # MODIFICATION 1: PER-MACHINE-ID NORMALIZATION
    # =========================================================================
    
    def fit_per_machine_normalization(self, 
                                      X_train: np.ndarray,
                                      machine_ids: List[str]) -> None:
        """
        Fit per-machine-ID normalization by computing mean MFCC for each machine.
        
        Args:
            X_train: Training data with shape (n_samples, n_features)
                    Data from multiple machines concatenated together
            machine_ids: List of machine IDs corresponding to X_train segments
                        e.g., ['id_00', 'id_00', ..., 'id_02', 'id_02', ...]
                        Must have same length as X_train
        
        Example:
            >>> X_train = load training data
            >>> machine_ids = ['id_00']*1000 + ['id_02']*1000
            >>> preprocessor = PreprocessingModifications()
            >>> preprocessor.fit_per_machine_normalization(X_train, machine_ids)
        """
        if len(machine_ids) != X_train.shape[0]:
            raise ValueError(
                f"Length of machine_ids ({len(machine_ids)}) must match "
                f"number of samples in X_train ({X_train.shape[0]})"
            )
        
        print("\n" + "="*70)
        print("FITTING MODIFICATION 1: Per-Machine-ID Normalization")
        print("="*70)
        
        # Compute mean MFCC for each unique machine ID
        unique_ids = sorted(set(machine_ids))
        
        for machine_id in unique_ids:
            # Get all samples for this machine
            mask = np.array(machine_ids) == machine_id
            X_machine = X_train[mask]
            
            # Compute mean
            machine_mean = X_machine.mean(axis=0)
            self.machine_means[machine_id] = machine_mean
            
            print(f"  {machine_id}: computed mean from {X_machine.shape[0]} samples")
            print(f"    Sample mean values: {machine_mean[:5]}")
    
    def apply_per_machine_normalization(self, 
                                       X: np.ndarray,
                                       machine_id: str) -> np.ndarray:
        """
        Apply per-machine-ID normalization to data.
        
        Subtracts the machine-specific mean from all features, removing the
        machine's "acoustic identity" and leaving only deviations from normal.
        
        Args:
            X: Data with shape (n_samples, n_features)
            machine_id: Machine ID (must have been seen during fitting)
        
        Returns:
            Normalized data with shape (n_samples, n_features)
        
        Example:
            >>> X_normalized = preprocessor.apply_per_machine_normalization(X_test, 'id_04')
        """
        if machine_id not in self.machine_means:
            raise ValueError(
                f"Machine ID '{machine_id}' not found in fitted means. "
                f"Available: {list(self.machine_means.keys())}"
            )
        
        # Subtract machine mean from all samples of that machine
        X_normalized = X - self.machine_means[machine_id]
        
        return X_normalized
    
    # =========================================================================
    # MODIFICATION 2: VARIANCE-WEIGHTED MFCC SELECTION
    # =========================================================================
    
    def fit_variance_weighting(self,
                              X_train: np.ndarray,
                              machine_ids: List[str],
                              threshold_percentile: float = 75.0) -> None:
        """
        Fit variance-weighted MFCC selection.
        
        Computes the variance of each MFCC coefficient:
        - BETWEEN machines (different machines have different "acoustic identity")
        - WITHIN machines (same machine has consistent MFCC)
        
        Coefficients with high between-ID variance encode machine identity (bad).
        Coefficients with high within-ID variance encode fault state (good).
        
        We downweight coefficients where between > within.
        
        Args:
            X_train: Training data with shape (n_samples, n_features)
            machine_ids: List of machine IDs corresponding to X_train segments
            threshold_percentile: Percentile of variance ratio to use as threshold
                                 (default 75 = downweight top 25% of coefficients)
        
        Example:
            >>> preprocessor.fit_variance_weighting(X_train, machine_ids)
        """
        if len(machine_ids) != X_train.shape[0]:
            raise ValueError(
                f"Length of machine_ids ({len(machine_ids)}) must match "
                f"number of samples in X_train ({X_train.shape[0]})"
            )
        
        print("\n" + "="*70)
        print("FITTING MODIFICATION 2: Variance-Weighted MFCC Selection")
        print("="*70)
        
        n_features = X_train.shape[1]
        unique_ids = sorted(set(machine_ids))
        
        # Compute within-ID variance (variance within each machine's data)
        within_variance = np.zeros(n_features)
        for machine_id in unique_ids:
            mask = np.array(machine_ids) == machine_id
            X_machine = X_train[mask]
            within_variance += X_machine.var(axis=0)
        within_variance /= len(unique_ids)  # Average across machines
        
        # Compute between-ID variance
        # Strategy: compute mean for each machine, then variance of means
        machine_means = np.zeros((len(unique_ids), n_features))
        for i, machine_id in enumerate(unique_ids):
            mask = np.array(machine_ids) == machine_id
            X_machine = X_train[mask]
            machine_means[i] = X_machine.mean(axis=0)
        
        between_variance = machine_means.var(axis=0)
        
        # Compute variance ratio
        epsilon = 1e-10
        variance_ratio = between_variance / (within_variance + epsilon)
        
        print(f"\n  Between-ID variance (machine identity): min={between_variance.min():.4f}, max={between_variance.max():.4f}")
        print(f"  Within-ID variance (fault state):      min={within_variance.min():.4f}, max={within_variance.max():.4f}")
        print(f"  Variance ratio (between/within):       min={variance_ratio.min():.4f}, max={variance_ratio.max():.4f}")
        
        threshold = np.percentile(variance_ratio, threshold_percentile)
        self.variance_weights = np.ones(n_features)
        
        # WEIGHTING FORMULA: weight = 1 / (1 + ratio/max_ratio)
        # ========================================================
        # Properties:
        # - Monotonically decreasing in variance_ratio
        # - Maps high between-variance → low weight, low between-variance → high weight
        # - Range: [0.5, 1.0] (floor at 0.5 is deliberate)
        # - 0.5 floor avoids zeroing any coefficient entirely
        #
        # Justification for 0.5 floor:
        # Even coefficients encoding machine identity may carry some fault-relevant signal.
        # Zeroing them entirely could lose information. A weight of 0.5 = 50% reduction.
        #
        # ALTERNATIVE: Bayesian interpretation
        # weight_alt = within / (within + between + eps)
        # - Cleaner probabilistic meaning: fraction of variance due to fault state
        # - Range [0, 1]
        # - More principled, but can zero coefficients entirely
        #
        # We use the first formula for robustness.
        
        high_variance_indices = variance_ratio > threshold
        max_ratio = variance_ratio.max()
        self.variance_weights[high_variance_indices] = 1.0 / (1.0 + variance_ratio[high_variance_indices] / max_ratio)
        
        print(f"\n  Variance ratio threshold (percentile {threshold_percentile}): {threshold:.4f}")
        print(f"  Coefficients downweighted: {high_variance_indices.sum()}/{n_features}")
        print(f"  Weight range: [{self.variance_weights.min():.4f}, {self.variance_weights.max():.4f}]")
        
        # Show which coefficients are downweighted
        if high_variance_indices.sum() > 0:
            downweighted_idx = np.where(high_variance_indices)[0]
            print(f"  Downweighted coefficient indices: {downweighted_idx.tolist()[:5]}... (showing first 5)")
    
    def apply_variance_weighting(self, X: np.ndarray) -> np.ndarray:
        """
        Apply variance weighting to data.
        
        Multiplies each feature by its weight, downweighting features that
        encode machine identity rather than fault state.
        
        Args:
            X: Data with shape (n_samples, n_features)
        
        Returns:
            Weighted data with shape (n_samples, n_features)
        
        Example:
            >>> X_weighted = preprocessor.apply_variance_weighting(X_test)
        """
        if self.variance_weights is None:
            raise ValueError("Variance weights not fitted. Call fit_variance_weighting first.")
        
        X_weighted = X * self.variance_weights
        
        return X_weighted
    
    # =========================================================================
    # COMBINED PIPELINE
    # =========================================================================
    
    def fit(self,
            X_train: np.ndarray,
            machine_ids_train: List[str],
            apply_both: bool = True,
            threshold_percentile: float = 75.0) -> None:
        """
        Fit both preprocessing modifications.
        
        Args:
            X_train: Training data with shape (n_samples, n_features)
            machine_ids_train: Machine IDs for each training sample
            apply_both: If True, fit both modifications; if False, fit normalization only
            threshold_percentile: Percentile for variance weighting threshold
        
        Example:
            >>> X_train = np.load('results/X_train_0db.npy')
            >>> machine_ids = ['id_00']*1011 + ['id_02']*1008
            >>> preprocessor = PreprocessingModifications()
            >>> preprocessor.fit(X_train, machine_ids)
        """
        # Modification 1: Per-machine-ID normalization
        self.fit_per_machine_normalization(X_train, machine_ids_train)
        
        # Modification 2: Variance-weighted selection
        if apply_both:
            self.fit_variance_weighting(X_train, machine_ids_train, threshold_percentile)
        
        self.is_fitted = True
        print("\n✓ Preprocessing modifications fitted successfully")
    
    def transform(self,
                 X: np.ndarray,
                 machine_ids: List[str] = None,
                 apply_normalization: bool = True,
                 apply_weighting: bool = True) -> np.ndarray:
        """
        Apply preprocessing modifications to data.
        
        Args:
            X: Data with shape (n_samples, n_features)
            machine_ids: Machine ID for each sample (required if apply_normalization=True)
                        Can be a single ID if all samples from same machine
            apply_normalization: Whether to apply per-machine-ID normalization
            apply_weighting: Whether to apply variance weighting
        
        Returns:
            Preprocessed data with shape (n_samples, n_features)
        
        Example:
            >>> X_test = np.load('results/X_test_0db.npy')
            >>> X_processed = preprocessor.transform(X_test, machine_ids='id_04')
        """
        if not self.is_fitted:
            raise ValueError("Preprocessing not fitted. Call fit() first.")
        
        X_processed = X.copy()
        
        # Apply normalization
        if apply_normalization:
            if machine_ids is None:
                raise ValueError("machine_ids required for normalization")
            
            # Handle case where machine_ids is a single string
            if isinstance(machine_ids, str):
                machine_ids = [machine_ids] * X.shape[0]
            
            # Apply per-machine normalization
            for machine_id in set(machine_ids):
                mask = np.array(machine_ids) == machine_id
                X_processed[mask] = self.apply_per_machine_normalization(
                    X_processed[mask], machine_id
                )
        
        # Apply weighting
        if apply_weighting:
            X_processed = self.apply_variance_weighting(X_processed)
        
        return X_processed
    
    def fit_transform(self,
                     X_train: np.ndarray,
                     machine_ids_train: List[str],
                     X_test: np.ndarray,
                     machine_ids_test: List[str] = None,
                     apply_both: bool = True,
                     apply_normalization: bool = True,
                     apply_weighting: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing on training data and transform both train and test.
        
        Args:
            X_train: Training data
            machine_ids_train: Machine IDs for training data
            X_test: Test data
            machine_ids_test: Machine IDs for test data (required if apply_normalization=True)
            apply_both: Whether to fit both modifications
            apply_normalization: Whether to apply normalization
            apply_weighting: Whether to apply weighting
        
        Returns:
            Tuple of (X_train_processed, X_test_processed)
        
        Example:
            >>> X_train_proc, X_test_proc = preprocessor.fit_transform(
            ...     X_train, machine_ids_train=['id_00']*1011 + ['id_02']*1008,
            ...     X_test, machine_ids_test=['id_04']*1010 + ['id_04']*109
            ... )
        """
        # Fit on training data
        self.fit(X_train, machine_ids_train, apply_both=apply_both)
        
        # Transform both
        X_train_processed = self.transform(
            X_train,
            machine_ids_train,
            apply_normalization=apply_normalization,
            apply_weighting=apply_weighting
        )
        
        X_test_processed = self.transform(
            X_test,
            machine_ids_test,
            apply_normalization=apply_normalization,
            apply_weighting=apply_weighting
        )
        
        return X_train_processed, X_test_processed
    
    # =========================================================================
    # SAVE/LOAD
    # =========================================================================
    
    def save(self, filepath: str) -> None:
        """
        Save fitted preprocessing parameters to disk.
        
        Args:
            filepath: Path to save file (.pkl)
        
        Example:
            >>> preprocessor.save('results/preprocessor.pkl')
        """
        state = {
            'machine_means': self.machine_means,
            'variance_weights': self.variance_weights,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ Preprocessor saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load fitted preprocessing parameters from disk.
        
        Args:
            filepath: Path to saved file (.pkl)
        
        Example:
            >>> preprocessor = PreprocessingModifications()
            >>> preprocessor.load('results/preprocessor.pkl')
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.machine_means = state['machine_means']
        self.variance_weights = state['variance_weights']
        self.is_fitted = state['is_fitted']
        
        print(f"✓ Preprocessor loaded from {filepath}")
    
    # =========================================================================
    # ANALYSIS & VISUALIZATION HELPERS
    # =========================================================================
    
    def get_summary(self) -> Dict:
        """
        Get summary of fitted preprocessing parameters.
        
        Returns:
            Dictionary with summary information
        
        Example:
            >>> summary = preprocessor.get_summary()
            >>> print(summary)
        """
        summary = {
            'is_fitted': self.is_fitted,
            'machine_ids': list(self.machine_means.keys()),
            'n_features': len(self.variance_weights) if self.variance_weights is not None else None,
            'n_downweighted': (self.variance_weights < 1.0).sum() if self.variance_weights is not None else None
        }
        return summary
    
    def print_summary(self) -> None:
        """Print human-readable summary of preprocessing"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        print(f"Fitted: {summary['is_fitted']}")
        print(f"Machine IDs: {summary['machine_ids']}")
        print(f"Features: {summary['n_features']}")
        print(f"Downweighted coefficients: {summary['n_downweighted']}/{summary['n_features']}")


# =============================================================================
# QUICK START EXAMPLE
# =============================================================================

def example_usage():
    """Example showing how to use PreprocessingModifications"""
    
    print("="*70)
    print("PREPROCESSING MODIFICATIONS - QUICK START EXAMPLE")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    X_train = np.load('results/X_train_0db.npy')
    X_test = np.load('results/X_test_0db.npy')
    y_test = np.load('results/y_test_0db.npy')
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    
    # Define machine IDs
    # Training: id_00 (1011 samples) + id_02 (1008 samples)
    # Test: id_04 (1010 normal + 109 abnormal)
    machine_ids_train = ['id_00']*1011 + ['id_02']*1008
    machine_ids_test = ['id_04'] * X_test.shape[0]
    
    # Create and fit preprocessor
    print("\n2. Creating preprocessor...")
    preprocessor = PreprocessingModifications()
    
    print("\n3. Fitting preprocessing modifications...")
    X_train_proc, X_test_proc = preprocessor.fit_transform(
        X_train, machine_ids_train,
        X_test, machine_ids_test,
        apply_both=True,
        apply_normalization=True,
        apply_weighting=True
    )
    
    # Summary
    print("\n4. Summary:")
    preprocessor.print_summary()
    
    # Save
    print("\n5. Saving preprocessor...")
    preprocessor.save('results/preprocessor.pkl')
    
    print("\nProcessed data ready for Phase 4 (Isolation Forest training)")
    return X_train_proc, X_test_proc, y_test


if __name__ == "__main__":
    # Uncomment to run example:
    # example_usage()
    
    print("Preprocessing module ready! Import and use PreprocessingModifications class.")
