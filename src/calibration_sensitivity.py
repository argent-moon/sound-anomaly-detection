"""
Calibration Size Sensitivity Analysis

This module tests how much adaptation data (normal clips from target machine)
is actually needed for the per-machine-ID normalization to work effectively.

Tests: 10, 50, 100, 500, 1000 normal clips
Evaluates on: Full test set (all normal + abnormal clips)
Metric: AUC-ROC
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, 'C:/Users/Letizia/Documents/sound-anomaly-detection/src')
from preprocessing import PreprocessingModifications
from audio_loader import AudioLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


class CalibrationSensitivityAnalysis:
    """
    Test how many normal clips from target machine are needed
    for effective per-machine-ID normalization.
    """
    
    def __init__(self):
        self.results = {}
        
    def run_analysis(self,
                     X_train: np.ndarray,
                     machine_ids_train: List[str],
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     test_machine_normal_features: np.ndarray,
                     test_id: str,
                     calibration_sizes: List[int] = None,
                     n_seeds: int = 10,
                     use_modified: bool = True) -> Dict:
        """
        Test different calibration sizes with proper feature space handling.
        
        CRITICAL FIX: Train and score in SAME feature space (both modified or both original)
        
        Args:
            X_train: Training features (id_00 + id_02 normal sounds)
            machine_ids_train: Machine IDs for training samples
            X_test: Test features (all clips from unseen machine)
            y_test: Test labels (0=normal, 1=abnormal)
            test_machine_normal_features: All normal clips from test machine
            test_id: Test machine ID (e.g., 'id_04')
            calibration_sizes: Sizes to test (default: auto-generated from actual data)
                              Includes [10, 50, 100, 500] + actual dataset max
            n_seeds: Number of random seeds for averaging (default: 10, matches Phase 4-5)
            use_modified: If True, applies preprocessing modifications
        
        Returns:
            Dictionary with results:
            {
                'sizes': [10, 50, 100, 500, 1033],
                'auc_means': [0.7234, 0.7891, 0.8123, 0.8456, 0.8847],
                'auc_stds': [0.0234, 0.0167, 0.0145, 0.0089, 0.0067],
                'all_aucs': {10: [0.71, 0.73, ...], 50: [...], ...},
                'use_modified': True
            }
        """
        if calibration_sizes is None:
            # Test sizes: DCASE range (10), progression, plus actual dataset max
            max_available = test_machine_normal_features.shape[0]
            calibration_sizes = [3, 10, 50, 100, 500, max_available]
        
        # Cap at available data
        max_available = test_machine_normal_features.shape[0]
        calibration_sizes = [min(size, max_available) for size in calibration_sizes]
        calibration_sizes = sorted(list(set(calibration_sizes)))  # Remove duplicates, sort
        
        print(f"\n{'='*70}")
        print(f"CALIBRATION SENSITIVITY ANALYSIS: {test_id}")
        print(f"{'='*70}")
        print(f"Available normal clips for calibration: {max_available}")
        print(f"Testing calibration sizes: {calibration_sizes}")
        print(f"Seeds per size: {n_seeds}")
        
        auc_means = []
        auc_stds = []
        all_aucs_per_size = {}
        
        for calib_size in calibration_sizes:
            print(f"\n  Calibration size: {calib_size}")
            
            # Randomly sample calibration clips
            # Use different random seed for each calibration size to ensure fair sampling
            rng = np.random.RandomState(42)
            
            aucs_for_this_size = []
            
            # Run multiple seeds with same calibration size
            for seed in range(n_seeds):
                # Sample calibration clips
                calib_indices = rng.choice(
                    test_machine_normal_features.shape[0],
                    size=calib_size,
                    replace=False
                )
                test_machine_mean = test_machine_normal_features[calib_indices].mean(axis=0)
                
                # Fit preprocessor on training data
                preprocessor = PreprocessingModifications()
                preprocessor.fit(X_train, machine_ids_train, apply_both=True)
                
                # Add calibration mean
                preprocessor.machine_means[test_id] = test_machine_mean
                
                # CRITICAL FIX: Apply same transformation to both train and test
                machine_ids_test = [test_id] * X_test.shape[0]
                
                if use_modified:
                    # Apply preprocessing modifications to BOTH train and test
                    X_train_for_fit = preprocessor.transform(
                        X_train,
                        machine_ids_train,
                        apply_normalization=True,
                        apply_weighting=True
                    )
                    X_test_for_eval = preprocessor.transform(
                        X_test,
                        machine_ids_test,
                        apply_normalization=True,
                        apply_weighting=True
                    )
                else:
                    # Use original (unmodified) features
                    X_train_for_fit = X_train
                    X_test_for_eval = X_test
                
                # Train and evaluate in SAME feature space
                clf = IsolationForest(
                    n_estimators=100,
                    max_features=0.8,
                    contamination='auto',
                    random_state=seed,
                    n_jobs=-1
                )
                clf.fit(X_train_for_fit)  # Train on correct feature space
                scores = -clf.score_samples(X_test_for_eval)  # Score on same feature space
                auc = roc_auc_score(y_test, scores)
                aucs_for_this_size.append(auc)
            
            # Statistics for this size
            auc_mean = np.mean(aucs_for_this_size)
            auc_std = np.std(aucs_for_this_size)
            
            auc_means.append(auc_mean)
            auc_stds.append(auc_std)
            all_aucs_per_size[calib_size] = aucs_for_this_size
            
            print(f"    AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        
        results = {
            'test_id': test_id,
            'sizes': calibration_sizes,
            'auc_means': auc_means,
            'auc_stds': auc_stds,
            'all_aucs': all_aucs_per_size,
            'use_modified': use_modified,
            'max_available': max_available
        }
        
        self.results[test_id] = results
        return results
    
    def plot_results(self, results_dict: Dict = None, figsize: Tuple = (12, 6)):
        """
        Plot AUC vs. calibration size.
        
        Args:
            results_dict: Results from run_analysis (or uses self.results)
            figsize: Figure size
        """
        if results_dict is None:
            results_dict = self.results
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for test_id, results in results_dict.items():
            sizes = results['sizes']
            auc_means = results['auc_means']
            auc_stds = results['auc_stds']
            
            # Plot with error bars
            ax.errorbar(
                sizes, auc_means, yerr=auc_stds,
                marker='o', markersize=8, linewidth=2,
                capsize=5, capthick=2,
                label=f'{test_id} (n={len(results["all_aucs"])})',
                alpha=0.8
            )
            
            # Add value labels
            for size, auc, std in zip(sizes, auc_means, auc_stds):
                ax.text(size, auc + std + 0.01, f'{auc:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Calibration Size (# of normal clips)', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('Calibration Size Sensitivity Analysis\n(How many normal clips needed for effective normalization?)',
                    fontsize=14)
        ax.set_xscale('log')
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self, results_dict: Dict = None):
        """
        Print summary of calibration sensitivity results.
        """
        if results_dict is None:
            results_dict = self.results
        
        print(f"\n{'='*70}")
        print("CALIBRATION SENSITIVITY ANALYSIS - SUMMARY")
        print(f"{'='*70}")
        
        for test_id, results in results_dict.items():
            print(f"\n{test_id}:")
            print(f"  {'Size':<10} {'AUC Mean':<12} {'AUC Std':<12}")
            print(f"  {'-'*34}")
            
            for size, auc_mean, auc_std in zip(
                results['sizes'],
                results['auc_means'],
                results['auc_stds']
            ):
                print(f"  {size:<10} {auc_mean:<12.4f} {auc_std:<12.4f}")
            
            # Improvement from smallest to largest
            improvement = results['auc_means'][-1] - results['auc_means'][0]
            pct_improvement = (improvement / results['auc_means'][0]) * 100
            print(f"\n  Improvement ({results['sizes'][0]}→{results['sizes'][-1]}): "
                  f"{improvement:+.4f} ({pct_improvement:+.1f}%)")
            
            # Saturation point (where improvement becomes small)
            diffs = np.diff(results['auc_means'])
            saturation_idx = np.argmax(diffs < 0.01)  # Less than 0.01 improvement
            if saturation_idx > 0:
                saturation_size = results['sizes'][saturation_idx]
                print(f"  Saturation point: ~{saturation_size} clips")
                print(f"    (Improvement < 0.01 AUC beyond this)")


# =============================================================================
# QUICK START EXAMPLE
# =============================================================================

def example_single_config():
    """Example: Test calibration sensitivity for fan, 0_dB, id_04"""
    
    from pathlib import Path
    import pickle
    
    results_dir = Path("results")
    
    print("Example: Calibration Sensitivity Analysis")
    print("Machine: fan | Condition: 0_dB | Test ID: id_04")
    
    # Load data
    X_train = np.load(results_dir / "X_train_0dB_fan.npy")
    X_test = np.load(results_dir / "X_test_0dB_fan_id_04.npy")
    y_test = np.load(results_dir / "y_test_0dB_fan_id_04.npy")
    
    with open(results_dir / "machine_ids_train_0dB_fan.pkl", 'rb') as f:
        machine_ids_train = pickle.load(f)
    
    # Load test machine's normal clips
    loader = AudioLoader(data_root="./data", sr=16000, n_mfcc=13, machine_type='fan')
    data_test = loader.load_condition_dataset(
        condition='0_dB',
        machine_ids=['id_04'],
        aggregate_method='mean'
    )
    test_normal_features = np.array(data_test['machine_ids']['id_04']['normal']['features'])
    
    # Run analysis
    analyzer = CalibrationSensitivityAnalysis()
    results = analyzer.run_analysis(
        X_train, machine_ids_train, X_test, y_test,
        test_normal_features, 'id_04',
        calibration_sizes=[3, 10, 50, 100, 500, 1000],
        n_seeds=5
    )
    
    # Visualize
    fig = analyzer.plot_results({('id_04'): results})
    plt.savefig(results_dir / 'calibration_sensitivity_example.png', dpi=100, bbox_inches='tight')
    
    # Summary
    analyzer.print_summary({'id_04': results})
    
    return results


if __name__ == "__main__":
    print("Calibration Sensitivity Analysis Module")
    print("\nImport and use CalibrationSensitivityAnalysis class")
    print("\nExample usage:")
    print("  analyzer = CalibrationSensitivityAnalysis()")
    print("  results = analyzer.run_analysis(X_train, machine_ids_train, X_test, y_test, ...)")
    print("  fig = analyzer.plot_results()")
