"""
Evaluation metrics for model performance assessment.

This module provides statistical metrics commonly used in atmospheric
model evaluation and validation studies.
"""

import numpy as np
from typing import Union, Optional
from scipy.stats import pearsonr, linregress


def _ensure_arrays(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray]) -> tuple:
    """
    Ensure inputs are numpy masked arrays and handle missing values.
    
    Args:
        obs: Observed values
        mod: Modeled values
        
    Returns:
        Tuple of matched masked arrays
    """
    obs = np.ma.array(obs)
    mod = np.ma.array(mod)
    
    # Create common mask for both arrays
    mask = np.ma.getmaskarray(obs) | np.ma.getmaskarray(mod)
    obs_masked = np.ma.masked_where(mask, obs)
    mod_masked = np.ma.masked_where(mask, mod)
    
    return obs_masked, mod_masked


class BasicMetrics:
    """Basic statistical metrics for model evaluation."""
    
    @staticmethod
    def mean_bias(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray], 
                  axis: Optional[int] = None) -> float:
        """
        Calculate mean bias (MB).
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Mean bias
        """
        obs, mod = _ensure_arrays(obs, mod)
        return (mod - obs).mean(axis=axis)
    
    @staticmethod
    def mean_absolute_error(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                           axis: Optional[int] = None) -> float:
        """
        Calculate mean absolute error (MAE).
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Mean absolute error
        """
        obs, mod = _ensure_arrays(obs, mod)
        return np.ma.abs(mod - obs).mean(axis=axis)
    
    @staticmethod
    def root_mean_square_error(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                              axis: Optional[int] = None) -> float:
        """
        Calculate root mean square error (RMSE).
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Root mean square error
        """
        obs, mod = _ensure_arrays(obs, mod)
        return np.ma.sqrt(((mod - obs) ** 2).mean(axis=axis))
    
    @staticmethod
    def coefficient_of_determination(obs: Union[list, np.ndarray], 
                                   mod: Union[list, np.ndarray]) -> float:
        """
        Calculate coefficient of determination (R²).
        
        Args:
            obs: Observed values
            mod: Modeled values
            
        Returns:
            Coefficient of determination
        """
        obs, mod = _ensure_arrays(obs, mod)
        obs_clean = obs.compressed()
        mod_clean = mod.compressed()
        
        if len(obs_clean) < 2:
            return np.nan
            
        correlation, _ = pearsonr(obs_clean, mod_clean)
        return correlation ** 2


class NormalizedMetrics:
    """Normalized metrics for model evaluation."""
    
    @staticmethod
    def normalized_mean_bias(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                           axis: Optional[int] = None) -> float:
        """
        Calculate normalized mean bias (NMB) as percentage.
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Normalized mean bias in percent
        """
        obs, mod = _ensure_arrays(obs, mod)
        return ((mod - obs).sum(axis=axis) / obs.sum(axis=axis)) * 100
    
    @staticmethod
    def normalized_mean_error(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                            axis: Optional[int] = None) -> float:
        """
        Calculate normalized mean error (NME) as percentage.
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Normalized mean error in percent
        """
        obs, mod = _ensure_arrays(obs, mod)
        return (np.ma.abs(mod - obs).sum(axis=axis) / obs.sum(axis=axis)) * 100
    
    @staticmethod
    def mean_normalized_bias(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                           axis: Optional[int] = None) -> float:
        """
        Calculate mean normalized bias (MNB) as percentage.
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Mean normalized bias in percent
        """
        obs, mod = _ensure_arrays(obs, mod)
        return (np.ma.masked_invalid((mod - obs) / obs).mean(axis=axis)) * 100
    
    @staticmethod
    def mean_normalized_error(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                            axis: Optional[int] = None) -> float:
        """
        Calculate mean normalized error (MNE) as percentage.
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Mean normalized error in percent
        """
        obs, mod = _ensure_arrays(obs, mod)
        normalized_error = np.ma.abs(mod - obs) / obs
        return (np.ma.masked_invalid(normalized_error).mean(axis=axis)) * 100


class FractionalMetrics:
    """Fractional metrics for model evaluation."""
    
    @staticmethod
    def fractional_bias(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                       axis: Optional[int] = None) -> float:
        """
        Calculate fractional bias (FB) as percentage.
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Fractional bias in percent
        """
        obs, mod = _ensure_arrays(obs, mod)
        fractional_diff = (mod - obs) / (mod + obs)
        return (np.ma.masked_invalid(fractional_diff).mean(axis=axis) * 2) * 100
    
    @staticmethod
    def fractional_error(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                        axis: Optional[int] = None) -> float:
        """
        Calculate fractional error (FE) as percentage.
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Fractional error in percent
        """
        obs, mod = _ensure_arrays(obs, mod)
        fractional_diff = np.ma.abs(mod - obs) / (mod + obs)
        return (fractional_diff.mean(axis=axis) * 2) * 100


class IndexMetrics:
    """Index-based metrics for model evaluation."""
    
    @staticmethod
    def index_of_agreement(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                         axis: Optional[int] = None) -> float:
        """
        Calculate index of agreement (IOA).
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Index of agreement (0 to 1)
        """
        obs, mod = _ensure_arrays(obs, mod)
        obs_mean = obs.mean(axis=axis)
        
        if axis is not None:
            obs_mean = np.expand_dims(obs_mean, axis=axis)
        
        numerator = ((obs - mod) ** 2).sum(axis=axis)
        denominator = ((np.ma.abs(mod - obs_mean) + np.ma.abs(obs - obs_mean)) ** 2).sum(axis=axis)
        
        return 1.0 - (numerator / denominator)
    
    @staticmethod
    def modified_index_of_agreement(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                                  axis: Optional[int] = None) -> float:
        """
        Calculate modified index of agreement (d1).
        
        Args:
            obs: Observed values
            mod: Modeled values
            axis: Axis along which to compute (None for all)
            
        Returns:
            Modified index of agreement
        """
        obs, mod = _ensure_arrays(obs, mod)
        obs_mean = obs.mean(axis=axis)
        
        numerator = np.ma.abs(obs - mod).sum(axis=axis)
        denominator = (np.ma.abs(mod - obs_mean) + np.ma.abs(obs - obs_mean)).sum(axis=axis)
        
        return 1.0 - (numerator / denominator)


class DescriptiveStatistics:
    """Descriptive statistics for model evaluation."""
    
    @staticmethod
    def mean_observations(obs: Union[list, np.ndarray], 
                         axis: Optional[int] = None) -> float:
        """Calculate mean of observations."""
        obs = np.ma.array(obs)
        return obs.mean(axis=axis)
    
    @staticmethod
    def mean_predictions(mod: Union[list, np.ndarray], 
                        axis: Optional[int] = None) -> float:
        """Calculate mean of predictions."""
        mod = np.ma.array(mod)
        return mod.mean(axis=axis)
    
    @staticmethod
    def std_observations(obs: Union[list, np.ndarray], 
                        axis: Optional[int] = None) -> float:
        """Calculate standard deviation of observations."""
        obs = np.ma.array(obs)
        return obs.std(axis=axis)
    
    @staticmethod
    def std_predictions(mod: Union[list, np.ndarray], 
                       axis: Optional[int] = None) -> float:
        """Calculate standard deviation of predictions."""
        mod = np.ma.array(mod)
        return mod.std(axis=axis)
    
    @staticmethod
    def count_pairs(obs: Union[list, np.ndarray], mod: Union[list, np.ndarray],
                   axis: Optional[int] = None) -> int:
        """Count valid observation-prediction pairs."""
        obs, mod = _ensure_arrays(obs, mod)
        return (~np.ma.getmaskarray(obs)).sum(axis=axis)


class ModelEvaluator:
    """Main class for comprehensive model evaluation."""
    
    def __init__(self, observations: Union[list, np.ndarray], 
                 predictions: Union[list, np.ndarray]):
        """
        Initialize model evaluator.
        
        Args:
            observations: Observed values
            predictions: Model predictions
        """
        self.obs, self.mod = _ensure_arrays(observations, predictions)
        
        # Initialize metric calculators
        self.basic = BasicMetrics()
        self.normalized = NormalizedMetrics()
        self.fractional = FractionalMetrics()
        self.index = IndexMetrics()
        self.stats = DescriptiveStatistics()
    
    def calculate_all_metrics(self) -> dict:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        return {
            # Basic metrics
            'mean_bias': self.basic.mean_bias(self.obs, self.mod),
            'mean_absolute_error': self.basic.mean_absolute_error(self.obs, self.mod),
            'root_mean_square_error': self.basic.root_mean_square_error(self.obs, self.mod),
            'r_squared': self.basic.coefficient_of_determination(self.obs, self.mod),
            
            # Normalized metrics
            'normalized_mean_bias': self.normalized.normalized_mean_bias(self.obs, self.mod),
            'normalized_mean_error': self.normalized.normalized_mean_error(self.obs, self.mod),
            'mean_normalized_bias': self.normalized.mean_normalized_bias(self.obs, self.mod),
            'mean_normalized_error': self.normalized.mean_normalized_error(self.obs, self.mod),
            
            # Fractional metrics
            'fractional_bias': self.fractional.fractional_bias(self.obs, self.mod),
            'fractional_error': self.fractional.fractional_error(self.obs, self.mod),
            
            # Index metrics
            'index_of_agreement': self.index.index_of_agreement(self.obs, self.mod),
            'modified_index_of_agreement': self.index.modified_index_of_agreement(self.obs, self.mod),
            
            # Descriptive statistics
            'mean_observations': self.stats.mean_observations(self.obs),
            'mean_predictions': self.stats.mean_predictions(self.mod),
            'std_observations': self.stats.std_observations(self.obs),
            'std_predictions': self.stats.std_predictions(self.mod),
            'count_pairs': self.stats.count_pairs(self.obs, self.mod)
        }


# Legacy function aliases for backward compatibility
def MB(obs, mod, axis=None):
    """Mean Bias (legacy function)."""
    return BasicMetrics.mean_bias(obs, mod, axis)

def ME(obs, mod, axis=None):
    """Mean Error (legacy function)."""
    return BasicMetrics.mean_absolute_error(obs, mod, axis)

def RMSE(obs, mod, axis=None):
    """Root Mean Square Error (legacy function)."""
    return BasicMetrics.root_mean_square_error(obs, mod, axis)

def R2(obs, mod, axis=None):
    """Coefficient of Determination (legacy function)."""
    return BasicMetrics.coefficient_of_determination(obs, mod)

def NMB(obs, mod, axis=None):
    """Normalized Mean Bias (legacy function)."""
    return NormalizedMetrics.normalized_mean_bias(obs, mod, axis)

def NME(obs, mod, axis=None):
    """Normalized Mean Error (legacy function)."""
    return NormalizedMetrics.normalized_mean_error(obs, mod, axis)

def FB(obs, mod, axis=None):
    """Fractional Bias (legacy function)."""
    return FractionalMetrics.fractional_bias(obs, mod, axis)

def IOA(obs, mod, axis=None):
    """Index of Agreement (legacy function)."""
    return IndexMetrics.index_of_agreement(obs, mod, axis)