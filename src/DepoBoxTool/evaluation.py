"""
Model evaluation metrics and statistical analysis tools.

This module provides comprehensive evaluation metrics for comparing
model predictions with observations, along with statistical analysis
and visualization tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from .utils import ensure_array, safe_divide


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    metric_name: str
    value: float
    description: str
    units: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def __str__(self) -> str:
        """String representation of the result."""
        ci_str = f" (CI: {self.confidence_interval})" if self.confidence_interval else ""
        units_str = f" {self.units}" if self.units else ""
        return f"{self.metric_name}: {self.value:.4f}{units_str}{ci_str} - {self.description}"


@dataclass
class EvaluationSummary:
    """Summary of multiple evaluation metrics."""
    
    results: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
    
    def get_result(self, metric_name: str) -> Optional[EvaluationResult]:
        """Get a specific result by metric name."""
        for result in self.results:
            if result.metric_name == metric_name:
                return result
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in self.results:
            data.append({
                'Metric': result.metric_name,
                'Value': result.value,
                'Description': result.description,
                'Units': result.units or '',
                'CI_Lower': result.confidence_interval[0] if result.confidence_interval else None,
                'CI_Upper': result.confidence_interval[1] if result.confidence_interval else None
            })
        return pd.DataFrame(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            'results': [
                {
                    'metric': r.metric_name,
                    'value': r.value,
                    'description': r.description,
                    'units': r.units,
                    'confidence_interval': r.confidence_interval
                }
                for r in self.results
            ],
            'metadata': self.metadata
        }


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for model performance assessment.
    
    This class provides a wide range of statistical metrics commonly used
    in atmospheric model evaluation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    def _prepare_data(self, 
                     observed: Union[np.ndarray, List, pd.Series],
                     predicted: Union[np.ndarray, List, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and validate input data.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            Tuple of (observed, predicted) as numpy arrays with matched masks
        """
        obs = ensure_array(observed)
        pred = ensure_array(predicted)
        
        if len(obs) != len(pred):
            raise ValueError(f"Length mismatch: observed={len(obs)}, predicted={len(pred)}")
        
        # Create masked arrays to handle NaN/invalid values
        obs_ma = np.ma.masked_invalid(obs)
        pred_ma = np.ma.masked_invalid(pred)
        
        # Apply common mask
        common_mask = obs_ma.mask | pred_ma.mask
        obs_ma.mask = common_mask
        pred_ma.mask = common_mask
        
        if obs_ma.count() == 0:
            raise ValueError("No valid data pairs found")
        
        return obs_ma, pred_ma
    
    def basic_statistics(self, 
                        observed: Union[np.ndarray, List],
                        predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Calculate basic statistical metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with basic statistics
        """
        obs, pred = self._prepare_data(observed, predicted)
        summary = EvaluationSummary()
        
        # Number of observations
        summary.add_result(EvaluationResult(
            'N', obs.count(), 'Number of valid observations', '#'
        ))
        
        # Mean values
        summary.add_result(EvaluationResult(
            'Mean_Obs', np.ma.mean(obs), 'Mean of observations'
        ))
        summary.add_result(EvaluationResult(
            'Mean_Pred', np.ma.mean(pred), 'Mean of predictions'
        ))
        
        # Standard deviations
        summary.add_result(EvaluationResult(
            'Std_Obs', np.ma.std(obs), 'Standard deviation of observations'
        ))
        summary.add_result(EvaluationResult(
            'Std_Pred', np.ma.std(pred), 'Standard deviation of predictions'
        ))
        
        # Median values
        summary.add_result(EvaluationResult(
            'Median_Obs', np.ma.median(obs), 'Median of observations'
        ))
        summary.add_result(EvaluationResult(
            'Median_Pred', np.ma.median(pred), 'Median of predictions'
        ))
        
        return summary
    
    def bias_metrics(self, 
                    observed: Union[np.ndarray, List],
                    predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Calculate bias-related metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with bias metrics
        """
        obs, pred = self._prepare_data(observed, predicted)
        summary = EvaluationSummary()
        
        # Mean Bias (MB)
        mb = np.ma.mean(pred - obs)
        summary.add_result(EvaluationResult(
            'MB', mb, 'Mean Bias'
        ))
        
        # Normalized Mean Bias (NMB)
        nmb = np.ma.sum(pred - obs) / np.ma.sum(obs) * 100
        summary.add_result(EvaluationResult(
            'NMB', nmb, 'Normalized Mean Bias', '%'
        ))
        
        # Mean Normalized Bias (MNB)
        mnb = np.ma.mean((pred - obs) / obs) * 100
        summary.add_result(EvaluationResult(
            'MNB', mnb, 'Mean Normalized Bias', '%'
        ))
        
        # Median Bias
        median_bias = np.ma.median(pred - obs)
        summary.add_result(EvaluationResult(
            'MdnB', median_bias, 'Median Bias'
        ))
        
        # Median Normalized Bias
        mdn_norm_bias = np.ma.median((pred - obs) / obs) * 100
        summary.add_result(EvaluationResult(
            'MdnNB', mdn_norm_bias, 'Median Normalized Bias', '%'
        ))
        
        return summary
    
    def error_metrics(self, 
                     observed: Union[np.ndarray, List],
                     predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Calculate error-related metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with error metrics
        """
        obs, pred = self._prepare_data(observed, predicted)
        summary = EvaluationSummary()
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.ma.mean((pred - obs) ** 2))
        summary.add_result(EvaluationResult(
            'RMSE', rmse, 'Root Mean Square Error'
        ))
        
        # Mean Absolute Error (MAE)
        mae = np.ma.mean(np.ma.abs(pred - obs))
        summary.add_result(EvaluationResult(
            'MAE', mae, 'Mean Absolute Error'
        ))
        
        # Mean Normalized Gross Error (MNGE)
        mnge = np.ma.mean(np.ma.abs(pred - obs) / obs) * 100
        summary.add_result(EvaluationResult(
            'MNGE', mnge, 'Mean Normalized Gross Error', '%'
        ))
        
        # Median Normalized Gross Error
        mdnge = np.ma.median(np.ma.abs(pred - obs) / obs) * 100
        summary.add_result(EvaluationResult(
            'MdnNGE', mdnge, 'Median Normalized Gross Error', '%'
        ))
        
        return summary
    
    def correlation_metrics(self, 
                           observed: Union[np.ndarray, List],
                           predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Calculate correlation-related metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with correlation metrics
        """
        obs, pred = self._prepare_data(observed, predicted)
        summary = EvaluationSummary()
        
        # Remove masked values for correlation calculation
        obs_clean = obs.compressed()
        pred_clean = pred.compressed()
        
        # Pearson correlation coefficient
        r, p_value = stats.pearsonr(obs_clean, pred_clean)
        summary.add_result(EvaluationResult(
            'R', r, 'Pearson correlation coefficient'
        ))
        
        # Coefficient of determination
        r2 = r ** 2
        summary.add_result(EvaluationResult(
            'R2', r2, 'Coefficient of determination'
        ))
        
        # Spearman rank correlation
        rho, _ = stats.spearmanr(obs_clean, pred_clean)
        summary.add_result(EvaluationResult(
            'Spearman', rho, 'Spearman rank correlation coefficient'
        ))
        
        return summary
    
    def fractional_metrics(self, 
                          observed: Union[np.ndarray, List],
                          predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Calculate fractional bias and error metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with fractional metrics
        """
        obs, pred = self._prepare_data(observed, predicted)
        summary = EvaluationSummary()
        
        # Fractional Bias (FB)
        fb = 2 * np.ma.mean(pred - obs) / np.ma.mean(pred + obs)
        summary.add_result(EvaluationResult(
            'FB', fb, 'Fractional Bias'
        ))
        
        # Fractional Error (FE)
        fe = 2 * np.ma.mean(np.ma.abs(pred - obs)) / np.ma.mean(pred + obs)
        summary.add_result(EvaluationResult(
            'FE', fe, 'Fractional Error'
        ))
        
        return summary
    
    def skill_metrics(self, 
                     observed: Union[np.ndarray, List],
                     predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Calculate model skill metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with skill metrics
        """
        obs, pred = self._prepare_data(observed, predicted)
        summary = EvaluationSummary()
        
        # Index of Agreement (IOA)
        obs_mean = np.ma.mean(obs)
        numerator = np.ma.sum((pred - obs) ** 2)
        denominator = np.ma.sum((np.ma.abs(pred - obs_mean) + np.ma.abs(obs - obs_mean)) ** 2)
        ioa = 1 - numerator / denominator
        summary.add_result(EvaluationResult(
            'IOA', ioa, 'Index of Agreement'
        ))
        
        # Nash-Sutcliffe Efficiency
        nse = 1 - np.ma.sum((obs - pred) ** 2) / np.ma.sum((obs - obs_mean) ** 2)
        summary.add_result(EvaluationResult(
            'NSE', nse, 'Nash-Sutcliffe Efficiency'
        ))
        
        return summary
    
    def comprehensive_evaluation(self, 
                                observed: Union[np.ndarray, List],
                                predicted: Union[np.ndarray, List]) -> EvaluationSummary:
        """
        Perform comprehensive model evaluation with all metrics.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            EvaluationSummary with all metrics
        """
        self.logger.info("Performing comprehensive model evaluation")
        
        # Combine all metric types
        summaries = [
            self.basic_statistics(observed, predicted),
            self.bias_metrics(observed, predicted),
            self.error_metrics(observed, predicted),
            self.correlation_metrics(observed, predicted),
            self.fractional_metrics(observed, predicted),
            self.skill_metrics(observed, predicted)
        ]
        
        # Merge all summaries
        combined = EvaluationSummary()
        for summary in summaries:
            combined.results.extend(summary.results)
        
        # Add metadata
        obs, pred = self._prepare_data(observed, predicted)
        combined.metadata = {
            'n_observations': obs.count(),
            'obs_range': (float(np.ma.min(obs)), float(np.ma.max(obs))),
            'pred_range': (float(np.ma.min(pred)), float(np.ma.max(pred)))
        }
        
        return combined


class ModelComparison:
    """
    Tools for comparing multiple models against observations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self.evaluator = EvaluationMetrics()
    
    def compare_models(self, 
                      observed: Union[np.ndarray, List],
                      predictions: Dict[str, Union[np.ndarray, List]],
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models using specified metrics.
        
        Args:
            observed: Observed values
            predictions: Dictionary of {model_name: predicted_values}
            metrics: List of metric names to calculate
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['R2', 'RMSE', 'NMB', 'FB', 'IOA']
        
        results = []
        
        for model_name, pred in predictions.items():
            self.logger.debug(f"Evaluating model: {model_name}")
            
            summary = self.evaluator.comprehensive_evaluation(observed, pred)
            
            row = {'Model': model_name}
            for metric in metrics:
                result = summary.get_result(metric)
                row[metric] = result.value if result else np.nan
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def rank_models(self, 
                   observed: Union[np.ndarray, List],
                   predictions: Dict[str, Union[np.ndarray, List]],
                   primary_metric: str = 'R2',
                   ascending: bool = False) -> pd.DataFrame:
        """
        Rank models based on a primary metric.
        
        Args:
            observed: Observed values
            predictions: Dictionary of predictions
            primary_metric: Metric to use for ranking
            ascending: Whether to rank in ascending order
            
        Returns:
            DataFrame with ranked models
        """
        comparison = self.compare_models(observed, predictions)
        
        if primary_metric not in comparison.columns:
            raise ValueError(f"Metric '{primary_metric}' not found in comparison results")
        
        ranked = comparison.sort_values(primary_metric, ascending=ascending)
        ranked['Rank'] = range(1, len(ranked) + 1)
        
        return ranked[['Rank', 'Model'] + [col for col in ranked.columns if col not in ['Rank', 'Model']]]


class VisualizationTools:
    """
    Visualization tools for model evaluation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    def scatter_plot(self, 
                    observed: Union[np.ndarray, List],
                    predicted: Union[np.ndarray, List],
                    title: str = "Model vs Observations",
                    xlabel: str = "Observed",
                    ylabel: str = "Predicted",
                    log_scale: bool = True,
                    save_path: Optional[str] = None,
                    show_stats: bool = True) -> plt.Figure:
        """
        Create scatter plot of predicted vs observed values.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale
            save_path: Path to save the plot
            show_stats: Whether to show statistics on plot
            
        Returns:
            Matplotlib figure object
        """
        obs = ensure_array(observed)
        pred = ensure_array(predicted)
        
        # Remove invalid values
        valid_mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
        obs_clean = obs[valid_mask]
        pred_clean = pred[valid_mask]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(obs_clean, pred_clean, alpha=0.6, s=50)
        
        # 1:1 line
        min_val = min(np.min(obs_clean), np.min(pred_clean))
        max_val = max(np.max(obs_clean), np.max(pred_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        
        # 2:1 and 1:2 lines
        ax.plot([min_val, max_val], [2*min_val, 2*max_val], 'g--', alpha=0.5, label='2:1 line')
        ax.plot([2*min_val, 2*max_val], [min_val, max_val], 'g--', alpha=0.5, label='1:2 line')
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if show_stats:
            evaluator = EvaluationMetrics()
            summary = evaluator.comprehensive_evaluation(obs_clean, pred_clean)
            
            stats_text = []
            for metric in ['R2', 'RMSE', 'NMB', 'FB']:
                result = summary.get_result(metric)
                if result:
                    stats_text.append(f"{metric}: {result.value:.3f}")
            
            ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Scatter plot saved to {save_path}")
        
        return fig
    
    def residual_plot(self, 
                     observed: Union[np.ndarray, List],
                     predicted: Union[np.ndarray, List],
                     title: str = "Residual Plot",
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create residual plot for model evaluation.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        obs = ensure_array(observed)
        pred = ensure_array(predicted)
        
        # Calculate residuals
        residuals = pred - obs
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs predicted
        ax1.scatter(pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Residual plot saved to {save_path}")
        
        return fig


# Legacy function compatibility
def MB(obs, mod, axis=None):
    """Mean Bias - Legacy function."""
    import warnings
    warnings.warn("Legacy function. Use EvaluationMetrics.bias_metrics instead.", DeprecationWarning)
    return np.ma.mean(mod - obs, axis=axis)

def NMB(obs, mod, axis=None):
    """Normalized Mean Bias - Legacy function."""
    import warnings
    warnings.warn("Legacy function. Use EvaluationMetrics.bias_metrics instead.", DeprecationWarning)
    return np.ma.sum(mod - obs, axis=axis) / np.ma.sum(obs, axis=axis) * 100

def FB(obs, mod, axis=None):
    """Fractional Bias - Legacy function."""
    import warnings
    warnings.warn("Legacy function. Use EvaluationMetrics.fractional_metrics instead.", DeprecationWarning)
    return 2 * np.ma.mean(mod - obs, axis=axis) / np.ma.mean(mod + obs, axis=axis)

def FE(obs, mod, axis=None):
    """Fractional Error - Legacy function."""
    import warnings
    warnings.warn("Legacy function. Use EvaluationMetrics.fractional_metrics instead.", DeprecationWarning)
    return 2 * np.ma.mean(np.ma.abs(mod - obs), axis=axis) / np.ma.mean(mod + obs, axis=axis)

def R2(obs, mod, axis=None):
    """Coefficient of determination - Legacy function."""
    import warnings
    warnings.warn("Legacy function. Use EvaluationMetrics.correlation_metrics instead.", DeprecationWarning)
    obs_clean = obs.compressed() if hasattr(obs, 'compressed') else obs
    mod_clean = mod.compressed() if hasattr(mod, 'compressed') else mod
    return stats.pearsonr(obs_clean, mod_clean)[0] ** 2

def RMSE(obs, mod, axis=None):
    """Root Mean Square Error - Legacy function."""
    import warnings
    warnings.warn("Legacy function. Use EvaluationMetrics.error_metrics instead.", DeprecationWarning)
    return np.sqrt(np.ma.mean((mod - obs) ** 2, axis=axis))