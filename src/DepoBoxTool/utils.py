"""
Utility functions for DepoBoxTool.

This module provides common utility functions including logging setup,
input validation, and helper functions used across the package.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the package.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('DepoBoxTool')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_inputs(**kwargs) -> Dict[str, Any]:
    """
    Validate input parameters for deposition models.
    
    Args:
        **kwargs: Parameter name-value pairs to validate
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}
    
    # Physical parameter ranges
    validation_rules = {
        'dens': (100, 10000, 'Particle density (kg/m³)'),
        'dim': (1e-9, 1e-3, 'Particle diameter (m)'),
        'temp': (200, 350, 'Temperature (K)'),
        'press': (50000, 150000, 'Pressure (Pa)'),
        'RH': (0, 1, 'Relative humidity (fraction)'),
        'ustar': (0.001, 10, 'Friction velocity (m/s)'),
        'Lo': (-1000, 1000, 'Monin-Obukhov length (m)'),
        'z': (0, 100, 'Measurement height (m)'),
        'z0': (1e-6, 10, 'Roughness length (m)'),
        'LAI': (0, 20, 'Leaf Area Index'),
        'sigmag': (1.0, 5.0, 'Geometric standard deviation'),
        'A': (1e-6, 1e-2, 'Characteristic length scale (m)')
    }
    
    for param, value in kwargs.items():
        if param in validation_rules:
            min_val, max_val, description = validation_rules[param]
            
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{description} must be numeric, got {type(value)}")
            
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{description} must be between {min_val} and {max_val}, "
                    f"got {value}"
                )
        
        validated[param] = value
    
    return validated


def ensure_array(value: Union[float, int, List, np.ndarray]) -> np.ndarray:
    """
    Ensure input is a numpy array.
    
    Args:
        value: Input value to convert
        
    Returns:
        Numpy array
    """
    if isinstance(value, (list, tuple)):
        return np.array(value)
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array([value])


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                default: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        default: Default value for division by zero
        
    Returns:
        Result of division with safe handling of zero division
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.divide(numerator, denominator)
        result = np.where(np.isfinite(result), result, default)
    return result


def calculate_stability_function(zeta: float) -> float:
    """
    Calculate atmospheric stability function for heat transfer.
    
    Args:
        zeta: Stability parameter (z/L where z is height and L is Obukhov length)
        
    Returns:
        Stability function value
    """
    if zeta < 0:  # Unstable conditions
        x = (1 - 16 * zeta) ** 0.25
        return 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
    elif zeta <= 1:  # Stable conditions
        return -5 * zeta
    else:  # Very stable conditions
        return 1 - 5 * zeta


def create_output_directory(path: str) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the created directory
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_scientific_notation(value: float, precision: int = 3) -> str:
    """
    Format number in scientific notation with specified precision.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{precision}e}"


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger('DepoBoxTool.Progress')
    
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current += increment
        if self.current % max(1, self.total // 10) == 0:
            percentage = (self.current / self.total) * 100
            self.logger.info(f"{self.description}: {percentage:.1f}% complete")
    
    def finish(self):
        """Mark progress as complete."""
        self.logger.info(f"{self.description}: 100% complete")


def deprecated(reason: str):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        reason: Reason for deprecation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator