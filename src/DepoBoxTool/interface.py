"""
High-level interface for DepoBoxTool.

This module provides simple, user-friendly functions for common
dry deposition modeling tasks.
"""

from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd

from .models import (
    DepositionModelFactory, ParticleProperties, AtmosphericConditions, DepositionResult
)
from .land_use import LandUseManager
from .metrics import ModelEvaluator
from .distributions import LognormalDistribution
from .constants import DEFAULT_PARAMS


def calculate_deposition_velocity(
    particle_density: float,
    particle_diameter: float,
    temperature: float = 298,
    pressure: float = 101325,
    relative_humidity: float = 0.9,
    friction_velocity: float = 0.5,
    monin_obukhov_length: float = 50,
    land_use_type: str = 'grass',
    model_name: str = 'VGLAI',
    lai: float = 1.0,
    roughness_length: float = 0.03,
    displacement_height: float = 0.0,
    measurement_height: float = 0.0,
    convective_velocity: float = 0.0
) -> DepositionResult:
    """
    Calculate deposition velocity for a single particle.
    
    Args:
        particle_density: Particle density in kg/m³
        particle_diameter: Particle diameter in meters
        temperature: Temperature in Kelvin (default: 298)
        pressure: Pressure in Pascal (default: 101325)
        relative_humidity: Relative humidity 0-1 (default: 0.9)
        friction_velocity: Friction velocity in m/s (default: 0.5)
        monin_obukhov_length: Monin-Obukhov length in meters (default: 50)
        land_use_type: Land use type (default: 'grass')
        model_name: Deposition model name (default: 'VGLAI')
        lai: Leaf Area Index (default: 1.0)
        roughness_length: Surface roughness length in meters (default: 0.03)
        displacement_height: Zero-plane displacement height in meters (default: 0.0)
        measurement_height: Measurement height in meters (default: 0.0)
        convective_velocity: Convective velocity scale in m/s (default: 0.0)
        
    Returns:
        DepositionResult object containing all calculated values
        
    Example:
        >>> result = calculate_deposition_velocity(
        ...     particle_density=1500,
        ...     particle_diameter=2.5e-6,
        ...     model_name='VGLAI',
        ...     land_use_type='grass'
        ... )
        >>> print(f"Deposition velocity: {result.deposition_velocity:.4f} m/s")
    """
    # Create particle properties
    particle = ParticleProperties(density=particle_density, diameter=particle_diameter)
    
    # Create atmospheric conditions
    atmosphere = AtmosphericConditions(
        temperature=temperature,
        pressure=pressure,
        relative_humidity=relative_humidity,
        friction_velocity=friction_velocity,
        monin_obukhov_length=monin_obukhov_length
    )
    
    # Get land use parameters
    land_use = LandUseManager.create_land_use(
        land_use_type=land_use_type,
        friction_velocity=friction_velocity,
        lai=lai,
        roughness_length=roughness_length,
        displacement_height=displacement_height,
        measurement_height=measurement_height,
        convective_velocity=convective_velocity
    )
    
    surface_params = land_use.get_parameters(model_name)
    
    # Create and run model
    model = DepositionModelFactory.create_model(
        model_name=model_name,
        particle=particle,
        atmosphere=atmosphere,
        surface_params=surface_params
    )
    
    return model.calculate_deposition_velocity()


def calculate_size_distribution_deposition(
    particle_density: float,
    geometric_mean_diameter: float,
    geometric_standard_deviation: float,
    num_bins: int = 50,
    temperature: float = 298,
    pressure: float = 101325,
    relative_humidity: float = 0.9,
    friction_velocity: float = 0.5,
    monin_obukhov_length: float = 50,
    land_use_type: str = 'grass',
    model_name: str = 'VGLAI',
    **land_use_kwargs
) -> pd.DataFrame:
    """
    Calculate deposition velocities for a lognormal size distribution.
    
    Args:
        particle_density: Particle density in kg/m³
        geometric_mean_diameter: Geometric mean diameter in micrometers
        geometric_standard_deviation: Geometric standard deviation
        num_bins: Number of size bins (default: 50)
        temperature: Temperature in Kelvin (default: 298)
        pressure: Pressure in Pascal (default: 101325)
        relative_humidity: Relative humidity 0-1 (default: 0.9)
        friction_velocity: Friction velocity in m/s (default: 0.5)
        monin_obukhov_length: Monin-Obukhov length in meters (default: 50)
        land_use_type: Land use type (default: 'grass')
        model_name: Deposition model name (default: 'VGLAI')
        **land_use_kwargs: Additional land use parameters
        
    Returns:
        DataFrame with diameters, weights, and deposition velocities
        
    Example:
        >>> df = calculate_size_distribution_deposition(
        ...     particle_density=1500,
        ...     geometric_mean_diameter=0.1,  # micrometers
        ...     geometric_standard_deviation=2.0,
        ...     model_name='VGLAI',
        ...     land_use_type='grass'
        ... )
        >>> print(df.head())
    """
    # Create size distribution
    dist_generator = LognormalDistribution(num_bins)
    size_dist = dist_generator.create_number_distribution(
        total_number=1.0,
        geometric_mean_diameter=geometric_mean_diameter,
        geometric_standard_deviation=geometric_standard_deviation
    )
    
    # Calculate deposition velocity for each size bin
    results = []
    
    for diameter_um, weight in zip(size_dist.diameters, size_dist.weights):
        diameter_m = diameter_um * 1e-6  # Convert micrometers to meters
        
        result = calculate_deposition_velocity(
            particle_density=particle_density,
            particle_diameter=diameter_m,
            temperature=temperature,
            pressure=pressure,
            relative_humidity=relative_humidity,
            friction_velocity=friction_velocity,
            monin_obukhov_length=monin_obukhov_length,
            land_use_type=land_use_type,
            model_name=model_name,
            **land_use_kwargs
        )
        
        results.append({
            'diameter_um': diameter_um,
            'weight': weight,
            'deposition_velocity': result.deposition_velocity,
            'settling_velocity': result.settling_velocity,
            'aerodynamic_resistance': result.aerodynamic_resistance,
            'surface_resistance': result.surface_resistance,
            'brownian_efficiency': result.brownian_efficiency,
            'impaction_efficiency': result.impaction_efficiency,
            'interception_efficiency': result.interception_efficiency
        })
    
    return pd.DataFrame(results)


def compare_models(
    particle_density: float,
    particle_diameter: float,
    model_names: List[str] = None,
    land_use_type: str = 'grass',
    **kwargs
) -> pd.DataFrame:
    """
    Compare multiple deposition models for the same conditions.
    
    Args:
        particle_density: Particle density in kg/m³
        particle_diameter: Particle diameter in meters
        model_names: List of model names to compare (default: all models)
        land_use_type: Land use type (default: 'grass')
        **kwargs: Additional parameters for calculation
        
    Returns:
        DataFrame comparing results from different models
        
    Example:
        >>> comparison = compare_models(
        ...     particle_density=1500,
        ...     particle_diameter=2.5e-6,
        ...     model_names=['PR11', 'VGLAI', 'Z01'],
        ...     land_use_type='grass'
        ... )
        >>> print(comparison)
    """
    if model_names is None:
        model_names = DepositionModelFactory.get_supported_models()
    
    results = []
    
    for model_name in model_names:
        try:
            result = calculate_deposition_velocity(
                particle_density=particle_density,
                particle_diameter=particle_diameter,
                model_name=model_name,
                land_use_type=land_use_type,
                **kwargs
            )
            
            results.append({
                'model': model_name,
                'deposition_velocity': result.deposition_velocity,
                'settling_velocity': result.settling_velocity,
                'aerodynamic_resistance': result.aerodynamic_resistance,
                'surface_resistance': result.surface_resistance,
                'brownian_efficiency': result.brownian_efficiency,
                'impaction_efficiency': result.impaction_efficiency,
                'interception_efficiency': result.interception_efficiency
            })
            
        except Exception as e:
            print(f"Warning: Model {model_name} failed with error: {e}")
            continue
    
    return pd.DataFrame(results)


def evaluate_model_performance(
    observations: Union[List, np.ndarray],
    predictions: Union[List, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate model performance against observations.
    
    Args:
        observations: Observed deposition velocities
        predictions: Modeled deposition velocities
        
    Returns:
        Dictionary of evaluation metrics
        
    Example:
        >>> obs = [0.001, 0.002, 0.003, 0.004, 0.005]
        >>> pred = [0.0012, 0.0018, 0.0032, 0.0038, 0.0051]
        >>> metrics = evaluate_model_performance(obs, pred)
        >>> print(f"R²: {metrics['r_squared']:.3f}")
        >>> print(f"RMSE: {metrics['root_mean_square_error']:.6f}")
    """
    evaluator = ModelEvaluator(observations, predictions)
    return evaluator.calculate_all_metrics()


def run_sensitivity_analysis(
    base_parameters: Dict[str, Any],
    sensitivity_parameters: Dict[str, List[float]],
    model_name: str = 'VGLAI',
    land_use_type: str = 'grass'
) -> pd.DataFrame:
    """
    Run sensitivity analysis by varying specified parameters.
    
    Args:
        base_parameters: Base parameter values
        sensitivity_parameters: Parameters to vary with their ranges
        model_name: Deposition model name (default: 'VGLAI')
        land_use_type: Land use type (default: 'grass')
        
    Returns:
        DataFrame with parameter combinations and results
        
    Example:
        >>> base_params = {
        ...     'particle_density': 1500,
        ...     'particle_diameter': 2.5e-6
        ... }
        >>> sensitivity_params = {
        ...     'friction_velocity': [0.1, 0.3, 0.5, 0.7, 1.0],
        ...     'temperature': [288, 298, 308]
        ... }
        >>> results = run_sensitivity_analysis(base_params, sensitivity_params)
        >>> print(results.head())
    """
    results = []
    
    # Generate all parameter combinations
    import itertools
    param_names = list(sensitivity_parameters.keys())
    param_values = list(sensitivity_parameters.values())
    
    for combination in itertools.product(*param_values):
        # Create parameters for this combination
        params = base_parameters.copy()
        
        for name, value in zip(param_names, combination):
            params[name] = value
        
        try:
            result = calculate_deposition_velocity(
                model_name=model_name,
                land_use_type=land_use_type,
                **params
            )
            
            # Record the combination and result
            record = params.copy()
            record.update({
                'deposition_velocity': result.deposition_velocity,
                'settling_velocity': result.settling_velocity,
                'aerodynamic_resistance': result.aerodynamic_resistance,
                'surface_resistance': result.surface_resistance
            })
            
            results.append(record)
            
        except Exception as e:
            print(f"Warning: Parameter combination {dict(zip(param_names, combination))} "
                  f"failed with error: {e}")
            continue
    
    return pd.DataFrame(results)


# Convenience functions for common use cases
def quick_calculation(
    particle_size_um: float = 2.5,
    density: float = 1500,
    land_use: str = 'grass',
    model: str = 'VGLAI'
) -> float:
    """
    Quick calculation with default atmospheric conditions.
    
    Args:
        particle_size_um: Particle diameter in micrometers (default: 2.5)
        density: Particle density in kg/m³ (default: 1500)
        land_use: Land use type (default: 'grass')
        model: Model name (default: 'VGLAI')
        
    Returns:
        Deposition velocity in m/s
    """
    result = calculate_deposition_velocity(
        particle_density=density,
        particle_diameter=particle_size_um * 1e-6,
        land_use_type=land_use,
        model_name=model
    )
    return result.deposition_velocity


def get_model_info() -> Dict[str, List[str]]:
    """
    Get information about available models and land use types.
    
    Returns:
        Dictionary with available models and land use types
    """
    return {
        'models': DepositionModelFactory.get_supported_models(),
        'land_use_types': LandUseManager.get_supported_types()
    }