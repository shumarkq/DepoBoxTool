"""
Physical calculations for atmospheric deposition modeling.

This module contains core physics calculations including air properties,
particle properties, and atmospheric stability functions.
"""

import numpy as np
from typing import Tuple
from .constants import (
    BOLTZMANN_CONSTANT, GRAVITY, PI, AIR_DENSITY, VON_KARMAN_CONSTANT, SQRT_2
)


def calculate_air_viscosity(temperature: float) -> Tuple[float, float]:
    """
    Calculate dynamic and kinematic viscosity of air.
    
    Args:
        temperature: Temperature in Kelvin
        
    Returns:
        Tuple of (dynamic_viscosity, kinematic_viscosity)
    """
    # Dynamic viscosity (kg/m/s) - ACP page 909, eq(19.18)
    dynamic_viscosity = 1.8e-5 * (temperature / 298) ** 0.85
    
    # Kinematic viscosity (m²/s)
    kinematic_viscosity = dynamic_viscosity / AIR_DENSITY
    
    return dynamic_viscosity, kinematic_viscosity


def calculate_mean_free_path(temperature: float, pressure: float, diameter: float) -> float:
    """
    Calculate air molecules mean free path.
    
    Args:
        temperature: Temperature in Kelvin
        pressure: Pressure in Pascal
        diameter: Particle diameter in meters
        
    Returns:
        Mean free path in meters
    """
    return BOLTZMANN_CONSTANT * temperature / (SQRT_2 * PI * pressure * diameter ** 2)


def calculate_slip_correction(diameter: float, mean_free_path: float) -> float:
    """
    Calculate Cunningham slip correction factor.
    
    Args:
        diameter: Particle diameter in meters
        mean_free_path: Air molecules mean free path in meters
        
    Returns:
        Slip correction factor (dimensionless)
    """
    knudsen_number = 2 * mean_free_path / diameter
    return 1 + knudsen_number * (1.257 + 0.4 * np.exp(-0.55 / knudsen_number))


def calculate_settling_velocity(density: float, diameter: float, temperature: float, 
                               pressure: float) -> float:
    """
    Calculate particle gravitational settling velocity.
    
    Args:
        density: Particle density in kg/m³
        diameter: Particle diameter in meters
        temperature: Temperature in Kelvin
        pressure: Pressure in Pascal
        
    Returns:
        Settling velocity in m/s
    """
    dynamic_viscosity, _ = calculate_air_viscosity(temperature)
    mean_free_path = calculate_mean_free_path(temperature, pressure, diameter)
    slip_correction = calculate_slip_correction(diameter, mean_free_path)
    
    return (density * diameter ** 2 * GRAVITY * slip_correction) / (18 * dynamic_viscosity)


def calculate_diffusivity(temperature: float, diameter: float, pressure: float) -> float:
    """
    Calculate particle Brownian diffusivity.
    
    Args:
        temperature: Temperature in Kelvin
        diameter: Particle diameter in meters
        pressure: Pressure in Pascal
        
    Returns:
        Diffusivity in m²/s
    """
    dynamic_viscosity, _ = calculate_air_viscosity(temperature)
    mean_free_path = calculate_mean_free_path(temperature, pressure, diameter)
    slip_correction = calculate_slip_correction(diameter, mean_free_path)
    
    return slip_correction * BOLTZMANN_CONSTANT * temperature / (3 * PI * dynamic_viscosity * diameter)


def calculate_stability_function(height_ratio: float, monin_obukhov_length: float) -> float:
    """
    Calculate atmospheric stability function for heat.
    
    Args:
        height_ratio: Measurement height / roughness length
        monin_obukhov_length: Monin-Obukhov length in meters
        
    Returns:
        Stability function value
    """
    stability_param = height_ratio / monin_obukhov_length
    
    if stability_param < 0:  # Unstable conditions
        return 2 * np.log((np.sqrt(1 - 16 * stability_param) + 1) / 2)
    elif stability_param <= 1:  # Stable conditions
        return -5 * stability_param
    else:  # Very stable conditions
        return 1 - 5 * stability_param


def calculate_aerodynamic_resistance(measurement_height: float, roughness_length: float,
                                   displacement_height: float, friction_velocity: float,
                                   monin_obukhov_length: float, 
                                   resistance_factor: float = 0.95) -> float:
    """
    Calculate aerodynamic resistance.
    
    Args:
        measurement_height: Height of measurement in meters
        roughness_length: Surface roughness length in meters
        displacement_height: Zero-plane displacement height in meters
        friction_velocity: Friction velocity in m/s
        monin_obukhov_length: Monin-Obukhov length in meters
        resistance_factor: Empirical resistance factor
        
    Returns:
        Aerodynamic resistance in s/m
    """
    effective_height = measurement_height - displacement_height
    height_ratio = effective_height / roughness_length
    
    stability_correction = calculate_stability_function(effective_height, monin_obukhov_length)
    
    return (resistance_factor * (np.log(height_ratio) - stability_correction) / 
            (VON_KARMAN_CONSTANT * friction_velocity))


def calculate_schmidt_number(kinematic_viscosity: float, diffusivity: float) -> float:
    """
    Calculate Schmidt number (dimensionless).
    
    Args:
        kinematic_viscosity: Kinematic viscosity in m²/s
        diffusivity: Particle diffusivity in m²/s
        
    Returns:
        Schmidt number (dimensionless)
    """
    return kinematic_viscosity / diffusivity


def calculate_stokes_number(settling_velocity: float, friction_velocity: float,
                          a_factor: float, gravity: float = GRAVITY) -> float:
    """
    Calculate Stokes number (dimensionless).
    
    Args:
        settling_velocity: Particle settling velocity in m/s
        friction_velocity: Friction velocity in m/s
        a_factor: Characteristic length scale in meters
        gravity: Gravitational acceleration in m/s²
        
    Returns:
        Stokes number (dimensionless)
    """
    return settling_velocity * friction_velocity / (gravity * a_factor)


def calculate_water_roughness_length(friction_velocity: float) -> float:
    """
    Calculate roughness length over water surface.
    
    Args:
        friction_velocity: Friction velocity in m/s
        
    Returns:
        Roughness length in meters
    """
    if friction_velocity <= 0.16:
        return 0.021 * friction_velocity ** 3.32
    else:
        return 0.00098 * friction_velocity ** 1.65