"""
Physical constants and default parameters for deposition calculations.

This module centralizes all physical constants and default parameter values
used throughout the deposition models.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used in deposition calculations."""
    
    # Mathematical constants
    PI: float = 3.1416
    
    # Physical constants
    GRAVITY: float = 9.8  # m/s²
    VON_KARMAN: float = 0.4  # Von Karman constant
    BOLTZMANN: float = 1.38e-23  # J/K
    
    # Air properties at standard conditions
    AIR_DENSITY_STD: float = 1.225  # kg/m³
    
    @staticmethod
    def dynamic_viscosity_air(temperature: float) -> float:
        """
        Calculate dynamic viscosity of air as a function of temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Dynamic viscosity in kg/(m·s)
        """
        return 1.8e-5 * (temperature / 298) ** 0.85
    
    @staticmethod
    def kinematic_viscosity_air(temperature: float) -> float:
        """
        Calculate kinematic viscosity of air as a function of temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Kinematic viscosity in m²/s
        """
        dynamic_visc = PhysicalConstants.dynamic_viscosity_air(temperature)
        return dynamic_visc / PhysicalConstants.AIR_DENSITY_STD
    
    @staticmethod
    def mean_free_path(temperature: float, pressure: float, diameter: float) -> float:
        """
        Calculate air molecules mean free path.
        
        Args:
            temperature: Temperature in Kelvin
            pressure: Pressure in Pascal
            diameter: Particle diameter in meters
            
        Returns:
            Mean free path in meters
        """
        return (PhysicalConstants.BOLTZMANN * temperature / 
                (np.sqrt(2) * PhysicalConstants.PI * pressure * diameter ** 2))


@dataclass
class DefaultParameters:
    """Default parameter values for deposition calculations."""
    
    # Particle properties
    particle_density: float = 1500  # kg/m³
    particle_diameter: float = 2.5e-6  # m
    
    # Atmospheric conditions
    temperature: float = 298  # K
    relative_humidity: float = 0.9  # fraction
    pressure: float = 101325  # Pa
    friction_velocity: float = 0.5  # m/s
    
    # Site characteristics
    measurement_height: float = 0  # m
    monin_obukhov_length: float = 50  # m
    reference_height: float = 1  # m
    roughness_length: float = 0.03  # m
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'dens': self.particle_density,
            'dim': self.particle_diameter,
            'temp': self.temperature,
            'RH': self.relative_humidity,
            'press': self.pressure,
            'ustar': self.friction_velocity,
            'z': self.measurement_height,
            'Lo': self.monin_obukhov_length,
            'zr': self.reference_height,
            'z0': self.roughness_length
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'DefaultParameters':
        """Create DefaultParameters from dictionary."""
        return cls(
            particle_density=params.get('dens', cls.particle_density),
            particle_diameter=params.get('dim', cls.particle_diameter),
            temperature=params.get('temp', cls.temperature),
            relative_humidity=params.get('RH', cls.relative_humidity),
            pressure=params.get('press', cls.pressure),
            friction_velocity=params.get('ustar', cls.friction_velocity),
            measurement_height=params.get('z', cls.measurement_height),
            monin_obukhov_length=params.get('Lo', cls.monin_obukhov_length),
            reference_height=params.get('zr', cls.reference_height),
            roughness_length=params.get('z0', cls.roughness_length)
        )


# Model-specific constants
MODEL_CONSTANTS = {
    'PR11': {
        'c_impaction': 400,
        'convection_factor_base': 0.24
    },
    'VGLAI': {
        'c_impaction': 1,
        'convection_factor_base': 0.24
    },
    'STAGE': {
        'empirical_constant': 1.0
    },
    'Z01': {
        'empirical_constant': 3.0,
        'gamma_default': 0.54,
        'alpha_default': 1.2,
        'beta_default': 2.0
    },
    'CSU': {
        'alpha_default': 1.2
    }
}