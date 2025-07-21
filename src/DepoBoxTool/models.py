"""
Dry deposition models for atmospheric particles.

This module provides implementations of various dry deposition schemes
commonly used in atmospheric modeling, including CMAQ schemes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .constants import SCHEME_COEFFICIENTS, GRAVITY, VON_KARMAN_CONSTANT
from .physics import (
    calculate_settling_velocity, calculate_diffusivity, 
    calculate_aerodynamic_resistance, calculate_schmidt_number,
    calculate_stokes_number, calculate_air_viscosity
)


@dataclass
class DepositionResult:
    """Container for deposition velocity calculation results."""
    deposition_velocity: float
    settling_velocity: float
    aerodynamic_resistance: float
    surface_resistance: float
    brownian_efficiency: float
    impaction_efficiency: float
    interception_efficiency: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy analysis."""
        return {
            'vd': self.deposition_velocity,
            'vg': self.settling_velocity,
            'ra': self.aerodynamic_resistance,
            'rs': self.surface_resistance,
            'eb': self.brownian_efficiency,
            'eim': self.impaction_efficiency,
            'ein': self.interception_efficiency
        }


@dataclass
class ParticleProperties:
    """Container for particle physical properties."""
    density: float  # kg/m³
    diameter: float  # m
    
    def __post_init__(self):
        """Validate particle properties."""
        if self.density <= 0:
            raise ValueError("Particle density must be positive")
        if self.diameter <= 0:
            raise ValueError("Particle diameter must be positive")


@dataclass
class AtmosphericConditions:
    """Container for atmospheric conditions."""
    temperature: float  # K
    pressure: float  # Pa
    relative_humidity: float  # 0-1
    friction_velocity: float  # m/s
    monin_obukhov_length: float  # m
    
    def __post_init__(self):
        """Validate atmospheric conditions."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if not 0 <= self.relative_humidity <= 1:
            raise ValueError("Relative humidity must be between 0 and 1")
        if self.friction_velocity <= 0:
            raise ValueError("Friction velocity must be positive")


class DepositionModelBase(ABC):
    """Abstract base class for dry deposition models."""
    
    def __init__(self, particle: ParticleProperties, atmosphere: AtmosphericConditions,
                 surface_params: Dict[str, Any]):
        """
        Initialize deposition model.
        
        Args:
            particle: Particle properties
            atmosphere: Atmospheric conditions
            surface_params: Surface-specific parameters
        """
        self.particle = particle
        self.atmosphere = atmosphere
        self.surface_params = surface_params
        
        # Calculate basic physics
        self.settling_velocity = calculate_settling_velocity(
            particle.density, particle.diameter, 
            atmosphere.temperature, atmosphere.pressure
        )
        
        self.diffusivity = calculate_diffusivity(
            atmosphere.temperature, particle.diameter, atmosphere.pressure
        )
        
        _, self.kinematic_viscosity = calculate_air_viscosity(atmosphere.temperature)
        
        # Calculate aerodynamic resistance
        self.aerodynamic_resistance = calculate_aerodynamic_resistance(
            measurement_height=surface_params.get('z', 0),
            roughness_length=surface_params.get('z0', 0.03),
            displacement_height=surface_params.get('d', 0),
            friction_velocity=atmosphere.friction_velocity,
            monin_obukhov_length=atmosphere.monin_obukhov_length
        )
    
    @abstractmethod
    def calculate_surface_resistance(self) -> Tuple[float, float, float, float]:
        """
        Calculate surface resistance and collection efficiencies.
        
        Returns:
            Tuple of (surface_resistance, brownian_eff, impaction_eff, interception_eff)
        """
        pass
    
    def calculate_deposition_velocity(self) -> DepositionResult:
        """
        Calculate total deposition velocity.
        
        Returns:
            DepositionResult containing all calculated values
        """
        surface_resistance, eb, eim, ein = self.calculate_surface_resistance()
        
        # Calculate deposition velocity using resistance analogy
        total_resistance = self.aerodynamic_resistance + surface_resistance
        
        if self.settling_velocity > 0 and total_resistance > 0:
            # Account for gravitational settling
            deposition_velocity = (self.settling_velocity / 
                                 (1 - np.exp(-self.settling_velocity * total_resistance)))
        else:
            deposition_velocity = 1 / total_resistance if total_resistance > 0 else 0
        
        return DepositionResult(
            deposition_velocity=deposition_velocity,
            settling_velocity=self.settling_velocity,
            aerodynamic_resistance=self.aerodynamic_resistance,
            surface_resistance=surface_resistance,
            brownian_efficiency=eb,
            impaction_efficiency=eim,
            interception_efficiency=ein
        )


class PR11Model(DepositionModelBase):
    """PR11 deposition scheme (original CMAQ scheme before v5.3)."""
    
    def calculate_surface_resistance(self) -> Tuple[float, float, float, float]:
        """Calculate surface resistance using PR11 formulation."""
        coeffs = SCHEME_COEFFICIENTS['PR11']
        
        # Schmidt number for Brownian diffusion
        schmidt_number = calculate_schmidt_number(self.kinematic_viscosity, self.diffusivity)
        brownian_efficiency = schmidt_number ** (-2/3)
        
        # Stokes number for impaction
        stokes_number = (self.settling_velocity * self.atmosphere.friction_velocity ** 2 / 
                        (GRAVITY * self.kinematic_viscosity))
        impaction_efficiency = (stokes_number ** 2 / 
                              (coeffs['c_factor'] + stokes_number ** 2))
        
        # No interception in PR11
        interception_efficiency = 0.0
        
        # Convective enhancement factor
        wstar = self.surface_params.get('wstar', 0)
        convective_factor = 1 + coeffs['stability_factor'] * (wstar / self.atmosphere.friction_velocity) ** 2
        
        # Surface resistance
        total_efficiency = brownian_efficiency + impaction_efficiency + interception_efficiency
        surface_resistance = 1 / (convective_factor * self.atmosphere.friction_velocity * total_efficiency)
        
        return surface_resistance, brownian_efficiency, impaction_efficiency, interception_efficiency


class VGLAIModel(DepositionModelBase):
    """VGLAI deposition scheme (CMAQ v5.3 scheme)."""
    
    def calculate_surface_resistance(self) -> Tuple[float, float, float, float]:
        """Calculate surface resistance using VGLAI formulation."""
        coeffs = SCHEME_COEFFICIENTS['VGLAI']
        
        # Schmidt number for Brownian diffusion
        schmidt_number = calculate_schmidt_number(self.kinematic_viscosity, self.diffusivity)
        brownian_efficiency = schmidt_number ** (-2/3)
        
        # Stokes number for impaction (modified formulation)
        a_factor = self.surface_params.get('A', 2e-3)
        stokes_number = calculate_stokes_number(
            self.settling_velocity, self.atmosphere.friction_velocity, a_factor
        )
        impaction_efficiency = stokes_number ** 2 / (coeffs['c_factor'] + stokes_number ** 2)
        
        # No interception in VGLAI
        interception_efficiency = 0.0
        
        # LAI and convective enhancement
        lai = self.surface_params.get('LAI', 1.0)
        wstar = self.surface_params.get('wstar', 0)
        
        vegetation_factor = 1 + max(0, lai - 1)
        convective_factor = 1 + coeffs['stability_factor'] * (wstar / self.atmosphere.friction_velocity) ** 2
        total_enhancement = vegetation_factor * convective_factor
        
        # Surface resistance
        total_efficiency = brownian_efficiency + impaction_efficiency + interception_efficiency
        surface_resistance = 1 / (total_enhancement * self.atmosphere.friction_velocity * total_efficiency)
        
        return surface_resistance, brownian_efficiency, impaction_efficiency, interception_efficiency


class Z01Model(DepositionModelBase):
    """Z01 deposition scheme."""
    
    def calculate_surface_resistance(self) -> Tuple[float, float, float, float]:
        """Calculate surface resistance using Z01 formulation."""
        coeffs = SCHEME_COEFFICIENTS['Z01']
        
        # Schmidt number for Brownian diffusion
        schmidt_number = calculate_schmidt_number(self.kinematic_viscosity, self.diffusivity)
        gamma = self.surface_params.get('gamma', 0.54)
        brownian_efficiency = schmidt_number ** (-gamma)
        
        # Surface type dependent calculations
        surface_type = self.surface_params.get('surface', 'vegetation')
        a_factor = self.surface_params.get('A', 2e-3)
        
        if surface_type == 'smooth':
            # Smooth surface (water)
            stokes_number = (self.settling_velocity * self.atmosphere.friction_velocity ** 2 / 
                           (GRAVITY * self.kinematic_viscosity))
            impaction_efficiency = 10 ** (-3 / stokes_number) if stokes_number > 0 else 0
        else:
            # Vegetation surface
            stokes_number = calculate_stokes_number(
                self.settling_velocity, self.atmosphere.friction_velocity, a_factor
            )
            alpha = self.surface_params.get('alpha', 1.2)
            beta = self.surface_params.get('beta', 2.0)
            impaction_efficiency = (stokes_number / (alpha + stokes_number)) ** beta
        
        # Interception efficiency
        interception_efficiency = 0.5 * (self.particle.diameter / a_factor) ** 2
        
        # Rebound factor
        rebound_factor = np.exp(-(stokes_number ** 0.5))
        
        # Surface resistance
        total_efficiency = brownian_efficiency + impaction_efficiency + interception_efficiency
        empirical_constant = coeffs['empirical_constant']
        surface_resistance = 1 / (empirical_constant * self.atmosphere.friction_velocity * 
                                total_efficiency * rebound_factor)
        
        return surface_resistance, brownian_efficiency, impaction_efficiency, interception_efficiency


class CSUModel(DepositionModelBase):
    """CSU deposition scheme."""
    
    def calculate_surface_resistance(self) -> Tuple[float, float, float, float]:
        """Calculate surface resistance using CSU formulation."""
        coeffs = SCHEME_COEFFICIENTS['CSU']
        
        # Schmidt number for Brownian diffusion
        schmidt_number = calculate_schmidt_number(self.kinematic_viscosity, self.diffusivity)
        brownian_efficiency = coeffs['eb_factor'] * schmidt_number ** (-2/3)
        
        # Stokes number for impaction
        a_factor = self.surface_params.get('A', 2e-3)
        stokes_number = calculate_stokes_number(
            self.settling_velocity, self.atmosphere.friction_velocity, a_factor
        )
        
        alpha = self.surface_params.get('alpha', 1.2)
        impaction_efficiency = (coeffs['eim_factor'] * 
                              (stokes_number / (alpha + stokes_number)) ** coeffs['eim_exponent'])
        
        # Interception efficiency
        interception_efficiency = (coeffs['ein_factor'] * 
                                 (self.particle.diameter / a_factor) ** coeffs['ein_exponent'])
        
        # Rebound and vegetation factors
        rebound_factor = np.exp(-(stokes_number ** 0.5))
        lai = self.surface_params.get('LAI', 1.0)
        vegetation_factor = max(lai, 1.0)
        
        # Surface resistance
        total_efficiency = brownian_efficiency + impaction_efficiency + interception_efficiency
        surface_resistance = 1 / (vegetation_factor * self.atmosphere.friction_velocity * 
                                total_efficiency * rebound_factor)
        
        return surface_resistance, brownian_efficiency, impaction_efficiency, interception_efficiency


class STAGEModel(DepositionModelBase):
    """STAGE deposition scheme."""
    
    def calculate_surface_resistance(self) -> Tuple[float, float, float, float]:
        """Calculate surface resistance using STAGE formulation."""
        # Schmidt number for Brownian diffusion
        schmidt_number = calculate_schmidt_number(self.kinematic_viscosity, self.diffusivity)
        brownian_efficiency = schmidt_number ** (-2/3)
        
        # Stokes number for impaction
        a_factor = self.surface_params.get('A', 2e-3)
        stokes_number = calculate_stokes_number(
            self.settling_velocity, self.atmosphere.friction_velocity, a_factor
        )
        impaction_efficiency = stokes_number ** 2 / (1 + stokes_number ** 2)
        
        # No interception in STAGE
        interception_efficiency = 0.0
        
        # Vegetation factor
        lai = self.surface_params.get('LAI', 1.0)
        vegetation_factor = max(lai, 1.0)
        
        # Surface resistance
        total_efficiency = brownian_efficiency + impaction_efficiency + interception_efficiency
        surface_resistance = 1 / (vegetation_factor * self.atmosphere.friction_velocity * total_efficiency)
        
        return surface_resistance, brownian_efficiency, impaction_efficiency, interception_efficiency


class DepositionModelFactory:
    """Factory for creating deposition model instances."""
    
    _models = {
        'PR11': PR11Model,
        'VGLAI': VGLAIModel,
        'Z01': Z01Model,
        'CSU': CSUModel,
        'STAGE': STAGEModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, particle: ParticleProperties,
                    atmosphere: AtmosphericConditions, 
                    surface_params: Dict[str, Any]) -> DepositionModelBase:
        """
        Create a deposition model instance.
        
        Args:
            model_name: Name of the model ('PR11', 'VGLAI', etc.)
            particle: Particle properties
            atmosphere: Atmospheric conditions
            surface_params: Surface parameters
            
        Returns:
            Appropriate model instance
            
        Raises:
            ValueError: If model name is not supported
        """
        model_name_upper = model_name.upper()
        
        if model_name_upper not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Available models: {available_models}")
        
        return cls._models[model_name_upper](particle, atmosphere, surface_params)
    
    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model names."""
        return list(cls._models.keys())