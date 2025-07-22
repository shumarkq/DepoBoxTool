"""
Deposition model implementations.

This module contains the core deposition algorithms implemented as classes
with proper error handling, logging, and documentation. All models follow
a common interface for consistency and ease of use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging
from functools import lru_cache

from .constants import PhysicalConstants, DefaultParameters, MODEL_CONSTANTS
from .utils import validate_inputs, ensure_array, calculate_stability_function
from .land_use import LandUseParameters, BaseLandUseParameters


logger = logging.getLogger(__name__)


@dataclass
class DepositionResult:
    """Container for deposition calculation results."""
    
    deposition_velocity: float  # Total deposition velocity (m/s)
    gravitational_velocity: float  # Gravitational settling velocity (m/s)
    aerodynamic_resistance: float  # Aerodynamic resistance (s/m)
    surface_resistance: float  # Surface resistance (s/m)
    
    # Component velocities for analysis
    brownian_velocity: Optional[float] = None
    impaction_velocity: Optional[float] = None
    interception_velocity: Optional[float] = None
    
    # Additional parameters
    schmidt_number: Optional[float] = None
    stokes_number: Optional[float] = None
    collection_efficiency: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'Vd': self.deposition_velocity,
            'Vg': self.gravitational_velocity,
            'Ra': self.aerodynamic_resistance,
            'Rs': self.surface_resistance,
            'Vd_brownian': self.brownian_velocity,
            'Vd_impaction': self.impaction_velocity,
            'Vd_interception': self.interception_velocity,
            'Sc': self.schmidt_number,
            'St': self.stokes_number,
            'collection_efficiency': self.collection_efficiency
        }


class DepositionModel(ABC):
    """
    Abstract base class for all deposition models.
    
    This class defines the common interface that all deposition models
    must implement, ensuring consistency across different algorithms.
    """
    
    def __init__(self, 
                 particle_density: float,
                 particle_diameter: float,
                 temperature: float,
                 pressure: float,
                 relative_humidity: float,
                 friction_velocity: float,
                 monin_obukhov_length: float,
                 land_use_params: Dict[str, Any]):
        """
        Initialize deposition model.
        
        Args:
            particle_density: Particle density (kg/m³)
            particle_diameter: Particle diameter (m)
            temperature: Air temperature (K)
            pressure: Air pressure (Pa)
            relative_humidity: Relative humidity (fraction)
            friction_velocity: Friction velocity (m/s)
            monin_obukhov_length: Monin-Obukhov length (m)
            land_use_params: Land use specific parameters
        """
        # Validate inputs
        validated_params = validate_inputs(
            dens=particle_density,
            dim=particle_diameter,
            temp=temperature,
            press=pressure,
            RH=relative_humidity,
            ustar=friction_velocity
        )
        
        # Store basic parameters
        self.particle_density = particle_density
        self.particle_diameter = particle_diameter
        self.temperature = temperature
        self.pressure = pressure
        self.relative_humidity = relative_humidity
        self.friction_velocity = friction_velocity
        self.monin_obukhov_length = monin_obukhov_length
        self.land_use_params = land_use_params
        
        # Initialize logger
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        
        # Calculate derived physical properties
        self._calculate_physical_properties()
    
    def _calculate_physical_properties(self):
        """Calculate derived physical properties."""
        constants = PhysicalConstants()
        
        # Air viscosity (temperature dependent)
        self.dynamic_viscosity = constants.dynamic_viscosity_air(self.temperature)
        self.kinematic_viscosity = constants.kinematic_viscosity_air(self.temperature)
        
        # Mean free path
        self.mean_free_path = constants.mean_free_path(
            self.temperature, self.pressure, self.particle_diameter
        )
        
        # Cunningham slip correction factor
        self.cunningham_correction = self._calculate_cunningham_correction()
        
        # Gravitational settling velocity
        self.gravitational_velocity = self._calculate_gravitational_velocity()
        
        # Diffusion coefficient
        self.diffusion_coefficient = self._calculate_diffusion_coefficient()
        
        # Schmidt number
        self.schmidt_number = self.kinematic_viscosity / self.diffusion_coefficient
    
    def _calculate_cunningham_correction(self) -> float:
        """Calculate Cunningham slip correction factor."""
        knudsen = 2 * self.mean_free_path / self.particle_diameter
        return 1 + knudsen * (1.257 + 0.4 * np.exp(-0.55 / knudsen))
    
    def _calculate_gravitational_velocity(self) -> float:
        """Calculate gravitational settling velocity."""
        return (
            self.particle_density * self.particle_diameter ** 2 * 
            PhysicalConstants.GRAVITY * self.cunningham_correction /
            (18 * self.dynamic_viscosity)
        )
    
    def _calculate_diffusion_coefficient(self) -> float:
        """Calculate particle diffusion coefficient."""
        return (
            self.cunningham_correction * PhysicalConstants.BOLTZMANN * 
            self.temperature / (3 * np.pi * self.dynamic_viscosity * 
            self.particle_diameter)
        )
    
    def _calculate_aerodynamic_resistance(self) -> float:
        """Calculate aerodynamic resistance."""
        # Reference height calculation
        z_ref = self.land_use_params['z'] - self.land_use_params.get('d', 0)
        z0 = self.land_use_params['z0']
        
        # Stability parameter
        zeta = z_ref / self.monin_obukhov_length if self.monin_obukhov_length != 0 else 0
        
        # Stability function
        phi_h = calculate_stability_function(zeta)
        
        # Aerodynamic resistance
        ra = (
            0.95 * (np.log(z_ref / z0) - phi_h) / 
            (PhysicalConstants.VON_KARMAN * self.friction_velocity)
        )
        
        return max(ra, 0.1)  # Ensure positive resistance
    
    @abstractmethod
    def calculate_collection_efficiencies(self) -> Dict[str, float]:
        """
        Calculate collection efficiencies for different mechanisms.
        
        Returns:
            Dictionary containing collection efficiencies
        """
        pass
    
    @abstractmethod
    def calculate_surface_resistance(self) -> float:
        """
        Calculate surface resistance.
        
        Returns:
            Surface resistance (s/m)
        """
        pass
    
    def calculate_deposition_velocity(self) -> DepositionResult:
        """
        Calculate total deposition velocity.
        
        Returns:
            DepositionResult containing all calculated values
        """
        self.logger.debug(f"Calculating deposition velocity for {self.__class__.__name__}")
        
        # Calculate resistances
        ra = self._calculate_aerodynamic_resistance()
        rs = self.calculate_surface_resistance()
        
        # Calculate collection efficiencies
        efficiencies = self.calculate_collection_efficiencies()
        
        # Total deposition velocity
        vd = self.gravitational_velocity / (1 - np.exp(-self.gravitational_velocity * (ra + rs)))
        
        # Create result object
        result = DepositionResult(
            deposition_velocity=vd,
            gravitational_velocity=self.gravitational_velocity,
            aerodynamic_resistance=ra,
            surface_resistance=rs,
            schmidt_number=self.schmidt_number,
            collection_efficiency=efficiencies
        )
        
        self.logger.debug(f"Calculated Vd = {vd:.6e} m/s")
        return result


class SingleDiameterModel(DepositionModel):
    """
    Base class for single diameter deposition models.
    
    This class provides common functionality for models that operate
    on single particle diameters.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "single_diameter"


class PR11Model(SingleDiameterModel):
    """
    Implementation of the PR11 deposition model.
    
    This is the original CMAQ deposition scheme (pre-v5.3) based on
    Pleim and Ran (2011).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "PR11"
        self.constants = MODEL_CONSTANTS['PR11']
    
    def calculate_collection_efficiencies(self) -> Dict[str, float]:
        """Calculate collection efficiencies for PR11 model."""
        # Brownian diffusion efficiency
        eb = self.schmidt_number ** (-2/3)
        
        # Impaction efficiency (original PR11 formulation)
        stokes_number = (
            self.gravitational_velocity * self.friction_velocity ** 2 /
            (PhysicalConstants.GRAVITY * self.kinematic_viscosity)
        )
        
        c_factor = self.constants['c_impaction']
        eim = stokes_number ** 2 / (c_factor + stokes_number ** 2)
        
        # Interception efficiency (not used in original PR11)
        ein = 0.0
        
        return {
            'brownian': eb,
            'impaction': eim,
            'interception': ein,
            'stokes_number': stokes_number
        }
    
    def calculate_surface_resistance(self) -> float:
        """Calculate surface resistance for PR11 model."""
        efficiencies = self.calculate_collection_efficiencies()
        
        # Convection factor
        wstar = self.land_use_params.get('wstar', 0.0)
        convection_factor = 1 + self.constants['convection_factor_base'] * (wstar / self.friction_velocity) ** 2
        
        # Total collection efficiency
        total_efficiency = (
            efficiencies['brownian'] + 
            efficiencies['impaction'] + 
            efficiencies['interception']
        )
        
        # Surface resistance
        rs = 1 / (convection_factor * self.friction_velocity * total_efficiency)
        
        return max(rs, 0.1)  # Ensure positive resistance


class VGLAIModel(SingleDiameterModel):
    """
    Implementation of the VGLAI deposition model.
    
    This is the improved CMAQ deposition scheme (v5.3+) that includes
    vegetation effects through Leaf Area Index (LAI).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "VGLAI"
        self.constants = MODEL_CONSTANTS['VGLAI']
    
    def calculate_collection_efficiencies(self) -> Dict[str, float]:
        """Calculate collection efficiencies for VGLAI model."""
        # Brownian diffusion efficiency
        eb = self.schmidt_number ** (-2/3)
        
        # Impaction efficiency (modified for vegetation)
        characteristic_length = self.land_use_params.get('A', 2e-3)
        stokes_number = (
            self.gravitational_velocity * self.friction_velocity /
            (PhysicalConstants.GRAVITY * characteristic_length)
        )
        
        c_factor = self.constants['c_impaction']
        eim = stokes_number ** 2 / (c_factor + stokes_number ** 2)
        
        # Interception efficiency (not used in VGLAI)
        ein = 0.0
        
        return {
            'brownian': eb,
            'impaction': eim,
            'interception': ein,
            'stokes_number': stokes_number
        }
    
    def calculate_surface_resistance(self) -> float:
        """Calculate surface resistance for VGLAI model."""
        efficiencies = self.calculate_collection_efficiencies()
        
        # Vegetation factor
        lai = self.land_use_params.get('LAI', 0.0)
        f_veg = 1  # For single land use
        
        # Convection factor with LAI enhancement
        wstar = self.land_use_params.get('wstar', 0.0)
        convection_factor = (
            (1 + f_veg * max(0, lai - 1)) *
            (1 + self.constants['convection_factor_base'] * (wstar / self.friction_velocity) ** 2)
        )
        
        # Total collection efficiency
        total_efficiency = (
            efficiencies['brownian'] + 
            efficiencies['impaction'] + 
            efficiencies['interception']
        )
        
        # Surface resistance
        rs = 1 / (convection_factor * self.friction_velocity * total_efficiency)
        
        return max(rs, 0.1)


class STAGEModel(SingleDiameterModel):
    """
    Implementation of the STAGE deposition model.
    
    Based on the scheme described in relevant literature.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "STAGE"
        self.constants = MODEL_CONSTANTS['STAGE']
    
    def calculate_collection_efficiencies(self) -> Dict[str, float]:
        """Calculate collection efficiencies for STAGE model."""
        # Brownian diffusion efficiency
        eb = self.schmidt_number ** (-2/3)
        
        # Impaction efficiency
        characteristic_length = self.land_use_params.get('A', 2e-3)
        stokes_number = (
            self.gravitational_velocity * self.friction_velocity /
            (PhysicalConstants.GRAVITY * characteristic_length)
        )
        
        eim = stokes_number ** 2 / (1 + stokes_number ** 2)
        
        # Interception efficiency (not used in STAGE)
        ein = 0.0
        
        return {
            'brownian': eb,
            'impaction': eim,
            'interception': ein,
            'stokes_number': stokes_number
        }
    
    def calculate_surface_resistance(self) -> float:
        """Calculate surface resistance for STAGE model."""
        efficiencies = self.calculate_collection_efficiencies()
        
        # Vegetation factor
        lai = self.land_use_params.get('LAI', 1.0)
        vegetation_factor = max(lai, 1.0)
        
        # Total collection efficiency
        total_efficiency = (
            efficiencies['brownian'] + 
            efficiencies['impaction'] + 
            efficiencies['interception']
        )
        
        # Surface resistance
        rs = 1 / (vegetation_factor * self.friction_velocity * total_efficiency)
        
        return max(rs, 0.1)


class Z01Model(SingleDiameterModel):
    """
    Implementation of the Z01 deposition model.
    
    Based on Zhang et al. (2001) parameterization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "Z01"
        self.constants = MODEL_CONSTANTS['Z01']
    
    def calculate_collection_efficiencies(self) -> Dict[str, float]:
        """Calculate collection efficiencies for Z01 model."""
        # Brownian diffusion efficiency
        gamma = self.land_use_params.get('gamma', self.constants['gamma_default'])
        eb = self.schmidt_number ** (-gamma)
        
        # Impaction efficiency (depends on surface type)
        surface_type = self.land_use_params.get('surface', 'vegetation')
        
        if surface_type == 'smooth':
            stokes_number = (
                self.gravitational_velocity * self.friction_velocity ** 2 /
                (PhysicalConstants.GRAVITY * self.kinematic_viscosity)
            )
            eim = 10 ** (-3 / stokes_number) if stokes_number > 0 else 0
        else:  # vegetation
            characteristic_length = self.land_use_params.get('A', 2e-3)
            stokes_number = (
                self.gravitational_velocity * self.friction_velocity /
                (PhysicalConstants.GRAVITY * characteristic_length)
            )
            
            alpha = self.land_use_params.get('alpha', self.constants['alpha_default'])
            beta = self.land_use_params.get('beta', self.constants['beta_default'])
            eim = (stokes_number / (alpha + stokes_number)) ** beta
        
        # Interception efficiency
        characteristic_length = self.land_use_params.get('A', 2e-3)
        ein = 0.5 * (self.particle_diameter / characteristic_length) ** 2
        
        return {
            'brownian': eb,
            'impaction': eim,
            'interception': ein,
            'stokes_number': stokes_number
        }
    
    def calculate_surface_resistance(self) -> float:
        """Calculate surface resistance for Z01 model."""
        efficiencies = self.calculate_collection_efficiencies()
        
        # Rebound factor
        stokes_number = efficiencies['stokes_number']
        rebound_factor = np.exp(-(stokes_number ** 0.5))
        
        # Empirical constant
        e_constant = self.constants['empirical_constant']
        
        # Total collection efficiency
        total_efficiency = (
            efficiencies['brownian'] + 
            efficiencies['impaction'] + 
            efficiencies['interception']
        )
        
        # Surface resistance
        rs = 1 / (e_constant * self.friction_velocity * total_efficiency * rebound_factor)
        
        return max(rs, 0.1)


class CSUModel(SingleDiameterModel):
    """
    Implementation of the CSU deposition model.
    
    Colorado State University parameterization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "CSU"
        self.constants = MODEL_CONSTANTS['CSU']
    
    def calculate_collection_efficiencies(self) -> Dict[str, float]:
        """Calculate collection efficiencies for CSU model."""
        # Brownian diffusion efficiency
        eb = self.schmidt_number ** (-2/3)
        
        # Impaction efficiency
        characteristic_length = self.land_use_params.get('A', 2e-3)
        stokes_number = (
            self.gravitational_velocity * self.friction_velocity /
            (PhysicalConstants.GRAVITY * characteristic_length)
        )
        
        alpha = self.land_use_params.get('alpha', self.constants['alpha_default'])
        eim = stokes_number / (alpha + stokes_number)
        
        # Interception efficiency
        ein = 0.5 * (self.particle_diameter / characteristic_length) ** 2
        
        return {
            'brownian': eb,
            'impaction': eim,
            'interception': ein,
            'stokes_number': stokes_number
        }
    
    def calculate_surface_resistance(self) -> float:
        """Calculate surface resistance for CSU model."""
        efficiencies = self.calculate_collection_efficiencies()
        
        # Total collection efficiency
        total_efficiency = (
            efficiencies['brownian'] + 
            efficiencies['impaction'] + 
            efficiencies['interception']
        )
        
        # Surface resistance (simplified)
        rs = 1 / (self.friction_velocity * total_efficiency)
        
        return max(rs, 0.1)


# Model factory and convenience functions
MODEL_REGISTRY = {
    'PR11': PR11Model,
    'VGLAI': VGLAIModel,
    'STAGE': STAGEModel,
    'Z01': Z01Model,
    'CSU': CSUModel
}


def create_model(model_name: str, **kwargs) -> DepositionModel:
    """
    Factory function to create deposition models.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Model initialization parameters
        
    Returns:
        Initialized deposition model
        
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


# Main interface classes for backward compatibility
class RUN_SID:
    """
    Main interface for single diameter deposition calculations.
    
    This class provides the primary interface for running single diameter
    deposition models, maintaining backward compatibility with the original API.
    """
    
    def __init__(self, 
                 particle_density: float,
                 particle_diameter: float,
                 temperature: float,
                 pressure: float,
                 relative_humidity: float,
                 friction_velocity: float,
                 monin_obukhov_length: float,
                 land_use_params: Dict[str, Any]):
        """
        Initialize single diameter runner.
        
        Args:
            particle_density: Particle density (kg/m³)
            particle_diameter: Particle diameter (m)
            temperature: Air temperature (K)
            pressure: Air pressure (Pa)
            relative_humidity: Relative humidity (fraction)
            friction_velocity: Friction velocity (m/s)
            monin_obukhov_length: Monin-Obukhov length (m)
            land_use_params: Land use specific parameters
        """
        self.params = {
            'particle_density': particle_density,
            'particle_diameter': particle_diameter,
            'temperature': temperature,
            'pressure': pressure,
            'relative_humidity': relative_humidity,
            'friction_velocity': friction_velocity,
            'monin_obukhov_length': monin_obukhov_length,
            'land_use_params': land_use_params
        }
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    def run_model(self, model_name: str) -> DepositionResult:
        """
        Run a specific deposition model.
        
        Args:
            model_name: Name of the model to run
            
        Returns:
            DepositionResult with calculated values
        """
        self.logger.info(f"Running {model_name} model")
        model = create_model(model_name, **self.params)
        return model.calculate_deposition_velocity()
    
    # Individual model methods for backward compatibility
    def PR11(self) -> List[float]:
        """Run PR11 model - legacy interface."""
        result = self.run_model('PR11')
        return [
            result.deposition_velocity,
            result.gravitational_velocity,
            result.aerodynamic_resistance,
            result.surface_resistance
        ]
    
    def VGLAI(self) -> List[float]:
        """Run VGLAI model - legacy interface."""
        result = self.run_model('VGLAI')
        return [
            result.deposition_velocity,
            result.gravitational_velocity,
            result.aerodynamic_resistance,
            result.surface_resistance
        ]
    
    def STAGE(self) -> List[float]:
        """Run STAGE model - legacy interface."""
        result = self.run_model('STAGE')
        return [
            result.deposition_velocity,
            result.gravitational_velocity,
            result.aerodynamic_resistance,
            result.surface_resistance
        ]
    
    def Z01(self) -> List[float]:
        """Run Z01 model - legacy interface."""
        result = self.run_model('Z01')
        return [
            result.deposition_velocity,
            result.gravitational_velocity,
            result.aerodynamic_resistance,
            result.surface_resistance
        ]
    
         def CSU(self) -> List[float]:
         """Run CSU model - legacy interface."""
         result = self.run_model('CSU')
         return [
             result.deposition_velocity,
             result.gravitational_velocity,
             result.aerodynamic_resistance,
             result.surface_resistance
         ]


class ModalModel(DepositionModel):
    """
    Base class for modal deposition models.
    
    This class handles deposition calculations for modal aerosol
    representations with log-normal size distributions.
    """
    
    def __init__(self, 
                 geometric_mean_diameter: float,
                 geometric_standard_deviation: float,
                 moment_method: str = "number",
                 **kwargs):
        """
        Initialize modal model.
        
        Args:
            geometric_mean_diameter: Geometric mean diameter (m)
            geometric_standard_deviation: Geometric standard deviation
            moment_method: Method for moment calculations ("number", "surface", "volume")
            **kwargs: Other parameters passed to parent class
        """
        # Use geometric mean diameter as the representative diameter
        super().__init__(particle_diameter=geometric_mean_diameter, **kwargs)
        
        self.geometric_mean_diameter = geometric_mean_diameter
        self.geometric_standard_deviation = geometric_standard_deviation
        self.moment_method = moment_method
        self.model_type = "modal"
        
        # Calculate modal corrections
        self._calculate_modal_corrections()
    
    def _calculate_modal_corrections(self):
        """Calculate corrections for modal representation."""
        # Log of geometric standard deviation
        ln_sigma = np.log(self.geometric_standard_deviation)
        
        # Moment corrections based on log-normal distribution theory
        if self.moment_method == "number":
            # No correction for number-weighted calculations
            self.diameter_correction = 1.0
            self.velocity_correction = np.exp(2 * ln_sigma ** 2)
        elif self.moment_method == "surface":
            # Surface area weighted
            self.diameter_correction = np.exp(2 * ln_sigma ** 2)
            self.velocity_correction = np.exp(4 * ln_sigma ** 2)
        elif self.moment_method == "volume":
            # Volume/mass weighted
            self.diameter_correction = np.exp(3 * ln_sigma ** 2)
            self.velocity_correction = np.exp(6 * ln_sigma ** 2)
        else:
            raise ValueError(f"Unknown moment method: {self.moment_method}")
    
    def calculate_deposition_velocity(self) -> DepositionResult:
        """Calculate modal deposition velocity with corrections."""
        result = super().calculate_deposition_velocity()
        
        # Apply modal corrections
        result.deposition_velocity *= self.velocity_correction
        result.gravitational_velocity *= self.velocity_correction
        
        return result


class RUN_MOD:
    """
    Main interface for modal deposition calculations.
    
    This class provides the interface for running modal deposition
    models with size distribution considerations.
    """
    
    def __init__(self, 
                 particle_density: float,
                 geometric_mean_diameter: float,
                 temperature: float,
                 pressure: float,
                 relative_humidity: float,
                 friction_velocity: float,
                 monin_obukhov_length: float,
                 land_use_params: Dict[str, Any],
                 geometric_standard_deviation: float,
                 moment_method: str = "number"):
        """
        Initialize modal diameter runner.
        
        Args:
            particle_density: Particle density (kg/m³)
            geometric_mean_diameter: Geometric mean diameter (m)
            temperature: Air temperature (K)
            pressure: Air pressure (Pa)
            relative_humidity: Relative humidity (fraction)
            friction_velocity: Friction velocity (m/s)
            monin_obukhov_length: Monin-Obukhov length (m)
            land_use_params: Land use specific parameters
            geometric_standard_deviation: Geometric standard deviation
            moment_method: Method for moment calculations
        """
        self.params = {
            'particle_density': particle_density,
            'temperature': temperature,
            'pressure': pressure,
            'relative_humidity': relative_humidity,
            'friction_velocity': friction_velocity,
            'monin_obukhov_length': monin_obukhov_length,
            'land_use_params': land_use_params,
            'geometric_mean_diameter': geometric_mean_diameter,
            'geometric_standard_deviation': geometric_standard_deviation,
            'moment_method': moment_method
        }
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    def run_model(self, model_name: str) -> DepositionResult:
        """
        Run a specific modal deposition model.
        
        Args:
            model_name: Name of the model to run
            
        Returns:
            DepositionResult with calculated values
        """
        self.logger.info(f"Running modal {model_name} model")
        
        # Create base model and convert to modal
        base_class = MODEL_REGISTRY[model_name]
        
        # Create a modal version of the model
        class ModalVersionModel(ModalModel, base_class):
            def __init__(self, **kwargs):
                ModalModel.__init__(self, **kwargs)
        
        model = ModalVersionModel(**self.params)
        return model.calculate_deposition_velocity()
    
    # Individual model methods for backward compatibility
    def PR11(self) -> List[float]:
        """Run modal PR11 model - legacy interface."""
        result = self.run_model('PR11')
        return [result.deposition_velocity]
    
    def VGLAI(self) -> List[float]:
        """Run modal VGLAI model - legacy interface."""
        result = self.run_model('VGLAI')
        return [result.deposition_velocity]