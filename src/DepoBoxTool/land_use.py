"""
Land use parameter management for atmospheric deposition modeling.

This module provides a clean interface for managing land use specific parameters
required by different deposition schemes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .constants import LAND_USE_PARAMS
from .physics import calculate_water_roughness_length


@dataclass
class SurfaceParameters:
    """Container for surface-specific parameters."""
    roughness_length: float
    displacement_height: float
    measurement_height: float
    lai: float
    convective_velocity: float
    a_factor: float
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    surface_type: str = 'vegetation'


class LandUseBase(ABC):
    """Abstract base class for land use parameter management."""
    
    def __init__(self, friction_velocity: float, lai: float = 1.0, 
                 roughness_length: float = 0.03, displacement_height: float = 0.0,
                 measurement_height: float = 0.0, convective_velocity: float = 0.0):
        """
        Initialize land use parameters.
        
        Args:
            friction_velocity: Friction velocity in m/s
            lai: Leaf Area Index (dimensionless)
            roughness_length: Surface roughness length in meters
            displacement_height: Zero-plane displacement height in meters
            measurement_height: Measurement height in meters
            convective_velocity: Convective velocity scale in m/s
        """
        self.friction_velocity = friction_velocity
        self.lai = lai
        self.roughness_length = roughness_length
        self.displacement_height = displacement_height
        self.measurement_height = measurement_height
        self.convective_velocity = convective_velocity
    
    @abstractmethod
    def get_parameters(self, scheme: str) -> Dict[str, Any]:
        """Get parameters for specific deposition scheme."""
        pass
    
    def _get_base_parameters(self) -> Dict[str, Any]:
        """Get common parameters for all schemes."""
        return {
            'z0': self.roughness_length,
            'd': self.displacement_height,
            'z': self.measurement_height,
            'wstar': self.convective_velocity,
            'LAI': self.lai
        }


class GrassLandUse(LandUseBase):
    """Grass land use parameters."""
    
    def get_parameters(self, scheme: str) -> Dict[str, Any]:
        """Get grass-specific parameters for given scheme."""
        base_params = self._get_base_parameters()
        grass_params = LAND_USE_PARAMS['grass']
        
        scheme_specific = {
            'PR11': {},
            'VGLAI': {
                'A': grass_params['A_factor']
            },
            'STAGE': {
                'surface': 'vegetation',
                'alpha': grass_params['alpha_z01'],
                'A': grass_params['A_factor']
            },
            'Z01': {
                'surface': 'vegetation',
                'gamma': 0.54,
                'alpha': grass_params['alpha_z01'],
                'beta': grass_params['beta'],
                'A': grass_params['A_factor']
            },
            'CSU': {
                'surface': 'vegetation',
                'alpha': grass_params['alpha_z01'],
                'A': grass_params['A_factor']
            }
        }
        
        return {**base_params, **scheme_specific.get(scheme, {})}


class ConiferousForestLandUse(LandUseBase):
    """Coniferous forest land use parameters."""
    
    def get_parameters(self, scheme: str) -> Dict[str, Any]:
        """Get coniferous forest-specific parameters for given scheme."""
        base_params = self._get_base_parameters()
        forest_params = LAND_USE_PARAMS['coniferous_forest']
        
        scheme_specific = {
            'PR11': {},
            'VGLAI': {
                'A': forest_params['A_factor']
            },
            'STAGE': {
                'surface': 'vegetation',
                'alpha': forest_params['alpha_csu'],
                'A': forest_params['A_factor']
            },
            'Z01': {
                'surface': 'vegetation',
                'gamma': forest_params['gamma'],
                'alpha': forest_params['alpha_z01'],
                'beta': forest_params['beta'],
                'A': forest_params['A_factor']
            },
            'CSU': {
                'surface': 'vegetation',
                'alpha': forest_params['alpha_csu'],
                'A': forest_params['A_factor']
            }
        }
        
        return {**base_params, **scheme_specific.get(scheme, {})}


class DeciduousForestLandUse(LandUseBase):
    """Deciduous forest land use parameters."""
    
    def get_parameters(self, scheme: str) -> Dict[str, Any]:
        """Get deciduous forest-specific parameters for given scheme."""
        base_params = self._get_base_parameters()
        forest_params = LAND_USE_PARAMS['deciduous_forest']
        
        scheme_specific = {
            'PR11': {},
            'VGLAI': {
                'A': forest_params['A_factor']
            },
            'STAGE': {
                'surface': 'vegetation',
                'alpha': forest_params['alpha_csu'],
                'A': forest_params['A_factor']
            },
            'Z01': {
                'surface': 'vegetation',
                'gamma': forest_params['gamma'],
                'alpha': forest_params['alpha_z01'],
                'beta': forest_params['beta'],
                'A': forest_params['A_factor']
            },
            'CSU': {
                'surface': 'vegetation',
                'alpha': forest_params['alpha_csu'],
                'A': forest_params['A_factor']
            }
        }
        
        return {**base_params, **scheme_specific.get(scheme, {})}


class WaterLandUse(LandUseBase):
    """Water surface land use parameters."""
    
    def get_parameters(self, scheme: str) -> Dict[str, Any]:
        """Get water surface-specific parameters for given scheme."""
        # Calculate water-specific roughness length
        water_roughness = calculate_water_roughness_length(self.friction_velocity)
        
        base_params = self._get_base_parameters()
        base_params['z0'] = water_roughness  # Override with water-specific value
        
        water_params = LAND_USE_PARAMS['water']
        
        scheme_specific = {
            'PR11': {},
            'VGLAI': {
                'A': water_params['A_factor']
            },
            'STAGE': {
                'surface': 'smooth',
                'alpha': water_params['alpha_csu'],
                'A': water_params['A_factor']
            },
            'Z01': {
                'surface': 'smooth',
                'gamma': water_params['gamma'],
                'alpha': water_params['alpha_z01'],
                'beta': water_params['beta'],
                'A': water_params['A_factor']
            },
            'CSU': {
                'surface': 'smooth',
                'alpha': water_params['alpha_csu'],
                'A': water_params['A_factor']
            }
        }
        
        return {**base_params, **scheme_specific.get(scheme, {})}


class LandUseManager:
    """Factory class for creating appropriate land use objects."""
    
    _land_use_classes = {
        'grass': GrassLandUse,
        'coniferous_forest': ConiferousForestLandUse,
        'coniferousforest': ConiferousForestLandUse,  # Alternative naming
        'deciduous_forest': DeciduousForestLandUse,
        'deciduousforest': DeciduousForestLandUse,  # Alternative naming
        'water': WaterLandUse
    }
    
    @classmethod
    def create_land_use(cls, land_use_type: str, **kwargs) -> LandUseBase:
        """
        Create appropriate land use object.
        
        Args:
            land_use_type: Type of land use ('grass', 'coniferous_forest', etc.)
            **kwargs: Parameters to pass to land use constructor
            
        Returns:
            Appropriate land use object
            
        Raises:
            ValueError: If land use type is not supported
        """
        land_use_type_lower = land_use_type.lower()
        
        if land_use_type_lower not in cls._land_use_classes:
            available_types = list(cls._land_use_classes.keys())
            raise ValueError(f"Unsupported land use type: {land_use_type}. "
                           f"Available types: {available_types}")
        
        return cls._land_use_classes[land_use_type_lower](**kwargs)
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported land use types."""
        return list(cls._land_use_classes.keys())