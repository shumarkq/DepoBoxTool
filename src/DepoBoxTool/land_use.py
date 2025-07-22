"""
Land use parameters for different surface types.

This module provides a comprehensive framework for managing land use
specific parameters required by various deposition models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging
from .utils import validate_inputs


logger = logging.getLogger(__name__)


@dataclass
class BaseLandUseParameters:
    """Base class for land use parameters."""
    
    # Surface characteristics
    roughness_length: float  # z0 (m)
    displacement_height: float  # d (m) 
    measurement_height: float  # z (m)
    convective_velocity: float  # wstar (m/s)
    
    # Optional parameters
    leaf_area_index: Optional[float] = None  # LAI
    characteristic_length: Optional[float] = None  # A (m)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        params = {
            'z0': self.roughness_length,
            'z': self.measurement_height,
            'wstar': self.convective_velocity
        }
        
        if self.leaf_area_index is not None:
            params['LAI'] = self.leaf_area_index
        if self.characteristic_length is not None:
            params['A'] = self.characteristic_length
            
        validate_inputs(**params)


class LandUseParameters:
    """
    Factory class for creating land use specific parameters.
    
    This class provides methods to generate parameter sets for different
    land use categories and deposition models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    def get_parameters(self, 
                      land_use_type: str, 
                      model_name: str,
                      base_params: BaseLandUseParameters,
                      **kwargs) -> Dict[str, Any]:
        """
        Get land use parameters for a specific model and surface type.
        
        Args:
            land_use_type: Type of land use ('grass', 'cforest', 'dforest', 'water')
            model_name: Name of deposition model ('PR11', 'VGLAI', 'STAGE', 'Z01', 'CSU')
            base_params: Base land use parameters
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary of model-specific parameters
            
        Raises:
            ValueError: If land use type or model is not supported
        """
        if land_use_type not in ['grass', 'cforest', 'dforest', 'water']:
            raise ValueError(f"Unsupported land use type: {land_use_type}")
            
        if model_name not in ['PR11', 'VGLAI', 'STAGE', 'Z01', 'CSU']:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get method name
        method_name = f"_{land_use_type}_{model_name.lower()}"
        
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            params = method(base_params, **kwargs)
            self.logger.debug(f"Generated {model_name} parameters for {land_use_type}: {params}")
            return params
        else:
            raise ValueError(f"Parameter generation not implemented for {model_name} with {land_use_type}")
    
    # Grass land use parameters
    def _grass_pr11(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate PR11 parameters for grass surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity
        }
    
    def _grass_vglai(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate VGLAI parameters for grass surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'LAI': params.leaf_area_index or kwargs.get('LAI', 2.0),
            'A': params.characteristic_length or kwargs.get('A', 2.0 / 1000)
        }
    
    def _grass_stage(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate STAGE parameters for grass surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'surface': 'vegetation',
            'alpha': kwargs.get('alpha', (1.0 + 0.6) / 2),
            'LAI': params.leaf_area_index or kwargs.get('LAI', 2.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 5) / 2 / 1000)
        }
    
    def _grass_z01(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate Z01 parameters for grass surfaces."""
        return {
            'z0': params.roughness_length,
            'z': params.measurement_height,
            'd': params.displacement_height,
            'surface': 'vegetation',
            'gamma': kwargs.get('gamma', 0.54),
            'alpha': kwargs.get('alpha', 1.2),
            'beta': kwargs.get('beta', 2.0),
            'A': params.characteristic_length or kwargs.get('A', 2.0 / 1000)
        }
    
    def _grass_csu(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate CSU parameters for grass surfaces."""
        return {
            'z0': params.roughness_length,
            'z': params.measurement_height,
            'd': params.displacement_height,
            'surface': 'vegetation',
            'alpha': kwargs.get('alpha', 1.2),
            'LAI': params.leaf_area_index or kwargs.get('LAI', 2.0),
            'A': params.characteristic_length or kwargs.get('A', 2.0 / 1000)
        }
    
    # Coniferous forest parameters
    def _cforest_pr11(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate PR11 parameters for coniferous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity
        }
    
    def _cforest_vglai(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate VGLAI parameters for coniferous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'LAI': params.leaf_area_index or kwargs.get('LAI', 6.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 5) / 2 / 1000)
        }
    
    def _cforest_stage(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate STAGE parameters for coniferous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'surface': 'vegetation',
            'alpha': kwargs.get('alpha', (1.0 + 0.6) / 2),
            'LAI': params.leaf_area_index or kwargs.get('LAI', 6.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 5) / 2 / 1000)
        }
    
    def _cforest_z01(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate Z01 parameters for coniferous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'surface': 'vegetation',
            'gamma': kwargs.get('gamma', 0.56),
            'alpha': kwargs.get('alpha', 1.0),
            'beta': kwargs.get('beta', 2.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 5) / 2 / 1000)
        }
    
    def _cforest_csu(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate CSU parameters for coniferous forest."""
        return {
            'z0': params.roughness_length,
            'z': params.measurement_height,
            'd': params.displacement_height,
            'surface': 'vegetation',
            'alpha': kwargs.get('alpha', 1.0),
            'LAI': params.leaf_area_index or kwargs.get('LAI', 6.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 5) / 2 / 1000)
        }
    
    # Deciduous forest parameters  
    def _dforest_pr11(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate PR11 parameters for deciduous forest."""
        return self._cforest_pr11(params, **kwargs)
    
    def _dforest_vglai(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate VGLAI parameters for deciduous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'LAI': params.leaf_area_index or kwargs.get('LAI', 5.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 8) / 2 / 1000)
        }
    
    def _dforest_stage(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate STAGE parameters for deciduous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'surface': 'vegetation',
            'alpha': kwargs.get('alpha', (1.5 + 0.6) / 2),
            'LAI': params.leaf_area_index or kwargs.get('LAI', 5.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 8) / 2 / 1000)
        }
    
    def _dforest_z01(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate Z01 parameters for deciduous forest."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'surface': 'vegetation',
            'gamma': kwargs.get('gamma', 0.56),
            'alpha': kwargs.get('alpha', 1.5),
            'beta': kwargs.get('beta', 2.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 8) / 2 / 1000)
        }
    
    def _dforest_csu(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate CSU parameters for deciduous forest."""
        return {
            'z0': params.roughness_length,
            'z': params.measurement_height,
            'd': params.displacement_height,
            'surface': 'vegetation',
            'alpha': kwargs.get('alpha', 1.5),
            'LAI': params.leaf_area_index or kwargs.get('LAI', 5.0),
            'A': params.characteristic_length or kwargs.get('A', (2 + 8) / 2 / 1000)
        }
    
    # Water surface parameters
    def _water_pr11(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate PR11 parameters for water surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity
        }
    
    def _water_vglai(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate VGLAI parameters for water surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'LAI': 0.0,  # No vegetation over water
            'A': kwargs.get('A', 1e-5)  # Very small characteristic length
        }
    
    def _water_stage(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate STAGE parameters for water surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'wstar': params.convective_velocity,
            'surface': 'smooth',
            'alpha': kwargs.get('alpha', 0.0),
            'LAI': 0.0,
            'A': kwargs.get('A', 1e-5)
        }
    
    def _water_z01(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate Z01 parameters for water surfaces."""
        return {
            'z0': params.roughness_length,
            'd': params.displacement_height,
            'z': params.measurement_height,
            'surface': 'smooth',
            'gamma': kwargs.get('gamma', 0.5),
            'alpha': kwargs.get('alpha', 0.0),
            'beta': kwargs.get('beta', 1.0),
            'A': kwargs.get('A', 1e-5)
        }
    
    def _water_csu(self, params: BaseLandUseParameters, **kwargs) -> Dict[str, Any]:
        """Generate CSU parameters for water surfaces."""
        return {
            'z0': params.roughness_length,
            'z': params.measurement_height,
            'd': params.displacement_height,
            'surface': 'smooth',
            'alpha': kwargs.get('alpha', 0.0),
            'LAI': 0.0,
            'A': kwargs.get('A', 1e-5)
        }


class StandardLandUseLibrary:
    """
    Library of standard land use parameter sets.
    
    This class provides pre-defined parameter sets for common
    land use categories based on literature values.
    """
    
    @staticmethod
    def grass_short() -> BaseLandUseParameters:
        """Parameters for short grass surfaces."""
        return BaseLandUseParameters(
            roughness_length=0.01,
            displacement_height=0.0,
            measurement_height=2.0,
            convective_velocity=1.0,
            leaf_area_index=2.0,
            characteristic_length=0.002
        )
    
    @staticmethod
    def grass_tall() -> BaseLandUseParameters:
        """Parameters for tall grass surfaces."""
        return BaseLandUseParameters(
            roughness_length=0.05,
            displacement_height=0.1,
            measurement_height=2.0,
            convective_velocity=1.0,
            leaf_area_index=3.0,
            characteristic_length=0.003
        )
    
    @staticmethod
    def coniferous_forest() -> BaseLandUseParameters:
        """Parameters for coniferous forest."""
        return BaseLandUseParameters(
            roughness_length=1.0,
            displacement_height=10.0,
            measurement_height=20.0,
            convective_velocity=2.0,
            leaf_area_index=6.0,
            characteristic_length=0.0035
        )
    
    @staticmethod
    def deciduous_forest() -> BaseLandUseParameters:
        """Parameters for deciduous forest."""
        return BaseLandUseParameters(
            roughness_length=1.5,
            displacement_height=12.0,
            measurement_height=20.0,
            convective_velocity=2.0,
            leaf_area_index=5.0,
            characteristic_length=0.005
        )
    
    @staticmethod
    def water_surface() -> BaseLandUseParameters:
        """Parameters for water surfaces."""
        return BaseLandUseParameters(
            roughness_length=0.0001,
            displacement_height=0.0,
            measurement_height=2.0,
            convective_velocity=0.5,
            leaf_area_index=0.0,
            characteristic_length=1e-5
        )