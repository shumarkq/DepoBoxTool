"""
DepoBoxTool - A comprehensive tool for testing dry deposition algorithms

This package provides a modular framework for evaluating different dry deposition
algorithms across various land use categories. It is designed to support CMAQ
(Community Multiscale Air Quality) model improvements.

Author: Qian Shu
Email: shumarkq@gmail.com
Version: 2.0
"""

from __future__ import annotations

__version__ = "2.0"
__author__ = "Qian Shu"
__email__ = "shumarkq@gmail.com"

# Core modules
from .models import DepositionModel, SingleDiameterModel, ModalModel
from .land_use import LandUseParameters 
from .particle_distribution import ParticleDistribution
from .evaluation import EvaluationMetrics
from .constants import PhysicalConstants

# Main interfaces
from .models import RUN_SID, RUN_MOD
from .utils import setup_logging, validate_inputs

__all__ = [
    # Core classes
    'DepositionModel',
    'SingleDiameterModel', 
    'ModalModel',
    'LandUseParameters',
    'ParticleDistribution',
    'EvaluationMetrics',
    'PhysicalConstants',
    
    # Main interfaces  
    'RUN_SID',
    'RUN_MOD',
    
    # Utilities
    'setup_logging',
    'validate_inputs'
]
