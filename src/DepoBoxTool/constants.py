"""
Physical constants and default parameters for DepoBoxTool.

This module contains all physical constants, default values, and configuration
parameters used throughout the deposition modeling calculations.
"""

import math
from typing import Dict, Any

# Physical constants
PI = 3.1416
GRAVITY = 9.8  # m/s²
VON_KARMAN_CONSTANT = 0.4
BOLTZMANN_CONSTANT = 1.38e-23  # J/K
AIR_DENSITY = 1.225  # kg/m³
SQRT_2 = math.sqrt(2)

# Default atmospheric conditions
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    'atmospheric': {
        'density': (1500, 'kg/m³', 'particle density'),
        'diameter': (2.5e-6, 'm', 'particle diameter'),
        'temperature': (298, 'K', 'temperature'),
        'relative_humidity': (0.9, '', 'relative humidity'),
        'pressure': (101325, 'Pa', 'pressure'),
        'friction_velocity': (0.5, 'm/s', 'friction velocity'),
    },
    'surface': {
        'measurement_height': (0, 'm', 'measurement height'),
        'monin_obukhov_length': (50, 'm', 'Monin-Obukhov length'),
        'reference_height': (1, 'm', 'height at which Vd is evaluated'),
        'roughness_length': (0.03, 'm', 'roughness length'),
    }
}

# Model-specific coefficients
SCHEME_COEFFICIENTS = {
    'PR11': {
        'c_factor': 400,
        'resistance_factor': 0.95,
        'stability_factor': 0.24,
    },
    'VGLAI': {
        'c_factor': 1,
        'resistance_factor': 0.95,
        'stability_factor': 0.24,
    },
    'Z01': {
        'empirical_constant': 3,
        'gamma_smooth': 0.50,
        'gamma_vegetation': (0.54, 0.56, 0.58),
    },
    'CSU': {
        'eb_factor': 0.2,
        'eim_factor': 0.4,
        'eim_exponent': 1.7,
        'ein_factor': 2.5,
        'ein_exponent': 0.8,
    }
}

# Land use parameters
LAND_USE_PARAMS = {
    'grass': {
        'A_factor': 2.0e-3,  # m
        'alpha_z01': 1.2,
        'beta': 2.0,
    },
    'coniferous_forest': {
        'A_factor': 3.5e-3,  # m
        'alpha_z01': 0.8,
        'alpha_csu': 0.8,
        'beta': 2.0,
        'gamma': 0.57,
    },
    'deciduous_forest': {
        'A_factor': 3.5e-3,  # m
        'alpha_z01': 0.95,
        'alpha_csu': 0.95,
        'beta': 2.0,
        'gamma': 0.56,
    },
    'water': {
        'A_factor': 2.0e-3,  # m
        'alpha_z01': 100.0,
        'alpha_csu': 100.0,
        'beta': 2.0,
        'gamma': 0.50,
    }
}