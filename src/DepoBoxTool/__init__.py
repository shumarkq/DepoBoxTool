"""
DepoBoxTool: A Python tool for atmospheric particle dry deposition modeling.

This package provides implementations of various dry deposition schemes
commonly used in atmospheric modeling, with a focus on CMAQ models.

Main modules:
- models: Deposition scheme implementations
- physics: Core physics calculations
- land_use: Land use parameter management
- distributions: Particle size distribution utilities
- metrics: Model evaluation metrics
- interface: High-level user interface

Example usage:
    >>> from DepoBoxTool import calculate_deposition_velocity
    >>> result = calculate_deposition_velocity(
    ...     particle_density=1500,
    ...     particle_diameter=2.5e-6,
    ...     model_name='VGLAI',
    ...     land_use_type='grass'
    ... )
    >>> print(f"Deposition velocity: {result.deposition_velocity:.4f} m/s")
"""

from __future__ import print_function

# Version information
__version__ = "2.1.0"
__author__ = "Qian Shu"
__email__ = "shumarkq@gmail.com"

# Import core modules (these should work without external dependencies)
from . import constants

# Try to import modules that may require external dependencies
_OPTIONAL_IMPORTS = {}

try:
    from . import physics
    _OPTIONAL_IMPORTS['physics'] = True
except ImportError as e:
    _OPTIONAL_IMPORTS['physics'] = False
    print(f"Warning: Could not import physics module: {e}")

try:
    from . import models
    _OPTIONAL_IMPORTS['models'] = True
except ImportError as e:
    _OPTIONAL_IMPORTS['models'] = False
    print(f"Warning: Could not import models module: {e}")

try:
    from . import land_use
    _OPTIONAL_IMPORTS['land_use'] = True
except ImportError as e:
    _OPTIONAL_IMPORTS['land_use'] = False
    print(f"Warning: Could not import land_use module: {e}")

try:
    from . import distributions
    _OPTIONAL_IMPORTS['distributions'] = True
except ImportError as e:
    _OPTIONAL_IMPORTS['distributions'] = False
    print(f"Warning: Could not import distributions module: {e}")

try:
    from . import metrics
    _OPTIONAL_IMPORTS['metrics'] = True
except ImportError as e:
    _OPTIONAL_IMPORTS['metrics'] = False
    print(f"Warning: Could not import metrics module: {e}")

# Try to import high-level interface functions
try:
    from .interface import (
        calculate_deposition_velocity,
        calculate_size_distribution_deposition,
        compare_models,
        evaluate_model_performance,
        run_sensitivity_analysis,
        quick_calculation,
        get_model_info
    )
    _OPTIONAL_IMPORTS['interface'] = True
except ImportError as e:
    _OPTIONAL_IMPORTS['interface'] = False
    print(f"Warning: Could not import interface functions: {e}")

# Try to import core classes if available
if _OPTIONAL_IMPORTS.get('models', False):
    try:
        from .models import (
            DepositionModelFactory,
            ParticleProperties,
            AtmosphericConditions,
            DepositionResult
        )
    except ImportError:
        pass

if _OPTIONAL_IMPORTS.get('land_use', False):
    try:
        from .land_use import LandUseManager
    except ImportError:
        pass

if _OPTIONAL_IMPORTS.get('distributions', False):
    try:
        from .distributions import LognormalDistribution, SizeDistribution
    except ImportError:
        pass

if _OPTIONAL_IMPORTS.get('metrics', False):
    try:
        from .metrics import ModelEvaluator
    except ImportError:
        pass

# Legacy compatibility imports (only if available)
if _OPTIONAL_IMPORTS.get('models', False):
    try:
        from .models import (
            PR11Model as RUN_SID_PR11,
            VGLAIModel as RUN_SID_VGLAI,
            Z01Model as RUN_SID_Z01,
            CSUModel as RUN_SID_CSU,
            STAGEModel as RUN_SID_STAGE
        )
    except ImportError:
        pass

# Legacy function imports for backward compatibility
if _OPTIONAL_IMPORTS.get('distributions', False):
    try:
        from .distributions import bins, mbins
    except ImportError:
        pass

if _OPTIONAL_IMPORTS.get('metrics', False):
    try:
        from .metrics import (
            MB, ME, RMSE, R2, NMB, NME, FB, IOA
        )
    except ImportError:
        pass

# Build public API based on what's available
__all__ = [
    # Version info (always available)
    '__version__',
    '__author__',
    '__email__',
    
    # Core constants (always available)
    'constants'
]

# Add optional components to public API
if _OPTIONAL_IMPORTS.get('interface', False):
    __all__.extend([
        'calculate_deposition_velocity',
        'calculate_size_distribution_deposition', 
        'compare_models',
        'evaluate_model_performance',
        'run_sensitivity_analysis',
        'quick_calculation',
        'get_model_info'
    ])

if _OPTIONAL_IMPORTS.get('models', False):
    __all__.extend([
        'DepositionModelFactory',
        'ParticleProperties',
        'AtmosphericConditions',
        'DepositionResult'
    ])

if _OPTIONAL_IMPORTS.get('land_use', False):
    __all__.append('LandUseManager')

if _OPTIONAL_IMPORTS.get('distributions', False):
    __all__.extend(['LognormalDistribution', 'SizeDistribution'])

if _OPTIONAL_IMPORTS.get('metrics', False):
    __all__.append('ModelEvaluator')

# Add available submodules
for module_name, available in _OPTIONAL_IMPORTS.items():
    if available:
        __all__.append(module_name)

def get_import_status():
    """Get the status of optional imports."""
    return _OPTIONAL_IMPORTS.copy()

def check_dependencies():
    """Check which optional dependencies are available."""
    missing_deps = []
    available_deps = []
    
    try:
        import numpy
        available_deps.append('numpy')
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import scipy
        available_deps.append('scipy')
    except ImportError:
        missing_deps.append('scipy')
    
    try:
        import pandas
        available_deps.append('pandas')
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import matplotlib
        available_deps.append('matplotlib')
    except ImportError:
        missing_deps.append('matplotlib')
    
    return {
        'available': available_deps,
        'missing': missing_deps,
        'modules': _OPTIONAL_IMPORTS
    }

# Add utility functions to public API
__all__.extend(['get_import_status', 'check_dependencies'])
