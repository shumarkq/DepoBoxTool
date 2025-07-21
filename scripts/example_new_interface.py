#!/usr/bin/env python3
"""
Example script demonstrating the new DepoBoxTool interface.

This script shows how to use the restructured and improved DepoBoxTool
package for various deposition modeling tasks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DepoBoxTool import (
    calculate_deposition_velocity,
    calculate_size_distribution_deposition,
    compare_models,
    evaluate_model_performance,
    quick_calculation,
    get_model_info
)

def main():
    print("="*60)
    print("DepoBoxTool v2.1 - Restructured and Improved")
    print("="*60)
    
    # 1. Basic single particle calculation
    print("\n1. Basic single particle calculation:")
    print("-" * 40)
    
    result = calculate_deposition_velocity(
        particle_density=1500,  # kg/m³
        particle_diameter=2.5e-6,  # meters (2.5 μm)
        model_name='VGLAI',
        land_use_type='grass',
        friction_velocity=0.5,
        temperature=298
    )
    
    print(f"Deposition velocity: {result.deposition_velocity:.4f} m/s")
    print(f"Settling velocity: {result.settling_velocity:.4f} m/s")
    print(f"Aerodynamic resistance: {result.aerodynamic_resistance:.2f} s/m")
    print(f"Surface resistance: {result.surface_resistance:.2f} s/m")
    
    # 2. Quick calculation with defaults
    print("\n2. Quick calculation with defaults:")
    print("-" * 40)
    
    vd_quick = quick_calculation(particle_size_um=2.5, density=1500)
    print(f"Quick calculation result: {vd_quick:.4f} m/s")
    
    # 3. Compare different models
    print("\n3. Model comparison:")
    print("-" * 40)
    
    comparison = compare_models(
        particle_density=1500,
        particle_diameter=2.5e-6,
        model_names=['PR11', 'VGLAI', 'Z01', 'CSU'],
        land_use_type='grass'
    )
    
    print("Model comparison results:")
    print(comparison[['model', 'deposition_velocity', 'settling_velocity']].round(6))
    
    # 4. Size distribution calculation
    print("\n4. Size distribution calculation:")
    print("-" * 40)
    
    size_dist_results = calculate_size_distribution_deposition(
        particle_density=1500,
        geometric_mean_diameter=0.1,  # micrometers
        geometric_standard_deviation=2.0,
        num_bins=20,
        model_name='VGLAI',
        land_use_type='grass'
    )
    
    print("Size distribution results (first 5 bins):")
    print(size_dist_results[['diameter_um', 'weight', 'deposition_velocity']].head())
    
    # 5. Model evaluation example
    print("\n5. Model evaluation example:")
    print("-" * 40)
    
    # Generate some synthetic data for demonstration
    np.random.seed(42)
    observed = np.random.lognormal(mean=-6, sigma=0.5, size=50)  # Log-normal distribution
    predicted = observed * np.random.normal(1.0, 0.2, size=50)  # Add some noise
    
    metrics = evaluate_model_performance(observed, predicted)
    
    print("Model performance metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            print(f"  {metric}: {value:.4f}")
    
    # 6. Show available models and land use types
    print("\n6. Available models and land use types:")
    print("-" * 40)
    
    info = get_model_info()
    print(f"Available models: {', '.join(info['models'])}")
    print(f"Available land use types: {', '.join(info['land_use_types'])}")
    
    # 7. Create a simple plot
    print("\n7. Creating visualization:")
    print("-" * 40)
    
    # Calculate deposition velocity vs particle size
    diameters = np.logspace(-2, 1, 50)  # 0.01 to 10 μm
    vd_values = []
    
    for diameter in diameters:
        vd = quick_calculation(particle_size_um=diameter, density=1500)
        vd_values.append(vd)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.loglog(diameters, vd_values, 'b-', linewidth=2, label='VGLAI - Grass')
    plt.xlabel('Particle Diameter (μm)')
    plt.ylabel('Deposition Velocity (m/s)')
    plt.title('Deposition Velocity vs Particle Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_path = os.path.join('..', 'figs', 'example_deposition_curve.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("The new interface is much cleaner and easier to use.")
    print("="*60)


if __name__ == '__main__':
    main()