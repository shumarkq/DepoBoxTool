"""
Particle size distribution utilities for atmospheric deposition modeling.

This module provides functions for creating and managing lognormal particle
size distributions for both number and mass concentrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from math import erf
from dataclasses import dataclass


@dataclass
class SizeDistribution:
    """Container for particle size distribution data."""
    diameters: np.ndarray
    weights: np.ndarray
    distribution_type: str
    total_concentration: float
    
    def plot(self, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plot the size distribution.
        
        Args:
            title: Optional plot title
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.diameters, self.weights, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Diameter (m)')
        plt.ylabel('Weight')
        plt.title(title or f'{self.distribution_type.title()} Size Distribution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class LognormalDistribution:
    """Lognormal particle size distribution generator."""
    
    def __init__(self, num_bins: int = 50, diameter_range: Tuple[float, float] = (1e-4, 1e3)):
        """
        Initialize lognormal distribution generator.
        
        Args:
            num_bins: Number of size bins
            diameter_range: Tuple of (min_diameter, max_diameter) in micrometers
        """
        self.num_bins = num_bins
        self.diameter_range = diameter_range
        
        # Create diameter bins
        self.diameter_edges = np.logspace(
            np.log10(diameter_range[0]), 
            np.log10(diameter_range[1]), 
            num_bins + 1
        )
        self.diameter_centers = self._calculate_bin_centers()
    
    def _calculate_bin_centers(self) -> np.ndarray:
        """Calculate geometric mean diameters for each bin."""
        return np.sqrt(self.diameter_edges[:-1] * self.diameter_edges[1:])
    
    def create_number_distribution(self, total_number: float, geometric_mean_diameter: float,
                                 geometric_standard_deviation: float) -> SizeDistribution:
        """
        Create number concentration lognormal size distribution.
        
        Args:
            total_number: Total number concentration
            geometric_mean_diameter: Geometric mean diameter in micrometers
            geometric_standard_deviation: Geometric standard deviation
            
        Returns:
            SizeDistribution object containing the distribution data
        """
        log_sigma = np.log(geometric_standard_deviation)
        sqrt_2_log_sigma = np.sqrt(2) * log_sigma
        
        # Calculate number concentration in each bin
        number_fractions = []
        for i in range(len(self.diameter_edges) - 1):
            d_lower = self.diameter_edges[i]
            d_upper = self.diameter_edges[i + 1]
            
            # Calculate cumulative distribution function values
            erf_upper = erf(np.log(d_upper / geometric_mean_diameter) / sqrt_2_log_sigma)
            erf_lower = erf(np.log(d_lower / geometric_mean_diameter) / sqrt_2_log_sigma)
            
            fraction = 0.5 * (erf_upper - erf_lower)
            number_fractions.append(fraction)
        
        number_concentrations = np.array(number_fractions) * total_number
        weights = number_concentrations / total_number
        
        return SizeDistribution(
            diameters=self.diameter_centers,
            weights=weights,
            distribution_type='number',
            total_concentration=total_number
        )
    
    def create_mass_distribution(self, total_mass: float, geometric_mean_diameter: float,
                               geometric_standard_deviation: float, 
                               particle_density: float) -> SizeDistribution:
        """
        Create mass concentration lognormal size distribution.
        
        Args:
            total_mass: Total mass concentration
            geometric_mean_diameter: Geometric mean diameter in micrometers
            geometric_standard_deviation: Geometric standard deviation
            particle_density: Particle density in kg/m³
            
        Returns:
            SizeDistribution object containing the distribution data
        """
        # First create number distribution
        number_dist = self.create_number_distribution(
            total_number=1.0,  # Normalize to 1 for now
            geometric_mean_diameter=geometric_mean_diameter,
            geometric_standard_deviation=geometric_standard_deviation
        )
        
        # Convert to mass using particle volume
        particle_volumes = (np.pi / 6) * (self.diameter_centers * 1e-6) ** 3  # Convert μm to m
        particle_masses = particle_volumes * particle_density
        
        # Calculate mass in each bin
        mass_concentrations = number_dist.weights * particle_masses
        total_calculated_mass = np.sum(mass_concentrations)
        
        # Normalize to desired total mass
        mass_weights = mass_concentrations / total_calculated_mass
        
        return SizeDistribution(
            diameters=self.diameter_centers,
            weights=mass_weights,
            distribution_type='mass',
            total_concentration=total_mass
        )
    
    def create_surface_area_distribution(self, total_surface_area: float, 
                                       geometric_mean_diameter: float,
                                       geometric_standard_deviation: float) -> SizeDistribution:
        """
        Create surface area lognormal size distribution.
        
        Args:
            total_surface_area: Total surface area concentration
            geometric_mean_diameter: Geometric mean diameter in micrometers
            geometric_standard_deviation: Geometric standard deviation
            
        Returns:
            SizeDistribution object containing the distribution data
        """
        # First create number distribution
        number_dist = self.create_number_distribution(
            total_number=1.0,  # Normalize to 1 for now
            geometric_mean_diameter=geometric_mean_diameter,
            geometric_standard_deviation=geometric_standard_deviation
        )
        
        # Convert to surface area
        particle_surface_areas = np.pi * (self.diameter_centers * 1e-6) ** 2  # Convert μm to m
        
        # Calculate surface area in each bin
        surface_area_concentrations = number_dist.weights * particle_surface_areas
        total_calculated_surface_area = np.sum(surface_area_concentrations)
        
        # Normalize to desired total surface area
        surface_area_weights = surface_area_concentrations / total_calculated_surface_area
        
        return SizeDistribution(
            diameters=self.diameter_centers,
            weights=surface_area_weights,
            distribution_type='surface_area',
            total_concentration=total_surface_area
        )


def create_bins(num_bins: int, geometric_standard_deviation: float, 
               geometric_mean_diameter: float, plot: bool = False) -> List[Tuple[float, float]]:
    """
    Create lognormal number concentration size distribution (legacy function).
    
    Args:
        num_bins: Number of size bins
        geometric_standard_deviation: Geometric standard deviation
        geometric_mean_diameter: Geometric mean diameter
        plot: Whether to plot the distribution
        
    Returns:
        List of tuples containing (diameter, weight) pairs
    """
    dist_generator = LognormalDistribution(num_bins)
    size_dist = dist_generator.create_number_distribution(
        total_number=1.0,
        geometric_mean_diameter=geometric_mean_diameter,
        geometric_standard_deviation=geometric_standard_deviation
    )
    
    if plot:
        size_dist.plot()
    
    return list(zip(size_dist.diameters, size_dist.weights))


def create_mass_bins(num_bins: int, geometric_standard_deviation: float, 
                    geometric_mean_diameter: float, particle_density: float,
                    plot: bool = False) -> List[Tuple[float, float]]:
    """
    Create lognormal mass concentration size distribution (legacy function).
    
    Args:
        num_bins: Number of size bins
        geometric_standard_deviation: Geometric standard deviation
        geometric_mean_diameter: Geometric mean diameter
        particle_density: Particle density in kg/m³
        plot: Whether to plot the distribution
        
    Returns:
        List of tuples containing (diameter, weight) pairs
    """
    dist_generator = LognormalDistribution(num_bins)
    size_dist = dist_generator.create_mass_distribution(
        total_mass=1.0,
        geometric_mean_diameter=geometric_mean_diameter,
        geometric_standard_deviation=geometric_standard_deviation,
        particle_density=particle_density
    )
    
    if plot:
        size_dist.plot()
    
    return list(zip(size_dist.diameters, size_dist.weights))


# Legacy function aliases for backward compatibility
bins = create_bins
mbins = create_mass_bins