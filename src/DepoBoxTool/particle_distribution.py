"""
Particle size distribution functions.

This module provides tools for creating and manipulating particle size
distributions, including log-normal distributions for both number and
mass concentrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import logging

from .utils import validate_inputs, ensure_array


logger = logging.getLogger(__name__)


@dataclass
class DistributionParameters:
    """Parameters for particle size distributions."""
    
    number_concentration: float  # Total number concentration
    geometric_mean_diameter: float  # Geometric mean diameter (m)
    geometric_standard_deviation: float  # Geometric standard deviation
    particle_density: float  # Particle density (kg/m³)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        validate_inputs(
            num=self.number_concentration,
            dpg=self.geometric_mean_diameter,
            sigmag=self.geometric_standard_deviation,
            dens=self.particle_density
        )


class ParticleDistribution:
    """
    Base class for particle size distributions.
    
    This class provides common functionality for creating and manipulating
    particle size distributions used in deposition calculations.
    """
    
    def __init__(self, 
                 num_bins: int = 50,
                 diameter_range: Tuple[float, float] = (1e-9, 1e-3)):
        """
        Initialize particle distribution.
        
        Args:
            num_bins: Number of size bins
            diameter_range: Tuple of (min_diameter, max_diameter) in meters
        """
        self.num_bins = num_bins
        self.diameter_range = diameter_range
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        
        # Validate inputs
        if num_bins < 2:
            raise ValueError("Number of bins must be at least 2")
        if diameter_range[0] >= diameter_range[1]:
            raise ValueError("Maximum diameter must be greater than minimum diameter")
        
        # Create diameter bins
        self._create_diameter_bins()
    
    def _create_diameter_bins(self):
        """Create logarithmically spaced diameter bins."""
        log_dmin = np.log10(self.diameter_range[0])
        log_dmax = np.log10(self.diameter_range[1])
        
        # Bin edges
        self.diameter_edges = np.logspace(log_dmin, log_dmax, self.num_bins + 1)
        
        # Bin centers (geometric mean)
        self.diameter_centers = np.sqrt(
            self.diameter_edges[:-1] * self.diameter_edges[1:]
        )
        
        # Bin widths
        self.diameter_widths = self.diameter_edges[1:] - self.diameter_edges[:-1]
    
    def create_lognormal_number_distribution(self, 
                                           params: DistributionParameters,
                                           normalize: bool = True) -> np.ndarray:
        """
        Create log-normal number size distribution.
        
        Args:
            params: Distribution parameters
            normalize: Whether to normalize to total number concentration
            
        Returns:
            Array of number concentrations for each bin
        """
        self.logger.debug(f"Creating log-normal number distribution with parameters: {params}")
        
        # Calculate number concentration in each bin using error function
        lower_bounds = self.diameter_edges[:-1]
        upper_bounds = self.diameter_edges[1:]
        
        # Normalized cumulative distribution function
        def norm_cdf(d):
            return 0.5 * (1 + erf(
                np.log(d / params.geometric_mean_diameter) / 
                (np.sqrt(2) * np.log(params.geometric_standard_deviation))
            ))
        
        # Number in each bin
        number_fractions = norm_cdf(upper_bounds) - norm_cdf(lower_bounds)
        
        if normalize:
            number_concentrations = params.number_concentration * number_fractions
        else:
            number_concentrations = number_fractions
        
        return number_concentrations
    
    def create_lognormal_mass_distribution(self, 
                                         params: DistributionParameters,
                                         normalize: bool = True) -> np.ndarray:
        """
        Create log-normal mass size distribution.
        
        Args:
            params: Distribution parameters
            normalize: Whether to normalize to total mass concentration
            
        Returns:
            Array of mass concentrations for each bin
        """
        self.logger.debug(f"Creating log-normal mass distribution with parameters: {params}")
        
        # Get number distribution
        number_dist = self.create_lognormal_number_distribution(params, normalize=False)
        
        # Calculate mass for each bin (assuming spherical particles)
        particle_volumes = (np.pi / 6) * self.diameter_centers ** 3
        particle_masses = particle_volumes * params.particle_density
        
        # Mass distribution
        mass_dist = number_dist * particle_masses
        
        if normalize:
            total_mass = np.sum(mass_dist)
            if total_mass > 0:
                # Normalize to unit total mass, then scale
                total_number = params.number_concentration
                avg_particle_volume = (np.pi / 6) * params.geometric_mean_diameter ** 3
                avg_particle_mass = avg_particle_volume * params.particle_density
                target_total_mass = total_number * avg_particle_mass
                mass_dist = mass_dist * (target_total_mass / total_mass)
        
        return mass_dist
    
    def create_modal_distribution(self, 
                                modes: List[DistributionParameters]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multi-modal size distribution.
        
        Args:
            modes: List of distribution parameters for each mode
            
        Returns:
            Tuple of (number_distribution, mass_distribution)
        """
        total_number = np.zeros(self.num_bins)
        total_mass = np.zeros(self.num_bins)
        
        for mode in modes:
            number_dist = self.create_lognormal_number_distribution(mode)
            mass_dist = self.create_lognormal_mass_distribution(mode)
            
            total_number += number_dist
            total_mass += mass_dist
        
        return total_number, total_mass
    
    def get_distribution_moments(self, 
                               distribution: np.ndarray,
                               moment_order: int = 0) -> float:
        """
        Calculate distribution moments.
        
        Args:
            distribution: Size distribution array
            moment_order: Order of the moment to calculate
            
        Returns:
            Calculated moment value
        """
        if moment_order == 0:
            return np.sum(distribution)
        else:
            weighted_sum = np.sum(distribution * (self.diameter_centers ** moment_order))
            total_concentration = np.sum(distribution)
            return weighted_sum / total_concentration if total_concentration > 0 else 0.0
    
    def get_effective_diameter(self, distribution: np.ndarray) -> float:
        """
        Calculate effective diameter of distribution.
        
        Args:
            distribution: Size distribution array
            
        Returns:
            Effective diameter in meters
        """
        # Volume-weighted mean diameter
        d3_moment = self.get_distribution_moments(distribution, 3)
        d2_moment = self.get_distribution_moments(distribution, 2)
        
        return d3_moment / d2_moment if d2_moment > 0 else 0.0
    
    def plot_distribution(self, 
                         distribution: np.ndarray,
                         title: str = "Particle Size Distribution",
                         xlabel: str = "Diameter (μm)",
                         ylabel: str = "dN/dlogDp",
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
        """
        Plot particle size distribution.
        
        Args:
            distribution: Size distribution to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to μm for plotting
        diameters_um = self.diameter_centers * 1e6
        
        # Calculate dN/dlogDp
        log_widths = np.log10(self.diameter_edges[1:]) - np.log10(self.diameter_edges[:-1])
        dn_dlogdp = distribution / log_widths
        
        ax.loglog(diameters_um, dn_dlogdp, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics to plot
        total_conc = np.sum(distribution)
        eff_diameter = self.get_effective_diameter(distribution) * 1e6
        
        stats_text = f"Total: {total_conc:.2e}\nEff. Diameter: {eff_diameter:.2f} μm"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig


class StandardDistributions:
    """
    Library of standard particle size distributions.
    
    This class provides pre-defined distribution parameters for common
    atmospheric particle types.
    """
    
    @staticmethod
    def urban_background() -> DistributionParameters:
        """Urban background aerosol distribution."""
        return DistributionParameters(
            number_concentration=1e4,  # particles/cm³
            geometric_mean_diameter=0.1e-6,  # 0.1 μm
            geometric_standard_deviation=2.0,
            particle_density=1500  # kg/m³
        )
    
    @staticmethod
    def continental_clean() -> DistributionParameters:
        """Continental clean aerosol distribution."""
        return DistributionParameters(
            number_concentration=3e3,  # particles/cm³
            geometric_mean_diameter=0.08e-6,  # 0.08 μm
            geometric_standard_deviation=1.8,
            particle_density=1400  # kg/m³
        )
    
    @staticmethod
    def marine_clean() -> DistributionParameters:
        """Marine clean aerosol distribution."""
        return DistributionParameters(
            number_concentration=1e3,  # particles/cm³
            geometric_mean_diameter=0.15e-6,  # 0.15 μm
            geometric_standard_deviation=1.6,
            particle_density=2200  # kg/m³ (sea salt)
        )
    
    @staticmethod
    def polluted_urban() -> DistributionParameters:
        """Polluted urban aerosol distribution."""
        return DistributionParameters(
            number_concentration=5e4,  # particles/cm³
            geometric_mean_diameter=0.05e-6,  # 0.05 μm
            geometric_standard_deviation=2.2,
            particle_density=1600  # kg/m³
        )
    
    @staticmethod
    def coarse_dust() -> DistributionParameters:
        """Coarse dust aerosol distribution."""
        return DistributionParameters(
            number_concentration=1e2,  # particles/cm³
            geometric_mean_diameter=2.0e-6,  # 2.0 μm
            geometric_standard_deviation=2.5,
            particle_density=2650  # kg/m³ (mineral dust)
        )


def create_standard_bins(num_bins: int = 50,
                        diameter_range: Tuple[float, float] = (1e-9, 1e-3)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create standard diameter bins for sectional models.
    
    Args:
        num_bins: Number of size bins
        diameter_range: Tuple of (min_diameter, max_diameter) in meters
        
    Returns:
        Tuple of (bin_centers, bin_weights) where weights are normalized
    """
    distribution = ParticleDistribution(num_bins, diameter_range)
    
    # Create uniform weights for bins
    weights = np.ones(num_bins) / num_bins
    
    return distribution.diameter_centers, weights


# Legacy function compatibility (marked as deprecated)
def bins(num: int, sigmag: float, dpg: float, plot: bool = False) -> List[Tuple[float, float]]:
    """
    Legacy function for creating number concentration lognorm size distribution.
    
    Args:
        num: Number of bins
        sigmag: Geometric standard deviation
        dpg: Geometric mean diameter
        plot: Whether to plot the distribution
        
    Returns:
        List of (diameter, weight) tuples
        
    .. deprecated:: 2.0
        Use ParticleDistribution.create_lognormal_number_distribution instead.
    """
    import warnings
    warnings.warn(
        "bins function is deprecated. Use ParticleDistribution.create_lognormal_number_distribution instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create distribution using new interface
    dist = ParticleDistribution(num_bins=num)
    params = DistributionParameters(
        number_concentration=1.0,
        geometric_mean_diameter=dpg,
        geometric_standard_deviation=sigmag,
        particle_density=1500
    )
    
    number_dist = dist.create_lognormal_number_distribution(params, normalize=False)
    
    if plot:
        dist.plot_distribution(number_dist, title="Legacy bins() function output")
    
    return [(dist.diameter_centers[i], number_dist[i]) for i in range(len(number_dist))]


def mbins(num: int, sigmag: float, dpg: float, dens: float, plot: bool = False) -> List[Tuple[float, float]]:
    """
    Legacy function for creating mass concentration lognorm size distribution.
    
    Args:
        num: Number of bins
        sigmag: Geometric standard deviation
        dpg: Geometric mean diameter
        dens: Particle density
        plot: Whether to plot the distribution
        
    Returns:
        List of (diameter, weight) tuples
        
    .. deprecated:: 2.0
        Use ParticleDistribution.create_lognormal_mass_distribution instead.
    """
    import warnings
    warnings.warn(
        "mbins function is deprecated. Use ParticleDistribution.create_lognormal_mass_distribution instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create distribution using new interface
    dist = ParticleDistribution(num_bins=num)
    params = DistributionParameters(
        number_concentration=1.0,
        geometric_mean_diameter=dpg,
        geometric_standard_deviation=sigmag,
        particle_density=dens
    )
    
    mass_dist = dist.create_lognormal_mass_distribution(params, normalize=False)
    
    if plot:
        dist.plot_distribution(mass_dist, title="Legacy mbins() function output", ylabel="dM/dlogDp")
    
    return [(dist.diameter_centers[i], mass_dist[i]) for i in range(len(mass_dist))]