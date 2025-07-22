"""
Unit tests for deposition models.

This module contains comprehensive tests for all deposition model
implementations, utilities, and evaluation functions.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from DepoBoxTool.constants import PhysicalConstants, DefaultParameters
from DepoBoxTool.utils import validate_inputs, ensure_array, setup_logging
from DepoBoxTool.land_use import LandUseParameters, BaseLandUseParameters, StandardLandUseLibrary
from DepoBoxTool.particle_distribution import ParticleDistribution, DistributionParameters
from DepoBoxTool.evaluation import EvaluationMetrics, EvaluationResult, ModelComparison
from DepoBoxTool.models import (
    create_model, RUN_SID, RUN_MOD, PR11Model, VGLAIModel, 
    DepositionResult, MODEL_REGISTRY
)


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constants and calculations."""
    
    def test_constants_values(self):
        """Test that physical constants have expected values."""
        constants = PhysicalConstants()
        
        self.assertAlmostEqual(constants.GRAVITY, 9.8)
        self.assertAlmostEqual(constants.VON_KARMAN, 0.4)
        self.assertAlmostEqual(constants.BOLTZMANN, 1.38e-23)
        self.assertAlmostEqual(constants.AIR_DENSITY_STD, 1.225)
    
    def test_dynamic_viscosity_calculation(self):
        """Test dynamic viscosity calculation."""
        # Test at standard conditions
        visc_298 = PhysicalConstants.dynamic_viscosity_air(298)
        self.assertAlmostEqual(visc_298, 1.8e-5, delta=1e-6)
        
        # Test temperature dependence
        visc_273 = PhysicalConstants.dynamic_viscosity_air(273)
        visc_323 = PhysicalConstants.dynamic_viscosity_air(323)
        
        # Viscosity should increase with temperature
        self.assertLess(visc_273, visc_298)
        self.assertLess(visc_298, visc_323)
    
    def test_kinematic_viscosity_calculation(self):
        """Test kinematic viscosity calculation."""
        kin_visc = PhysicalConstants.kinematic_viscosity_air(298)
        dyn_visc = PhysicalConstants.dynamic_viscosity_air(298)
        
        expected = dyn_visc / PhysicalConstants.AIR_DENSITY_STD
        self.assertAlmostEqual(kin_visc, expected)
    
    def test_mean_free_path_calculation(self):
        """Test mean free path calculation."""
        mfp = PhysicalConstants.mean_free_path(298, 101325, 1e-6)
        
        # Should be on the order of 1e-7 m at standard conditions
        self.assertGreater(mfp, 1e-8)
        self.assertLess(mfp, 1e-6)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid parameters."""
        params = {
            'dens': 1500,
            'dim': 1e-6,
            'temp': 298,
            'press': 101325,
            'ustar': 0.5
        }
        
        validated = validate_inputs(**params)
        self.assertEqual(validated, params)
    
    def test_validate_inputs_invalid(self):
        """Test input validation with invalid parameters."""
        # Test invalid density
        with self.assertRaises(ValueError):
            validate_inputs(dens=50)  # Too low
        
        # Test invalid temperature
        with self.assertRaises(ValueError):
            validate_inputs(temp=100)  # Too low
        
        # Test non-numeric input
        with self.assertRaises(ValueError):
            validate_inputs(dens="invalid")
    
    def test_ensure_array(self):
        """Test array conversion utility."""
        # Test scalar
        result = ensure_array(5.0)
        np.testing.assert_array_equal(result, np.array([5.0]))
        
        # Test list
        result = ensure_array([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        
        # Test numpy array (should return unchanged)
        arr = np.array([1, 2, 3])
        result = ensure_array(arr)
        np.testing.assert_array_equal(result, arr)
    
    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            logger = setup_logging(level='DEBUG', log_file=log_file)
            
            self.assertEqual(logger.level, 10)  # DEBUG level
            self.assertTrue(os.path.exists(log_file))


class TestLandUseParameters(unittest.TestCase):
    """Test land use parameter handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.luc = LandUseParameters()
        self.base_params = BaseLandUseParameters(
            roughness_length=0.1,
            displacement_height=0.5,
            measurement_height=2.0,
            convective_velocity=1.0,
            leaf_area_index=3.0,
            characteristic_length=0.002
        )
    
    def test_base_parameters_validation(self):
        """Test base parameter validation."""
        # Valid parameters should not raise error
        params = BaseLandUseParameters(
            roughness_length=0.1,
            displacement_height=0.0,
            measurement_height=2.0,
            convective_velocity=1.0
        )
        self.assertIsInstance(params, BaseLandUseParameters)
    
    def test_get_parameters_grass_pr11(self):
        """Test grass parameters for PR11 model."""
        params = self.luc.get_parameters('grass', 'PR11', self.base_params)
        
        expected_keys = ['z0', 'd', 'z', 'wstar']
        for key in expected_keys:
            self.assertIn(key, params)
    
    def test_get_parameters_grass_vglai(self):
        """Test grass parameters for VGLAI model."""
        params = self.luc.get_parameters('grass', 'VGLAI', self.base_params)
        
        expected_keys = ['z0', 'd', 'z', 'wstar', 'LAI', 'A']
        for key in expected_keys:
            self.assertIn(key, params)
        
        self.assertEqual(params['LAI'], 3.0)
        self.assertEqual(params['A'], 0.002)
    
    def test_unsupported_combinations(self):
        """Test unsupported land use and model combinations."""
        with self.assertRaises(ValueError):
            self.luc.get_parameters('invalid_land_use', 'PR11', self.base_params)
        
        with self.assertRaises(ValueError):
            self.luc.get_parameters('grass', 'INVALID_MODEL', self.base_params)
    
    def test_standard_library(self):
        """Test standard land use library."""
        grass = StandardLandUseLibrary.grass_short()
        self.assertIsInstance(grass, BaseLandUseParameters)
        self.assertEqual(grass.roughness_length, 0.01)
        
        forest = StandardLandUseLibrary.coniferous_forest()
        self.assertIsInstance(forest, BaseLandUseParameters)
        self.assertGreater(forest.roughness_length, grass.roughness_length)


class TestParticleDistribution(unittest.TestCase):
    """Test particle distribution functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.distribution = ParticleDistribution(num_bins=20)
        self.params = DistributionParameters(
            number_concentration=1000,
            geometric_mean_diameter=0.1e-6,
            geometric_standard_deviation=2.0,
            particle_density=1500
        )
    
    def test_distribution_initialization(self):
        """Test distribution initialization."""
        self.assertEqual(self.distribution.num_bins, 20)
        self.assertEqual(len(self.distribution.diameter_centers), 20)
        self.assertEqual(len(self.distribution.diameter_edges), 21)
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        with self.assertRaises(ValueError):
            ParticleDistribution(num_bins=1)  # Too few bins
        
        with self.assertRaises(ValueError):
            ParticleDistribution(diameter_range=(1e-6, 1e-7))  # Invalid range
    
    def test_lognormal_number_distribution(self):
        """Test log-normal number distribution creation."""
        dist = self.distribution.create_lognormal_number_distribution(self.params)
        
        self.assertEqual(len(dist), 20)
        self.assertAlmostEqual(np.sum(dist), 1000, delta=1)  # Total concentration
        self.assertTrue(np.all(dist >= 0))  # All positive
    
    def test_lognormal_mass_distribution(self):
        """Test log-normal mass distribution creation."""
        mass_dist = self.distribution.create_lognormal_mass_distribution(self.params)
        
        self.assertEqual(len(mass_dist), 20)
        self.assertTrue(np.all(mass_dist >= 0))  # All positive
    
    def test_distribution_moments(self):
        """Test distribution moment calculations."""
        dist = self.distribution.create_lognormal_number_distribution(self.params)
        
        # 0th moment (total concentration)
        moment_0 = self.distribution.get_distribution_moments(dist, 0)
        self.assertAlmostEqual(moment_0, 1000, delta=1)
        
        # 1st moment
        moment_1 = self.distribution.get_distribution_moments(dist, 1)
        self.assertGreater(moment_1, 0)
    
    def test_effective_diameter(self):
        """Test effective diameter calculation."""
        dist = self.distribution.create_lognormal_number_distribution(self.params)
        eff_diameter = self.distribution.get_effective_diameter(dist)
        
        self.assertGreater(eff_diameter, 0)
        self.assertLess(eff_diameter, 1e-3)  # Reasonable size range


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics and statistics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = EvaluationMetrics()
        
        # Create test data
        np.random.seed(42)
        self.observed = np.random.lognormal(0, 0.5, 100)
        self.predicted = self.observed * np.random.lognormal(0, 0.2, 100)
    
    def test_basic_statistics(self):
        """Test basic statistical calculations."""
        summary = self.evaluator.basic_statistics(self.observed, self.predicted)
        
        # Check that all expected metrics are present
        expected_metrics = ['N', 'Mean_Obs', 'Mean_Pred', 'Std_Obs', 'Std_Pred', 
                           'Median_Obs', 'Median_Pred']
        
        for metric in expected_metrics:
            result = summary.get_result(metric)
            self.assertIsNotNone(result)
            self.assertIsInstance(result.value, (int, float))
    
    def test_bias_metrics(self):
        """Test bias calculations."""
        summary = self.evaluator.bias_metrics(self.observed, self.predicted)
        
        # Check for expected bias metrics
        expected_metrics = ['MB', 'NMB', 'MNB', 'MdnB', 'MdnNB']
        
        for metric in expected_metrics:
            result = summary.get_result(metric)
            self.assertIsNotNone(result)
    
    def test_error_metrics(self):
        """Test error calculations."""
        summary = self.evaluator.error_metrics(self.observed, self.predicted)
        
        # Check for expected error metrics
        expected_metrics = ['RMSE', 'MAE', 'MNGE', 'MdnNGE']
        
        for metric in expected_metrics:
            result = summary.get_result(metric)
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.value, 0)  # Errors should be non-negative
    
    def test_correlation_metrics(self):
        """Test correlation calculations."""
        summary = self.evaluator.correlation_metrics(self.observed, self.predicted)
        
        # Check correlation metrics
        r_result = summary.get_result('R')
        r2_result = summary.get_result('R2')
        
        self.assertIsNotNone(r_result)
        self.assertIsNotNone(r2_result)
        
        # R2 should be R squared
        self.assertAlmostEqual(r2_result.value, r_result.value ** 2, places=6)
        
        # Correlation should be between -1 and 1
        self.assertGreaterEqual(r_result.value, -1)
        self.assertLessEqual(r_result.value, 1)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        summary = self.evaluator.comprehensive_evaluation(self.observed, self.predicted)
        
        # Should have multiple metrics
        self.assertGreater(len(summary.results), 10)
        
        # Should have metadata
        self.assertIn('n_observations', summary.metadata)
        self.assertIn('obs_range', summary.metadata)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        # Test with NaN values
        obs_with_nan = np.array([1, 2, np.nan, 4, 5])
        pred_with_nan = np.array([1.1, 2.1, 3.1, 4.1, np.nan])
        
        summary = self.evaluator.basic_statistics(obs_with_nan, pred_with_nan)
        n_result = summary.get_result('N')
        
        # Should only count valid pairs
        self.assertEqual(n_result.value, 3)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched array lengths."""
        obs = np.array([1, 2, 3])
        pred = np.array([1, 2])  # Different length
        
        with self.assertRaises(ValueError):
            self.evaluator.basic_statistics(obs, pred)


class TestDepositionModels(unittest.TestCase):
    """Test deposition model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.land_use_params = {
            'z0': 0.1,
            'd': 0.5,
            'z': 2.0,
            'wstar': 1.0,
            'LAI': 3.0,
            'A': 0.002
        }
        
        self.model_params = {
            'particle_density': 1500,
            'particle_diameter': 2.5e-6,
            'temperature': 298,
            'pressure': 101325,
            'relative_humidity': 0.8,
            'friction_velocity': 0.5,
            'monin_obukhov_length': 50,
            'land_use_params': self.land_use_params
        }
    
    def test_model_factory(self):
        """Test model creation factory."""
        # Test valid models
        for model_name in MODEL_REGISTRY.keys():
            model = create_model(model_name, **self.model_params)
            self.assertIsNotNone(model)
            self.assertEqual(model.model_name, model_name)
        
        # Test invalid model
        with self.assertRaises(ValueError):
            create_model('INVALID_MODEL', **self.model_params)
    
    def test_pr11_model(self):
        """Test PR11 model implementation."""
        model = PR11Model(**self.model_params)
        result = model.calculate_deposition_velocity()
        
        self.assertIsInstance(result, DepositionResult)
        self.assertGreater(result.deposition_velocity, 0)
        self.assertGreater(result.gravitational_velocity, 0)
        self.assertGreater(result.aerodynamic_resistance, 0)
        self.assertGreater(result.surface_resistance, 0)
    
    def test_vglai_model(self):
        """Test VGLAI model implementation."""
        model = VGLAIModel(**self.model_params)
        result = model.calculate_deposition_velocity()
        
        self.assertIsInstance(result, DepositionResult)
        self.assertGreater(result.deposition_velocity, 0)
        
        # Test collection efficiencies
        efficiencies = model.calculate_collection_efficiencies()
        self.assertIn('brownian', efficiencies)
        self.assertIn('impaction', efficiencies)
        self.assertIn('stokes_number', efficiencies)
    
    def test_run_sid_interface(self):
        """Test RUN_SID interface."""
        runner = RUN_SID(**self.model_params)
        
        # Test individual model runs
        pr11_result = runner.PR11()
        self.assertEqual(len(pr11_result), 4)  # [Vd, Vg, Ra, Rs]
        self.assertTrue(all(x > 0 for x in pr11_result))
        
        vglai_result = runner.VGLAI()
        self.assertEqual(len(vglai_result), 4)
        self.assertTrue(all(x > 0 for x in vglai_result))
    
    def test_model_consistency(self):
        """Test consistency between different interfaces."""
        # Create model directly
        direct_model = PR11Model(**self.model_params)
        direct_result = direct_model.calculate_deposition_velocity()
        
        # Create via RUN_SID interface
        runner = RUN_SID(**self.model_params)
        interface_result = runner.PR11()
        
        # Results should be consistent
        self.assertAlmostEqual(direct_result.deposition_velocity, interface_result[0], places=10)
        self.assertAlmostEqual(direct_result.gravitational_velocity, interface_result[1], places=10)
    
    def test_parameter_sensitivity(self):
        """Test model sensitivity to parameter changes."""
        base_model = PR11Model(**self.model_params)
        base_result = base_model.calculate_deposition_velocity()
        
        # Test particle size sensitivity
        large_particle_params = self.model_params.copy()
        large_particle_params['particle_diameter'] = 10e-6
        
        large_model = PR11Model(**large_particle_params)
        large_result = large_model.calculate_deposition_velocity()
        
        # Larger particles should have higher settling velocity
        self.assertGreater(large_result.gravitational_velocity, base_result.gravitational_velocity)
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        invalid_params = self.model_params.copy()
        invalid_params['temperature'] = -100  # Invalid temperature
        
        with self.assertRaises(ValueError):
            PR11Model(**invalid_params)


class TestModelComparison(unittest.TestCase):
    """Test model comparison utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparison = ModelComparison()
        
        # Create synthetic data
        np.random.seed(42)
        self.observed = np.random.lognormal(0, 0.5, 50)
        
        # Create "predictions" from different models
        self.predictions = {
            'Model_A': self.observed * np.random.lognormal(0, 0.1, 50),
            'Model_B': self.observed * np.random.lognormal(0.1, 0.2, 50),
            'Model_C': self.observed * np.random.lognormal(-0.1, 0.15, 50)
        }
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        comparison_df = self.comparison.compare_models(self.observed, self.predictions)
        
        # Check structure
        self.assertEqual(len(comparison_df), 3)  # 3 models
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('R2', comparison_df.columns)
        self.assertIn('RMSE', comparison_df.columns)
        
        # Check that all models are present
        models = set(comparison_df['Model'].values)
        expected_models = set(self.predictions.keys())
        self.assertEqual(models, expected_models)
    
    def test_rank_models(self):
        """Test model ranking functionality."""
        ranked_df = self.comparison.rank_models(
            self.observed, self.predictions, primary_metric='R2', ascending=False
        )
        
        # Check structure
        self.assertIn('Rank', ranked_df.columns)
        self.assertEqual(len(ranked_df), 3)
        
        # Check ranking order
        ranks = ranked_df['Rank'].values
        np.testing.assert_array_equal(ranks, [1, 2, 3])
        
        # Check that R2 values are in descending order
        r2_values = ranked_df['R2'].values
        self.assertTrue(np.all(r2_values[:-1] >= r2_values[1:]))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test a complete deposition calculation workflow."""
        # Set up parameters
        land_use = LandUseParameters()
        base_params = StandardLandUseLibrary.grass_short()
        
        # Get land use parameters for different models
        pr11_params = land_use.get_parameters('grass', 'PR11', base_params)
        vglai_params = land_use.get_parameters('grass', 'VGLAI', base_params)
        
        # Create models
        common_params = {
            'particle_density': 1500,
            'particle_diameter': 1e-6,
            'temperature': 298,
            'pressure': 101325,
            'relative_humidity': 0.7,
            'friction_velocity': 0.3,
            'monin_obukhov_length': 100
        }
        
        pr11_model = create_model('PR11', land_use_params=pr11_params, **common_params)
        vglai_model = create_model('VGLAI', land_use_params=vglai_params, **common_params)
        
        # Calculate deposition velocities
        pr11_result = pr11_model.calculate_deposition_velocity()
        vglai_result = vglai_model.calculate_deposition_velocity()
        
        # Both should produce valid results
        self.assertGreater(pr11_result.deposition_velocity, 0)
        self.assertGreater(vglai_result.deposition_velocity, 0)
        
        # Results should be different (different models)
        self.assertNotEqual(pr11_result.deposition_velocity, vglai_result.deposition_velocity)
    
    def test_evaluation_workflow(self):
        """Test evaluation workflow with synthetic data."""
        # Create synthetic observational data
        np.random.seed(42)
        true_vd = np.random.lognormal(-3, 0.5, 100)  # cm/s
        
        # Simulate model predictions with some bias and error
        model_pred = true_vd * np.random.lognormal(0.1, 0.3, 100)
        
        # Evaluate model performance
        evaluator = EvaluationMetrics()
        summary = evaluator.comprehensive_evaluation(true_vd, model_pred)
        
        # Check that evaluation produces reasonable results
        r2_result = summary.get_result('R2')
        rmse_result = summary.get_result('RMSE')
        
        self.assertIsNotNone(r2_result)
        self.assertIsNotNone(rmse_result)
        self.assertGreater(r2_result.value, 0)  # Should have some correlation
        self.assertGreater(rmse_result.value, 0)  # Should have some error


if __name__ == '__main__':
    unittest.main()