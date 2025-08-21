#!/usr/bin/env python3
"""
Test script for ASDP (AI Survey Data Processor) Application
Ministry of Statistics and Programme Implementation (MoSPI)
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from io import BytesIO

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class"""
    
    def setUp(self):
        """Set up test data"""
        self.processor = DataProcessor()
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'id': range(1, 11),
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            'education': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'weight': [1.0, 1.1, 1.2, 1.0, 0.9, 1.1, 1.0, 1.2, 0.8, 1.1]
        })
        
        # Add some missing values and outliers for testing
        self.test_data.loc[2, 'age'] = np.nan
        self.test_data.loc[5, 'income'] = np.nan
        self.test_data.loc[8, 'income'] = 500000  # Outlier
        
    def test_load_data(self):
        """Test data loading functionality"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            self.test_data.to_csv(tmp_file.name, index=False)
            tmp_filename = tmp_file.name
        
        try:
            # Test loading the data
            result = self.processor.load_data(tmp_filename)
            self.assertTrue(result)
            self.assertEqual(len(self.processor.data), 10)
            self.assertEqual(len(self.processor.data.columns), 5)
        finally:
            # Clean up
            os.unlink(tmp_filename)
    
    def test_detect_missing_values(self):
        """Test missing value detection"""
        self.processor.data = self.test_data
        missing_report = self.processor.detect_missing_values()
        
        # Should detect 2 missing values
        self.assertEqual(len(missing_report), 2)
        self.assertIn('age', missing_report['Column'].values)
        self.assertIn('income', missing_report['Column'].values)
    
    def test_impute_missing_values(self):
        """Test missing value imputation"""
        self.processor.data = self.test_data.copy()
        
        # Test mean imputation
        self.processor.impute_missing_values(method='mean')
        
        # Check that missing values are filled
        self.assertFalse(self.processor.data['age'].isnull().any())
        self.assertFalse(self.processor.data['income'].isnull().any())
        
        # Check that the imputed values are reasonable
        self.assertGreater(self.processor.data.loc[2, 'age'], 0)
        self.assertGreater(self.processor.data.loc[5, 'income'], 0)
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        self.processor.data = self.test_data
        outliers_report = self.processor.detect_outliers(method='iqr')
        
        # Should detect the outlier in income
        self.assertIn('income', outliers_report)
        self.assertGreater(outliers_report['income']['count'], 0)
    
    def test_handle_outliers(self):
        """Test outlier handling"""
        self.processor.data = self.test_data.copy()
        
        # Test winsorization
        self.processor.handle_outliers(method='winsorize')
        
        # Check that the extreme outlier is clipped
        max_income = self.processor.data['income'].max()
        self.assertLess(max_income, 500000)  # Should be clipped
    
    def test_apply_weights(self):
        """Test weight application"""
        self.processor.data = self.test_data
        result = self.processor.apply_weights('weight')
        
        self.assertTrue(result)
        self.assertIsNotNone(self.processor.weights)
        self.assertEqual(len(self.processor.weights), 10)
    
    def test_calculate_estimates(self):
        """Test statistical estimates calculation"""
        self.processor.data = self.test_data
        self.processor.apply_weights('weight')
        
        estimates = self.processor.calculate_estimates()
        
        # Check that estimates are calculated for numeric columns
        self.assertIn('age', estimates)
        self.assertIn('income', estimates)
        self.assertIn('education', estimates)
        
        # Check that both weighted and unweighted estimates exist
        for var, est in estimates.items():
            self.assertIn('unweighted', est)
            self.assertIn('weighted', est)
            
            # Check that estimates have required keys
            for est_type in ['unweighted', 'weighted']:
                self.assertIn('mean', est[est_type])
                self.assertIn('std', est[est_type])
                self.assertIn('se', est[est_type])
                self.assertIn('ci_95_lower', est[est_type])
                self.assertIn('ci_95_upper', est[est_type])
    
    def test_generate_visualizations(self):
        """Test visualization generation"""
        self.processor.data = self.test_data
        plots = self.processor.generate_visualizations()
        
        # Should generate some plots
        self.assertGreater(len(plots), 0)
        
        # Check that plots are HTML strings
        for plot_name, plot_html in plots.items():
            self.assertIsInstance(plot_html, str)
            self.assertIn('<div', plot_html)
    
    def test_generate_reports(self):
        """Test report generation"""
        self.processor.data = self.test_data
        self.processor.apply_weights('weight')
        self.processor.calculate_estimates()
        
        # Test HTML report
        html_report = self.processor.generate_report(format='html')
        self.assertIsInstance(html_report, str)
        self.assertIn('<!DOCTYPE html>', html_report)
        
        # Test PDF report
        pdf_report = self.processor.generate_report(format='pdf')
        self.assertIsInstance(pdf_report, BytesIO)
    
    def test_cleaning_log(self):
        """Test cleaning log functionality"""
        self.processor.data = self.test_data.copy()
        
        # Perform some operations
        self.processor.impute_missing_values(method='mean')
        self.processor.handle_outliers(method='winsorize')
        self.processor.apply_weights('weight')
        
        # Check that operations are logged
        self.assertGreater(len(self.processor.cleaning_log), 0)
        
        # Check log content
        log_text = ' '.join(self.processor.cleaning_log)
        self.assertIn('imputed', log_text.lower())
        self.assertIn('outliers', log_text.lower())
        self.assertIn('weights', log_text.lower())

def run_tests():
    """Run all tests"""
    print("Running tests for ASDP (AI Survey Data Processor) Application...")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDataProcessor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
