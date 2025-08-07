#!/usr/bin/env python3
"""
Test runner script for RAG chatbot components
Runs all tests and provides detailed output to identify failing components
"""

import unittest
import sys
import os
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_single_test_module(module_name):
    """Run a single test module and return results"""
    print(f"\n{'='*60}")
    print(f"Running {module_name}")
    print('='*60)
    
    # Create a test suite for this module
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    
    # Create a test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Print the output
    output = stream.getvalue()
    print(output)
    
    return result

def run_all_tests():
    """Run all test modules and summarize results"""
    print("RAG Chatbot Component Test Suite")
    print("=" * 60)
    
    # Test modules to run
    test_modules = [
        'test_course_search_tool',
        'test_ai_generator', 
        'test_rag_system'
    ]
    
    all_results = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    # Run each test module
    for module in test_modules:
        try:
            result = run_single_test_module(module)
            all_results.append((module, result))
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
        except Exception as e:
            print(f"ERROR: Failed to run {module}: {e}")
            total_errors += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success rate: {((total_tests - total_failures - total_errors) / max(total_tests, 1)) * 100:.1f}%")
    
    # Detailed failure analysis
    if total_failures > 0 or total_errors > 0:
        print(f"\n{'='*60}")
        print("FAILURE ANALYSIS")
        print('='*60)
        
        for module_name, result in all_results:
            if result.failures or result.errors:
                print(f"\n{module_name.upper()} ISSUES:")
                print("-" * 40)
                
                for test, traceback in result.failures:
                    print(f"FAILURE: {test}")
                    print(f"Details: {traceback}")
                    print()
                
                for test, traceback in result.errors:
                    print(f"ERROR: {test}")
                    print(f"Details: {traceback}")
                    print()
    
    return total_failures == 0 and total_errors == 0


def run_specific_component_test(component):
    """Run tests for a specific component"""
    component_tests = {
        'search': 'test_course_search_tool',
        'ai': 'test_ai_generator',
        'rag': 'test_rag_system'
    }
    
    if component not in component_tests:
        print(f"Unknown component: {component}")
        print(f"Available components: {', '.join(component_tests.keys())}")
        return False
    
    result = run_single_test_module(component_tests[component])
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{component.upper()} COMPONENT TEST RESULT: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific component test
        component = sys.argv[1].lower()
        success = run_specific_component_test(component)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)