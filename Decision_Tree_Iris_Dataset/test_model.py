#!/usr/bin/env python3
"""
Test Script for Simple Iris Decision Tree Model
==============================================

This script demonstrates how to test the trained model with various inputs.
"""

from simple_iris_decision_tree import SimpleIrisDecisionTree
import numpy as np

def test_model_accuracy():
    """Test model accuracy"""
    print("Testing Model Accuracy")
    print("=" * 30)
    
    analyzer = SimpleIrisDecisionTree()
    analyzer.load_data()
    analyzer.split_data()
    analyzer.train_model()
    results = analyzer.evaluate_model()
    
    # Assert minimum accuracy (0.85 is acceptable for this dataset)
    assert results['test_accuracy'] > 0.85, f"Model accuracy too low: {results['test_accuracy']}"
    print(f"Model accuracy test passed: {results['test_accuracy']:.4f}")
    
    return analyzer

def test_predictions():
    """Test specific predictions"""
    print("\nTesting Specific Predictions")
    print("=" * 35)
    
    analyzer = SimpleIrisDecisionTree()
    analyzer.load_data()
    analyzer.split_data()
    analyzer.train_model()
    
    # Test cases with known characteristics
    test_cases = [
        {
            'name': 'Typical Setosa',
            'features': [5.1, 3.5, 1.4, 0.2],
            'expected': 'setosa'
        },
        {
            'name': 'Typical Versicolor', 
            'features': [6.0, 2.7, 4.0, 1.3],
            'expected': 'versicolor'
        },
        {
            'name': 'Typical Virginica',
            'features': [6.5, 3.0, 5.5, 2.0],
            'expected': 'virginica'
        }
    ]
    
    correct_predictions = 0
    
    for test_case in test_cases:
        sample = np.array([test_case['features']])
        prediction_idx = analyzer.model.predict(sample)[0]
        prediction = analyzer.iris.target_names[prediction_idx]
        confidence = analyzer.model.predict_proba(sample)[0][prediction_idx]
        
        is_correct = prediction == test_case['expected']
        if is_correct:
            correct_predictions += 1
        
        print(f"\n{test_case['name']}:")
        print(f"  Features: {test_case['features']}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Predicted: {prediction}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Result: {'Correct' if is_correct else 'Incorrect'}")
    
    accuracy = correct_predictions / len(test_cases)
    print(f"\nTest Case Accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_cases)})")
    
    return analyzer

def test_model_properties():
    """Test model properties"""
    print("\nTesting Model Properties")
    print("=" * 30)
    
    analyzer = SimpleIrisDecisionTree()
    analyzer.load_data()
    analyzer.split_data()
    analyzer.train_model()
    
    # Test model exists
    assert analyzer.model is not None, "Model should be trained"
    print("Model exists")
    
    # Test model has reasonable depth
    depth = analyzer.model.get_depth()
    assert 1 <= depth <= 10, f"Tree depth should be reasonable: {depth}"
    print(f"Tree depth is reasonable: {depth}")
    
    # Test model has leaves
    leaves = analyzer.model.get_n_leaves()
    assert leaves >= 3, f"Should have at least 3 leaves for 3 classes: {leaves}"
    print(f"Number of leaves is appropriate: {leaves}")
    
    # Test feature importance
    importance = analyzer.model.feature_importances_
    assert len(importance) == 4, "Should have importance for all 4 features"
    assert np.isclose(np.sum(importance), 1.0, rtol=1e-10), f"Feature importance should sum to 1, got {np.sum(importance)}"
    print(f"Feature importance is valid: {importance}")
    print(f"Feature importance sum: {np.sum(importance):.10f}")
    
    return analyzer

def run_all_tests():
    """Run all tests"""
    print("Running All Tests for Simple Iris Decision Tree")
    print("=" * 60)
    
    try:
        # Test 1: Model Accuracy
        analyzer1 = test_model_accuracy()
        
        # Test 2: Specific Predictions
        analyzer2 = test_predictions()
        
        # Test 3: Model Properties
        analyzer3 = test_model_properties()
        
        print("\nAll Tests Passed Successfully!")
        print("=" * 40)
        print("The model is working correctly and is ready for use.")
        
        return True
        
    except AssertionError as e:
        print(f"\nTest Failed: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\nModel Testing Complete - All Systems Go!")
    else:
        print("\nSome tests failed - Please check the model implementation.")
