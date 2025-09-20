#!/usr/bin/env python3
"""
Simple Iris Decision Tree Analysis
=================================

A streamlined approach to decision tree analysis focused on model generation and testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class SimpleIrisDecisionTree:
    """Simple Iris Decision Tree Analysis"""
    
    def __init__(self):
        """Initialize the analysis"""
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and display basic dataset information"""
        print("Iris Dataset Information")
        print("=" * 40)
        print(f"Samples: {self.X.shape[0]}")
        print(f"Features: {self.X.shape[1]}")
        print(f"Classes: {len(self.iris.target_names)}")
        print(f"Class names: {list(self.iris.target_names)}")
        print(f"Feature names: {list(self.iris.feature_names)}")
        
        # Create DataFrame for easy viewing
        df = pd.DataFrame(self.X, columns=self.iris.feature_names)
        df['species'] = pd.Categorical.from_codes(self.y, self.iris.target_names)
        
        print("\nFirst 5 samples:")
        print(df.head())
        
        return df
    
    def split_data(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"\nData Split Complete")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
    def train_model(self, max_depth=4, min_samples_split=2, random_state=42):
        """Train the decision tree model"""
        print(f"\nTraining Decision Tree Model")
        print(f"Parameters: max_depth={max_depth}, min_samples_split={min_samples_split}")
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed!")
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("No model found. Please train a model first.")
            return
        
        print(f"\nModel Evaluation")
        print("=" * 30)
        
        # Training accuracy
        train_pred = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_pred)
        
        # Test accuracy
        test_pred = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Cross-Validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        print(f"\nFeature Importance:")
        for i, importance in enumerate(self.model.feature_importances_):
            print(f"  {self.iris.feature_names[i]}: {importance:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, test_pred, target_names=self.iris.target_names))
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def visualize_tree(self, figsize=(15, 10)):
        """Create a simple tree visualization"""
        if self.model is None:
            print("No model found. Please train a model first.")
            return
        
        print(f"\nCreating Tree Visualization")
        
        plt.figure(figsize=figsize)
        plot_tree(self.model, 
                  feature_names=self.iris.feature_names,
                  class_names=self.iris.target_names,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title('Decision Tree for Iris Classification', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def show_confusion_matrix(self):
        """Display confusion matrix"""
        if self.model is None:
            print("No model found. Please train a model first.")
            return
        
        test_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.iris.target_names,
                    yticklabels=self.iris.target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def test_predictions(self, sample_indices=None):
        """Test the model with specific samples"""
        if self.model is None:
            print("No model found. Please train a model first.")
            return
        
        if sample_indices is None:
            # Test with first 3 samples from test set
            sample_indices = [0, 1, 2]
        
        print(f"\nTesting Model Predictions")
        print("=" * 40)
        
        for i in sample_indices:
            if i < len(self.X_test):
                sample = self.X_test[i].reshape(1, -1)
                prediction = self.model.predict(sample)[0]
                probability = self.model.predict_proba(sample)[0]
                actual = self.y_test[i]
                
                print(f"\nSample {i+1}:")
                print(f"  Features: {self.X_test[i]}")
                print(f"  Predicted: {self.iris.target_names[prediction]}")
                print(f"  Actual: {self.iris.target_names[actual]}")
                print(f"  Confidence: {probability[prediction]:.4f}")
                print(f"  Correct: {'Correct' if prediction == actual else 'Incorrect'}")
    
    def run_complete_analysis(self):
        """Run the complete simple analysis"""
        print("Starting Simple Iris Decision Tree Analysis")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Split data
        self.split_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        results = self.evaluate_model()
        
        # Test predictions
        self.test_predictions()
        
        # Create visualizations
        self.visualize_tree()
        self.show_confusion_matrix()
        
        print(f"\nAnalysis Complete!")
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        
        return self.model, results


def main():
    """Main function to run the analysis"""
    analyzer = SimpleIrisDecisionTree()
    model, results = analyzer.run_complete_analysis()
    
    # Additional testing
    print(f"\nAdditional Model Testing")
    print("=" * 40)
    
    # Test with custom data
    custom_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Typical setosa
    prediction = model.predict(custom_sample)[0]
    probability = model.predict_proba(custom_sample)[0]
    
    print(f"Custom sample: {custom_sample[0]}")
    print(f"Predicted class: {analyzer.iris.target_names[prediction]}")
    print(f"Confidence: {probability[prediction]:.4f}")


if __name__ == "__main__":
    main()
