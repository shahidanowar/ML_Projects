# Handwritten Digit Recognition

A machine learning project that trains a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) using the MNIST dataset, with an interactive web interface for real-time digit recognition.

## Features

- **CNN Model Training**: Trains a convolutional neural network on the MNIST dataset
- **Interactive Web Interface**: Draw digits on a canvas and get real-time predictions
- **High Accuracy**: Achieves ~98% accuracy on test data
- **Real-time Visualization**: See training progress and prediction results

## Project Structure

```
├── train.py              # Model training script
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── digit_model.keras    # Trained model (generated after training)
└── README.md           # This file
```

## Installation

1. **Clone or download the project files**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

Run the training script to create and train the CNN model:

```bash
python train.py
```

This will:
- Download and preprocess the MNIST dataset
- Train a CNN model for 5 epochs
- Evaluate the model on test data
- Save the trained model as `digit_model.keras`
- Display a sample prediction with visualization

**Expected output:**
```
Test accuracy: 0.9800+
```

### Step 2: Launch the Web Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a web browser with the interactive digit recognition interface where you can:
- Draw digits on a 280x280 canvas
- Click "Predict" to get real-time recognition results
- Clear and redraw to test different digits

## Model Architecture

The CNN model consists of:

1. **Convolutional Layer**: 32 filters, 3x3 kernel, ReLU activation
2. **Max Pooling Layer**: 2x2 pooling to reduce spatial dimensions
3. **Flatten Layer**: Converts 2D feature maps to 1D vector
4. **Dense Layer**: 64 neurons with ReLU activation
5. **Output Layer**: 10 neurons with softmax activation (for digits 0-9)

## Technical Details

### Data Preprocessing
- Images normalized to range [0, 1]
- Reshaped to 28x28x1 format for grayscale processing
- Canvas drawings resized and processed to match training data format

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical crossentropy
- **Metric**: Accuracy
- **Epochs**: 5 (can be adjusted for better performance)

### Web Interface
- Built with Streamlit for easy deployment
- Uses `streamlit-drawable-canvas` for interactive drawing
- OpenCV for image processing and resizing
- Real-time prediction with confidence display

## Dependencies

- **tensorflow**: Deep learning framework for model training and inference
- **streamlit**: Web framework for the interactive interface
- **streamlit-drawable-canvas**: Canvas widget for drawing
- **numpy**: Numerical computations
- **matplotlib**: Visualization and plotting
- **opencv-python**: Image processing
- **scikit-learn**: Additional ML utilities
- **pandas**: Data manipulation

## Performance

- **Training Time**: ~2-3 minutes on CPU
- **Test Accuracy**: ~98%
- **Model Size**: ~500KB
- **Prediction Time**: <100ms per digit

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure you run `train.py` first to generate `digit_model.keras`

2. **Canvas not responding**
   - Try refreshing the browser page
   - Check that all dependencies are installed

3. **Low prediction accuracy**
   - Draw digits clearly and centered
   - Use white strokes on black background
   - Make sure digits are reasonably sized

4. **Installation issues**
   - Try updating pip: `pip install --upgrade pip`
   - Use virtual environment for clean installation

## Extending the Project

### Possible Improvements

1. **Model Enhancement**
   - Add more convolutional layers
   - Implement data augmentation
   - Add dropout for regularization
   - Experiment with different optimizers

2. **Interface Improvements**
   - Add confidence scores display
   - Show prediction probabilities for all digits
   - Add clear button for canvas
   - Implement batch prediction

3. **Additional Features**
   - Save and load custom drawings
   - Add preprocessing options
   - Implement different model architectures
   - Add model comparison features

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- TensorFlow and Keras teams for the deep learning framework
- Streamlit team for the web framework