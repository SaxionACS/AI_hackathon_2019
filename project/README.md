# Predicting Respiratory Rate from ECG & PPG signals

The program in this repository aims to predict a continuous respiratory signal from ECG (electrocardiogram) and PLETH (plethysmogram) signals.

To reproduce the data you'll need:

Python 3.7 with the following libraries installed (and their dependencies):

- TensorFlow (2.0.0)
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn


Run the file main.py and select one of the options:

l - to load a pre-trained model
t - to train a model on one of the data sets
e - to evaluate a trained or loaded model on one of the data sets

Always either load a train a model before evaluating it.

When you evaluate a model you'll see the plots of measured (blue) and model-predicted (red) Respiratory signals.
Two different plots are produced, one with a 2-minute long section with both the input and the output signals visible; 
another with just a 30-second long measured and predicted signal.

The model used for this project is a 4-layer Recurrent Neural Network build with Keras on top of TensorFlow

