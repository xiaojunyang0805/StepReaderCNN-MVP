# SEE_Step Analysis

This repository contains two distinct methods for step detection in Surface Electrical Stimulation (SEE) experimental signals. These methods are specifically designed for different use cases: low sampling frequency and high sampling frequency. Each method can be found in its respective folder.

## Repository Structure

Each folder in this repository includes:

- A Jupyter notebook script that runs the step detection algorithm.
- An example data file to demonstrate the usage of the script.
- A pre-trained model saved in `.keras` format.
- The corresponding training history of the model, saved in `.pkl` format.

## Requirements

Before running the scripts, make sure the following Python packages are installed in your environment:

- numpy
- scipy
- pandas
- pywt
- TensorFlow
- plotly 

## Usage

The step detection algorithms are implemented in Jupyter notebook scripts. Please be aware that currently, only CPU support is available. As a result, running the script, especially the section involving Convolutional Neural Networks (CNN), may take a significant amount of time due to CPU limitations.
