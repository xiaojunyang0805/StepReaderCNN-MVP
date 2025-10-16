# StepReaderCNN - User Guide

**Version**: 1.0
**Last Updated**: October 16, 2025

A comprehensive CNN-based framework for electrochemical sensor signal processing with interactive GUI.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Explorer](#data-explorer)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Synthetic Data Generation](#synthetic-data-generation)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

1. **Clone the repository** (or extract files):
   ```bash
   cd StepReaderCNN
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the GUI**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   - Open your browser to: `http://localhost:8501`
   - The GUI will launch automatically

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional (CUDA-enabled GPU for faster training)

---

## Data Explorer

### Loading Data

**Option 1: Load from TestData Folder**
1. Navigate to **Data Explorer** → **Data Import**
2. Click **"Load from TestData"**
3. Wait for loading to complete
4. View summary statistics

**Option 2: Upload Custom Files**
1. Navigate to **Data Explorer** → **Data Import** → **Upload Files**
2. Drag and drop CSV files (or click to browse)
3. Supported format: Two columns (`Time (ms)`, `Current (Channel 1)`)
4. Files are automatically labeled based on filename (e.g., "PS 1um 01.csv" → label "1um")

### Exploring Data

**Dataset Overview**:
- View total samples, number of classes, average signal length
- Class distribution bar chart and pie chart
- Detailed statistics table by class
- Signal length distribution box plots

**Signal Viewer**:
- **Select Class**: Choose particle size (1um, 2um, 3um)
- **Navigate Samples**: Use ◀ Prev / Next ▶ buttons to browse signals
- **Display Options**:
  - Toggle grid lines
  - Show/hide markers
  - Downsample for faster rendering
- **Compare Mode**: View multiple samples simultaneously (up to 5)

**Signal Statistics**:
- Length and duration
- Mean and standard deviation
- Min/max values
- Sampling rate
- Peak-to-peak amplitude

---

## Model Training

### Configuration Tab

**Model Settings**:
- **Architecture**: Choose from SimpleCNN1D, ResNet1D, or MultiScaleCNN1D
  - SimpleCNN1D: Lightweight, fast training (135K parameters)
  - ResNet1D: Best balance, skip connections (964K parameters)
  - MultiScaleCNN1D: Highest capacity, multi-scale (2.1M parameters)
- **Base Filters**: 16, 32, 64, or 128 (higher = more capacity)
- **Dropout Rate**: 0.0-0.8 (0.5 recommended for regularization)

**Training Settings**:
- **Batch Size**: 4, 8, 16, 32, or 64 (8 recommended for small datasets)
- **Number of Epochs**: 1-200 (50 recommended, early stopping will apply)
- **Learning Rate**: 0.0001-0.01 (0.001 recommended)

**Advanced Settings**:
- **Early Stopping**: Stop when validation stops improving (patience: 10 epochs)
- **Class Weights**: Handle imbalanced datasets automatically
- **Signal Length**: Target length for normalization (10,000 default)
- **Normalization**: zscore, minmax, or robust
- **Data Augmentation**: Apply augmentations to training set

**Data Split**:
- Train: 70% (recommended)
- Validation: 15% (recommended)
- Test: 15% (recommended)

### Training Monitor Tab

Real-time visualization during training:
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Training and validation accuracy over epochs
- **Learning Rate Schedule**: Visualize LR reduction on plateau
- **Overfitting Indicator**: Loss difference between train/val

**Metrics Cards**:
- Final train/val loss and accuracy
- Delta from best performance

### Results Tab

**Test Set Evaluation**:
1. Click **"Evaluate on Test Set"** after training
2. View overall metrics:
   - Accuracy
   - Macro F1-Score
   - Precision
   - Recall

**Per-Class Metrics**:
- Table with accuracy, precision, recall, F1-score, and support for each class
- Sortable by any column

**Confusion Matrix**:
- Interactive heatmap showing predicted vs actual labels
- Hover to see exact counts

**Save Model**:
1. Enter custom model name (e.g., "best_model_1um_2um_3um")
2. Click **"Save Model"**
3. Model saved to `outputs/trained_models/`

---

## Model Evaluation

### Model Selection Tab

1. Click **"Load All Models"** to discover trained models
2. View model cards with:
   - Model class and parameters
   - Best validation accuracy
   - Test accuracy
   - Device (CPU/CUDA)
3. Click **"Use for Prediction"** to set active model

### Prediction Tab

**Load from TestData**:
1. Select active model from dropdown
2. Click **"Load TestData"**
3. Select class and sample
4. Click **"Predict"**
5. View results:
   - Predicted class
   - Confidence percentage
   - Correct/incorrect indicator
   - Class probability bar chart
   - Signal visualization with prediction overlay

**Upload Custom Signal**:
1. Select active model
2. Upload CSV file (same format as training data)
3. View signal plot
4. Click **"Predict"**
5. View prediction results

### Model Comparison Tab

1. Ensure multiple models are loaded
2. Click **"Run Comparison on Test Set"**
3. Wait for evaluation to complete
4. View comparison table:
   - Model name, parameters
   - Test accuracy, Macro F1, Precision, Recall
5. View performance bar charts (Accuracy and F1-Score)

---

## Synthetic Data Generation

### Generate Signals Tab

**Single Signal Generation**:
1. Select class (1um, 2um, 3um)
2. Choose signal parameters:
   - **Use Typical Length**: Auto-select realistic length
   - **Signal Length**: Manual override (10,000-500,000 points)
   - **Noise Level**: 0.0-2.0 (1.0 = typical noise from real data)
3. Optional features:
   - **Manual Step Count**: Enable to control collision steps (1-15)
     - When OFF: Automatically generates 3-6 random steps
   - **Add Baseline Drift**: Slow sinusoidal baseline variation
   - **Add Random Spikes**: Random impulse artifacts
4. Click **"Generate Signal"**
5. Preview appears in Preview & Analysis tab
6. Optional: Save signal to file

**Class Statistics**:
- View parameters extracted from real data (mean, std, typical length)

### Preview & Analysis Tab

After generating a signal:
- **Time Domain Plot**: Interactive plot with zoom/pan
- **Frequency Domain Plot**: FFT analysis (log-log scale)
- **Signal Statistics**: Length, duration, mean, std
- **Comparison with Real Data**: Table comparing generated vs real stats

### Batch Generation Tab

**Balance Dataset Mode**:
1. Click **"Load Real Data (TestData)"** if not loaded
2. View current distribution
3. Enter **"Target Samples per Class"** (e.g., 50)
4. View auto-calculated synthetic samples needed
5. Optional: Configure advanced parameters (noise range, drift, spikes)
6. Click **"Generate Balanced Dataset"**
7. Samples automatically saved to `data/synthetic/`

**Custom Batch Mode**:
1. Enter counts for each class (1um, 2um, 3um)
2. Configure advanced parameters
3. Click **"Generate Batch"**
4. View results summary

---

## Advanced Features

### Keyboard Shortcuts

- **Data Explorer**:
  - Left Arrow: Previous sample
  - Right Arrow: Next sample

### Performance Tips

1. **Large Datasets**: Use downsampling in Signal Viewer
2. **Training Speed**: Use smaller batch sizes for limited RAM
3. **GPU Training**: Install CUDA-enabled PyTorch for 10-50x speedup
4. **Synthetic Data**: Generate synthetic data to balance classes before training

### File Formats

**Input CSV Format**:
```csv
Time (ms),Current (Channel 1)
0.0,1.234
0.8192,1.235
1.6384,1.236
...
```

**Model Checkpoint Format**:
- Binary `.pth` files containing:
  - Model state dictionary
  - Model configuration
  - Label mapping
  - Training history
  - Test metrics

---

## Troubleshooting

### Common Issues

**"No data loaded" Warning**:
- Solution: Load data in Data Explorer → Data Import first

**Training Very Slow**:
- Solution 1: Reduce batch size
- Solution 2: Reduce signal length (e.g., 5000 instead of 10000)
- Solution 3: Use SimpleCNN1D instead of larger models
- Solution 4: Install CUDA-enabled PyTorch

**Out of Memory Error**:
- Solution 1: Reduce batch size (try 4 or 8)
- Solution 2: Reduce signal length
- Solution 3: Close other applications

**Model Not Found**:
- Solution: Train a model first or check `outputs/trained_models/` directory

**Synthetic Signals Look Wrong**:
- Check noise level (should be around 1.0)
- Verify class statistics match real data
- Ensure "Manual Step Count" is OFF for automatic generation

### Error Messages

**StreamlitDuplicateElementId**:
- This should not occur in the latest version
- If it does, restart the Streamlit server

**PyTorch Version Warnings**:
- Install PyTorch 2.6+ for best compatibility
- Use `weights_only=False` when loading models

### Getting Help

1. Check this user guide
2. Review `Dev_note.md` for technical details
3. Check GitHub issues (if repository is public)
4. Contact the development team

---

## Best Practices

### Data Collection

- Collect at least 20 samples per class (50+ recommended)
- Ensure consistent sampling rate
- Remove obviously corrupted signals before training

### Training

- Start with SimpleCNN1D for quick baseline
- Use early stopping to prevent overfitting
- Enable data augmentation for small datasets
- Monitor validation curves for overfitting

### Model Selection

- SimpleCNN1D: Production deployment, limited resources
- ResNet1D: Best accuracy, recommended for most cases
- MultiScaleCNN1D: Complex signals, research applications

### Synthetic Data

- Use to balance class distribution
- Generate 2-3x synthetic samples to match smallest class
- Validate synthetic signals visually before training

---

## Glossary

- **Collision Steps**: Stochastic downward current decreases caused by nanoparticle collisions
- **Single-Entity Electrochemistry**: Technique for detecting individual nanoparticle collision events
- **Signal Plateau**: Flat region between collision steps
- **Stochastic**: Random, probabilistic nature of collision timing
- **Staircase Pattern**: Descending step pattern characteristic of collision signals

---

*For technical documentation, see `API_DOCUMENTATION.md`*
*For deployment instructions, see `DEPLOYMENT_GUIDE.md`*
