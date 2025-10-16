# StepReaderCNN MVP Workplan

## Project Overview
Build a CNN-based framework for automatic electrochemical sensor signal processing, specifically for classifying particle size (1um, 2um, 3um) from single-entity sensing data.

## Dataset Summary
- **Total Files**: 42 CSV files in TestData folder
  - PS 1um: 7 samples
  - PS 2um: 9 samples
  - PS 3um: 26 samples
- **Data Format**: Time-series (time [ms], current channel 1)
- **Sampling Rate**: ~0.8192 ms intervals
- **Task**: Multi-class classification (3 classes)

---

## MVP Phases

### Phase 1: Project Setup & Data Exploration ✅ COMPLETED
**Objective**: Set up development environment and understand data characteristics

#### Tasks:
1. **Project Structure Setup**
   - Create organized directory structure:
     ```
     StepReaderCNN/
     ├── data/              # Data storage
     │   ├── raw/           # Original data
     │   ├── processed/     # Preprocessed data
     │   └── synthetic/     # Generated data
     ├── src/               # Source code
     │   ├── data/          # Data processing
     │   ├── models/        # Model architectures
     │   ├── training/      # Training logic
     │   ├── evaluation/    # Evaluation tools
     │   ├── api/           # FastAPI backend
     │   └── gui/           # GUI components
     ├── models/            # Saved models
     ├── outputs/           # Results, plots
     ├── notebooks/         # Jupyter notebooks
     ├── tests/             # Unit tests
     ├── configs/           # Configuration files
     ├── app.py             # Streamlit GUI
     ├── requirements.txt   # Dependencies
     └── docker-compose.yml # Deployment
     ```
   - Set up Python virtual environment
   - Create comprehensive requirements.txt:
     - PyTorch, NumPy, Pandas, SciPy
     - FastAPI, Streamlit
     - Plotly, Matplotlib, Seaborn
     - pytest, black
   - Initialize git repository
   - Create configuration templates (YAML)

2. **Data Exploration & Visualization**
   - Load and parse CSV files
   - Analyze signal characteristics (length, amplitude, noise)
   - Visualize sample signals from each class
   - Compute basic statistics (mean, std, min, max per class)
   - Check for class imbalance issues
   - Determine optimal sequence length

3. **Data Quality Assessment**
   - Check for missing values or anomalies
   - Analyze signal-to-noise ratio
   - Identify outliers
   - Assess data quality per file

**Deliverables**:
- Complete project structure
- Python environment with all dependencies
- Jupyter notebook with exploratory data analysis
- Data quality report
- Initial configuration files

---

### Phase 2: Basic GUI for Data Exploration
**Objective**: Build interactive interface for data import and visualization (early GUI for immediate usability)

#### Tasks:
1. **GUI Framework Setup**
   - Set up Streamlit project structure
   - Install GUI dependencies
   - Create basic layout template
   - Configure routing and navigation

2. **Data Import Interface**
   - File upload panel (drag-and-drop)
     - Support CSV, NPY, HDF5 formats
     - Progress indicators
     - Error handling
   - Dataset overview dashboard
     - Sample count per class
     - Class distribution visualization
     - Signal statistics display

3. **Interactive Data Viewer**
   - Plot selected signals
     - Zoom/pan capabilities
     - Compare multiple samples side-by-side
   - Signal analysis tools
     - Statistics calculator
     - Quality metrics
   - Export functionality (plots, data summaries)

**Deliverables**:
- `app.py` - Streamlit GUI application (data exploration module)
- `src/gui/data_viewer.py` - Data visualization components
- `src/gui/upload_handler.py` - File upload utilities
- Interactive data exploration dashboard

---

### Phase 3: Data Preprocessing Pipeline
**Objective**: Build robust data preprocessing pipeline

#### Tasks:
1. **Data Loader Implementation**
   - CSV parsing utility
   - Train/validation/test split (stratified)
   - Data normalization/standardization options

2. **Signal Processing**
   - Noise filtering (if needed)
   - Signal alignment/synchronization
   - Length normalization (padding/truncation)
   - Feature extraction baseline (for comparison)

3. **Data Augmentation**
   - Time-series augmentation techniques:
     - Time warping
     - Magnitude scaling
     - Gaussian noise injection
     - Time shifting
   - Validation of augmentation quality

**Deliverables**:
- `src/data/preprocessing.py` - preprocessing functions
- `src/data/augmentation.py` - augmentation utilities
- Updated data loader with preprocessing integration

---

### Phase 4: CNN Model Development
**Objective**: Design and implement CNN architecture for time-series classification

#### Tasks:
1. **CNN Architecture Design**
   - 1D CNN for time-series data
   - Architecture options:
     - Simple baseline (3-4 conv layers)
     - ResNet-inspired (with skip connections)
     - Multi-scale CNN (parallel conv paths)
   - Batch normalization and dropout for regularization

2. **Model Implementation**
   - Define model class in PyTorch/TensorFlow
   - Implement forward pass
   - Add model summary/visualization

3. **Training Pipeline**
   - Loss function (CrossEntropy)
   - Optimizer setup (Adam, SGD)
   - Learning rate scheduler
   - Training loop with validation
   - Metrics tracking (accuracy, F1-score, confusion matrix)

**Deliverables**:
- `src/models/cnn_model.py` - CNN architecture
- `src/train.py` - training script
- `src/evaluate.py` - evaluation utilities

---

### Phase 5: Training GUI & Real-time Monitoring
**Objective**: Build training control interface with live metrics tracking

#### Tasks:
1. **Training Control Dashboard**
   - Model selection dropdown
   - Hyperparameter input fields (epochs, batch size, learning rate)
   - Start/stop/pause training buttons
   - Training configuration save/load

2. **Real-time Metrics Display**
   - Live loss curves (train & validation)
   - Live accuracy curves
   - Progress bars with time estimates
   - Training status panel
   - WebSocket integration for real-time updates

3. **Backend API for Training**
   - REST API endpoints:
     - `/api/models/train` - start training
     - `/api/models/stop` - stop training
     - `/api/models/list` - list available models
   - WebSocket for real-time training updates
   - Background task handling for training
   - Training state management

**Deliverables**:
- `src/gui/training_dashboard.py` - Training control interface
- `src/api/training_routes.py` - Training API endpoints
- Real-time training monitoring system
- Training configuration management

---

### Phase 6: Model Training with Real Data
**Objective**: Train CNN on actual electrochemical sensing data

#### Tasks:
1. **Baseline Training**
   - Train on real data only
   - Cross-validation (k-fold)
   - Hyperparameter tuning:
     - Learning rate
     - Batch size
     - Number of layers/filters
     - Dropout rate

2. **Performance Analysis**
   - Per-class accuracy analysis
   - Confusion matrix visualization
   - Learning curves (train vs validation)
   - Identify failure cases

3. **Model Optimization**
   - Address overfitting/underfitting
   - Class imbalance handling (weighted loss, oversampling)
   - Fine-tune architecture based on results

**Deliverables**:
- Trained baseline model checkpoint
- Training logs and metrics
- Performance analysis report
- Visualizations (learning curves, confusion matrix)

---

### Phase 7: Evaluation & Prediction Interface
**Objective**: Build interface for model evaluation and predictions

#### Tasks:
1. **Evaluation Dashboard**
   - Model selection for evaluation
   - Test set evaluation metrics
     - Accuracy, precision, recall, F1
     - Confusion matrix visualization
     - ROC curves and precision-recall curves
   - Error analysis viewer
     - Misclassified samples
     - Confidence distribution

2. **Prediction Interface**
   - Upload new data for prediction
   - Display prediction results
     - Predicted class
     - Confidence scores
     - Visualization of input signal
   - Batch prediction support
   - Export prediction results

3. **Backend API for Evaluation**
   - REST API endpoints:
     - `/api/evaluate` - run evaluation
     - `/api/predict` - make predictions
     - `/api/results/export` - export results
   - Results caching and storage

**Deliverables**:
- `src/gui/evaluation_dashboard.py` - Evaluation interface
- `src/gui/prediction_interface.py` - Prediction UI
- `src/api/evaluation_routes.py` - Evaluation API endpoints
- Results export functionality

---

### Phase 8: Synthetic Data Generation
**Objective**: Generate realistic synthetic electrochemical signals

#### Tasks:
1. **Signal Modeling**
   - Analyze real signal characteristics:
     - Baseline current levels per class
     - Noise characteristics
     - Signal patterns and transitions
   - Define parametric signal model

2. **Synthetic Data Generator**
   - Implement generation algorithm:
     - Physics-inspired model (if applicable)
     - Statistical model based on real data
     - GAN-based approach (optional, advanced)
   - Parameter randomization for diversity
   - Class-specific signal generation

3. **Synthetic Data Validation**
   - Visual comparison with real data
   - Statistical similarity tests
   - Distribution matching (KL divergence, etc.)

**Deliverables**:
- `src/synthetic_generator.py` - data generation module
- Generated synthetic dataset (configurable size)
- Validation report comparing real vs synthetic

---

### Phase 9: Model Testing & Final Integration
**Objective**: Comprehensive testing with real and synthetic data, final system integration

#### Tasks:
1. **Test Set Preparation**
   - Hold-out real data test set
   - Generated synthetic test set
   - Mixed test scenarios

2. **Model Evaluation**
   - Test on real data only
   - Test on synthetic data only
   - Test on mixed data
   - Cross-dataset validation

3. **Performance Comparison**
   - Baseline model (real data only) vs augmented model (real + synthetic)
   - Analyze where synthetic data helps/hurts
   - ROC curves and precision-recall analysis

**Deliverables**:
- Comprehensive test results
- Performance comparison report
- Recommendations for data strategy
- Complete integrated system
- Final bug fixes and optimizations

**Additional Integration Tasks**:
1. **System Integration**
   - Connect all GUI modules with backend API
   - Integrate all data processing modules
   - End-to-end workflow testing
   - Configuration management system

2. **Deployment Setup**
   - Docker containerization
     - Backend container (FastAPI + PyTorch)
     - Database container
   - Docker Compose orchestration
   - Environment configuration (.env)
   - Startup scripts

3. **Documentation**
   - User Guide
     - Installation instructions
     - Quick start tutorial
     - GUI usage guide
     - Troubleshooting
   - Developer Guide
     - Architecture overview
     - API documentation
     - Code structure
     - Adding new models
   - README with complete setup
   - Model architecture diagrams

4. **Testing & Quality**
   - Unit tests for core modules
   - Integration tests
   - GUI functionality tests
   - Code cleanup and refactoring
   - Comprehensive docstrings

5. **Demo & Examples**
   - Interactive demo notebook
   - Sample workflows
   - Example datasets
   - Results interpretation guide

**Deliverables**:
- Complete integrated system
- Docker deployment ready
- Comprehensive documentation
- User and developer guides
- Demo materials
- Clean, tested codebase

---

## Technology Stack

### Backend
- **Deep Learning**: PyTorch
- **Data Processing**: NumPy, Pandas, SciPy
- **API Framework**: FastAPI (REST API + WebSocket support)
- **Database**: SQLite (development) / PostgreSQL (production)
- **Task Queue**: FastAPI BackgroundTasks (MVP) / Celery + Redis (production)
- **Metrics**: scikit-learn

### Frontend/GUI
- **Primary (MVP)**: Streamlit (fast development, Python-only, real-time updates)
- **Alternative**: Dash (Python + Plotly, more customization)
- **Production Option**: React + Plotly + TailwindCSS (maximum flexibility)

### Visualization
- **Core**: Matplotlib, Seaborn, Plotly
- **Real-time Plots**: Plotly for interactive dashboards

### Development Tools
- **Testing**: pytest
- **Code Quality**: black, flake8
- **Experiment Tracking**: TensorBoard (built-in), Weights & Biases (optional)
- **Hyperparameter Tuning**: Optuna, Ray Tune (optional)
- **Containerization**: Docker, Docker Compose

### Optional/Advanced
- **Synthetic Data**: GANs (if pursuing generative approach)
- **Model Serving**: ONNX export for deployment

---

## Success Criteria

### Minimum Viable Product (MVP)

#### Technical Requirements
- ✓ Working CNN model trained on real data
- ✓ Achieves >70% accuracy on test set
- ✓ Synthetic data generator producing realistic signals
- ✓ Complete end-to-end pipeline demonstrated
- ✓ Functional GUI for all major operations
- ✓ Real-time training monitoring
- ✓ API backend operational
- ✓ Clear documentation and reproducible results

#### GUI Requirements
- ✓ Users can upload data through GUI without code
- ✓ Training can be initiated and monitored via dashboard
- ✓ Real-time metrics update during training
- ✓ Evaluation results displayed interactively
- ✓ Prediction interface functional
- ✓ Responsive and intuitive interface

#### Performance Requirements
- ✓ Data loading handles 42 CSV files efficiently
- ✓ GUI responds without lag during normal operations
- ✓ Training completes successfully
- ✓ API response time <500ms (excluding long operations)
- ✓ No crashes or data loss

### Stretch Goals
- Achieve >85% accuracy
- Real-time inference capability (<1 second per sample)
- Explainability/interpretability analysis (Grad-CAM, attention)
- Deployment-ready model (ONNX export, inference optimization)
- Multi-user support with authentication
- Dark mode for GUI
- Export comprehensive PDF reports

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web GUI                       │
│         (Data Viz + Training Monitor + Eval)            │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│              FastAPI Backend                             │
│      (REST API + WebSocket + Task Queue)                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│         Data Processing & Model Engine                   │
│  (Preprocessing → CNN Model → Training → Inference)     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│            SQLite Database + File Storage                │
│      (Metadata, Models, Results, Data Files)            │
└─────────────────────────────────────────────────────────┘
```

---

## Project Timeline Estimate

| Phase | Estimated Time |
|-------|---------------|
| Phase 1: Setup & EDA ✅ | 1-2 days |
| Phase 2: Basic GUI for Data Exploration | 2-3 days |
| Phase 3: Data Preprocessing Pipeline | 2-3 days |
| Phase 4: CNN Model Development | 2-3 days |
| Phase 5: Training GUI & Real-time Monitoring | 2-3 days |
| Phase 6: Model Training with Real Data | 3-4 days |
| Phase 7: Evaluation & Prediction Interface | 2-3 days |
| Phase 8: Synthetic Data Generation | 3-4 days |
| Phase 9: Testing & Final Integration | 2-3 days |
| **Total** | **19-30 days** |

---

## Risk Mitigation

### Potential Challenges
1. **Class Imbalance**: PS 3um has 3x more samples
   - *Mitigation*: Use weighted loss, SMOTE, or oversample minority classes

2. **Limited Data**: Only 42 total samples
   - *Mitigation*: Heavy data augmentation, synthetic data generation, cross-validation

3. **Signal Variability**: High noise in electrochemical signals
   - *Mitigation*: Robust preprocessing, noise-aware augmentation

4. **Overfitting**: Small dataset risk
   - *Mitigation*: Strong regularization (dropout, L2), early stopping, data augmentation

---

## GUI Framework Comparison

### Recommended: Streamlit (for MVP)
**Pros**:
- Pure Python (no frontend coding required)
- Extremely rapid development
- Built-in components for ML applications
- Real-time updates support
- Easy deployment
- Perfect for MVP and demos

**Cons**:
- Less customization than React
- Single-page architecture limitations

**Best for**: Quick MVP, Python developers, research applications

### Alternative: Dash (Python + Plotly)
**Pros**:
- Python-based with reactive components
- Excellent for data visualization
- Good balance of customization and ease
- Built on Plotly

**Cons**:
- Steeper learning curve for callbacks
- Can become complex for large apps

**Best for**: Data-heavy dashboards, analytical tools

### Production Option: React + Plotly
**Pros**:
- Maximum customization and flexibility
- Best performance
- Professional appearance
- Large ecosystem

**Cons**:
- Requires JavaScript/frontend expertise
- Longer development time
- More complex setup

**Best for**: Production applications, commercial products

---

## Quick Start Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Development
python -m src.data.preprocessing     # Preprocess data
python -m src.training.train         # Train model
python -m src.evaluation.evaluate    # Evaluate model
streamlit run app.py                 # Run GUI

# API Backend
uvicorn src.api.main:app --reload   # Run FastAPI backend

# Production
docker-compose up -d                 # Start all services
docker-compose logs -f               # View logs
docker-compose down                  # Stop services
```

---

## Next Steps
1. ✅ Phase 1 completed (Setup & Data Exploration)
2. Begin Phase 2: Basic GUI for Data Exploration
3. Implement Streamlit interface for data import and visualization
4. Enable early user interaction with the dataset through GUI
