# MVP Work Plan: 1D CNN Sensor Analysis with GUI
## AI-Assisted Development with Claude + Modern Interface

---

## Project Overview
**Project Name**: 1D CNN Sensor Data Classification System  
**Goal**: Build an AI-powered system with modern GUI for analyzing capacitance-time sensor data  
**Timeline**: 6-8 weeks  
**Dataset**: Start with 100-500 samples (>6000 signals per measurement)  
**Development Approach**: Claude-assisted development with interactive GUI

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Modern Web GUI                        │
│  (React/Streamlit/Gradio - Real-time Visualization)    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│              Backend API Layer                           │
│           (FastAPI/Flask - REST API)                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│         Data Processing & Model Engine                   │
│  (Preprocessing → CNN Model → Inference → Results)      │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Project Setup & Architecture (Week 1)

### 1.1 Environment Setup (Day 1)

**Task**: Set up complete development environment

**Claude Prompt**:
```
I'm starting a 1D CNN project for sensor data analysis. Please help me:

1. Create a complete project folder structure for:
   - Data processing pipeline
   - Multiple CNN model architectures
   - Training and evaluation modules
   - API backend
   - Frontend GUI components
   - Testing and documentation

2. Generate a comprehensive requirements.txt that includes:
   - PyTorch for deep learning
   - Data processing libraries (numpy, pandas, scipy)
   - API framework (FastAPI recommended)
   - Visualization tools (matplotlib, seaborn, plotly)
   - GUI framework (suggest best option for real-time data viz)
   - Development tools (pytest, black, tensorboard)

3. Create setup instructions for:
   - Virtual environment
   - GPU configuration (if available)
   - Development vs production settings

Please provide the complete folder structure and requirements.txt content.
```

**Deliverables**:
- ✅ Project structure created
- ✅ Dependencies installed
- ✅ Environment configured

---

### 1.2 Data Format & Documentation (Day 2)

**Task**: Define data formats and create documentation templates

**Claude Prompt**:
```
For my sensor data analysis project, I need to standardize data formats:

Project specs:
- 6000+ signals per measurement
- Time series data at 1ms sampling rate
- Data can span hours
- Need to handle both raw and preprocessed data

Please create:

1. Data format specifications document that defines:
   - Raw data structure (recommend efficient format: HDF5, NPY, or Parquet)
   - Metadata schema (sample info, timestamps, labels, measurement conditions)
   - Preprocessed data format
   - API data exchange format (JSON schema)

2. README template with sections:
   - Project overview
   - Data format guide
   - Quick start instructions
   - API documentation structure

3. Configuration file template (YAML or JSON) for:
   - Model hyperparameters
   - Preprocessing settings
   - Path configurations
   - GUI settings

Make it production-ready and easy to understand.
```

**Deliverables**:
- ✅ Data format standardized
- ✅ Documentation templates created
- ✅ Configuration system designed

---

### 1.3 GUI Technology Selection (Day 3)

**Task**: Choose and set up GUI framework

**Claude Prompt**:
```
I need a modern GUI for a sensor data analysis application with these requirements:

Requirements:
- Real-time data visualization (plot 6000+ signals efficiently)
- Interactive data import/export
- Live parameter monitoring during model training
- Dashboard showing dataset insights (statistics, distributions, quality metrics)
- Model evaluation results visualization
- Cross-platform (web-based preferred)
- Professional appearance

Please recommend:

1. Best GUI framework option between:
   - Streamlit (fast development)
   - Gradio (ML-focused)
   - React + D3.js/Plotly (most flexible)
   - Dash (Python-based interactive)

2. For your recommended option, provide:
   - Pros and cons for this specific use case
   - Setup instructions
   - Basic project structure for GUI components
   - Performance considerations for handling large time-series data
   - Best practices for real-time updates

3. Suggest optimal architecture for:
   - Frontend-backend communication
   - Real-time data streaming
   - Asynchronous task handling (training updates)

Include starter code structure showing how to organize GUI components.
```

**Deliverables**:
- ✅ GUI framework selected
- ✅ GUI project structure created
- ✅ Basic UI template ready

---

## Phase 2: Data Pipeline & GUI - Data Management (Week 2)

### 2.1 Data Preprocessing Module (Day 1-2)

**Task**: Build complete data preprocessing pipeline

**Claude Prompt**:
```
Create a comprehensive data preprocessing module for 1D sensor time-series data:

Context:
- Input: 6000 signals, 1ms sampling rate, hours of data per sample
- Need efficient processing for large datasets
- Must be configurable and reusable

Please develop:

1. A preprocessing pipeline class that includes:
   - Downsampling (configurable factor: 10x, 100x, 1000x)
   - Normalization methods (z-score, min-max, robust)
   - Sequence truncation/padding to fixed length
   - Data splitting (train/val/test)
   - Memory-efficient loading for large files

2. Data quality checking functions:
   - Missing value detection
   - Outlier detection
   - Signal-to-noise ratio calculation
   - Data drift detection

3. Save/load functionality for:
   - Preprocessing statistics (mean, std, etc.)
   - Processed datasets
   - Transformation history

4. Command-line interface for preprocessing:
   - Input/output paths
   - Configuration options
   - Progress tracking

Make it modular, well-documented, and include error handling.
Provide complete implementation with type hints and docstrings.
```

**Deliverables**:
- ✅ Preprocessing module implemented
- ✅ Data quality checks included
- ✅ CLI tool created

---

### 2.2 GUI - Data Import/Export Interface (Day 3-4)

**Task**: Build data management interface

**Claude Prompt**:
```
Create a modern GUI interface for data import/export and exploration:

Required features:

1. Data Import Panel:
   - Drag-and-drop file upload
   - Support multiple formats (CSV, NPY, HDF5, MAT)
   - Batch import capability
   - Data validation on upload
   - Preview raw data structure
   - Progress indicator for large files

2. Data Export Panel:
   - Export preprocessed data
   - Download model predictions
   - Export visualizations (PNG, PDF, SVG)
   - Batch export functionality
   - Format selection

3. Dataset Overview Dashboard:
   - Total samples count
   - Class distribution (pie chart)
   - Signal count and sequence length
   - Data size and memory usage
   - Missing data summary
   - Quality score indicators

4. Interactive Data Viewer:
   - Plot selected signals (handle 6000+ efficiently - use sampling/selection)
   - Zoom and pan capabilities
   - Select time ranges
   - Compare multiple samples side-by-side
   - Signal selection/filtering

5. Data Statistics Panel:
   - Mean, std, min, max per signal
   - Distribution histograms
   - Correlation heatmaps (for selected signals)
   - Time-domain statistics

Please provide:
- Component structure
- State management approach
- Efficient rendering strategy for large data
- Complete implementation
- Responsive design considerations

Use modern UI/UX principles with clean, professional styling.
```

**Deliverables**:
- ✅ Data import/export interface
- ✅ Interactive data viewer
- ✅ Dataset statistics dashboard

---

### 2.3 Real-time Data Insights (Day 5)

**Task**: Implement live data analysis features

**Claude Prompt**:
```
Develop real-time data insights and monitoring system:

Features needed:

1. Live Data Quality Dashboard:
   - Signal quality metrics (updated as data loads)
   - Anomaly detection indicators
   - Class balance visualization
   - Data completeness percentage
   - Processing status (idle/processing/complete)

2. Exploratory Analysis Tools:
   - Auto-generated summary statistics
   - Distribution analysis plots
   - Pattern detection (trends, seasonality)
   - Feature importance preview (if applicable)
   - Correlation analysis

3. Data Comparison Features:
   - Compare statistics across different datasets
   - Before/after preprocessing comparison
   - Class-wise analysis
   - Signal-wise analysis

4. Alert System:
   - Warn about data quality issues
   - Flag imbalanced classes
   - Notify about missing values
   - Suggest preprocessing steps

5. Export Reports:
   - Generate PDF summary report
   - Include key visualizations
   - Statistical summaries
   - Recommendations

Implement efficient computation that doesn't block the UI.
Use background workers or async processing.
Provide progress indicators and cancellation options.
```

**Deliverables**:
- ✅ Live insights dashboard
- ✅ Data quality monitoring
- ✅ Automated report generation

---

## Phase 3: CNN Model Development (Week 3)

### 3.1 Model Architecture Implementation (Day 1-3)

**Task**: Implement multiple CNN architectures

**Claude Prompt**:
```
Create a comprehensive 1D CNN model library for time-series classification:

Requirements:
- Input: (batch_size, 6000 signals, variable sequence_length)
- Output: Classification probabilities
- Must be memory-efficient
- Support transfer learning

Please implement these three architectures:

1. **Baseline SimpleCNN1D**:
   - 3-4 convolutional blocks
   - Batch normalization
   - Max pooling
   - Global average pooling
   - Dropout for regularization
   - ~5-10M parameters

2. **ResNet1D**:
   - Residual blocks with skip connections
   - Progressive channel expansion
   - Better for deeper networks
   - ~10-20M parameters

3. **Multi-scale CNN1D**:
   - Parallel branches with different kernel sizes (3, 7, 15)
   - Captures short, medium, and long-term patterns
   - Feature concatenation
   - ~15-25M parameters

For each model provide:
- Complete class implementation
- Forward pass logic
- Parameter count calculation
- Input/output shape documentation
- Initialization strategies
- Model summary function

Also create:
- Model factory function (get_model by name)
- Model configuration dataclass
- Model comparison utilities
- Save/load utilities with metadata

Include proper error handling and input validation.
```

**Deliverables**:
- ✅ Three CNN architectures implemented
- ✅ Model utilities created
- ✅ Models tested and validated

---

### 3.2 Training Pipeline (Day 4-5)

**Task**: Build robust training system

**Claude Prompt**:
```
Develop a production-ready training pipeline for CNN models:

Core Requirements:

1. **Trainer Class** with:
   - Training loop with progress tracking
   - Validation during training
   - Automatic checkpoint saving (best & latest)
   - Early stopping with patience
   - Learning rate scheduling
   - Mixed precision training (optional)
   - Gradient clipping
   - Training history tracking

2. **Logging & Monitoring**:
   - TensorBoard integration
   - Console logging with rich formatting
   - Metrics tracking (loss, accuracy, precision, recall, F1)
   - Learning rate tracking
   - Training time estimation

3. **Configuration System**:
   - Training hyperparameters (epochs, batch size, LR)
   - Model selection
   - Data augmentation settings
   - Optimizer options (Adam, SGD, AdamW)
   - Scheduler options (ReduceLROnPlateau, CosineAnnealing)

4. **Experiment Management**:
   - Automatic experiment naming with timestamps
   - Configuration saving (reproducibility)
   - Model checkpointing strategy
   - Resume training capability
   - Multi-GPU support (if available)

5. **CLI Interface**:
   - Easy command-line training
   - Argument parsing
   - Configuration file support
   - Override parameters

Please provide complete implementation with:
- Clean code structure
- Type hints
- Comprehensive docstrings
- Error handling
- Progress bars (tqdm)
- Example usage

Make it flexible and extensible for future improvements.
```

**Deliverables**:
- ✅ Training pipeline implemented
- ✅ Experiment management system
- ✅ CLI training interface

---

## Phase 4: GUI - Training Monitor & Evaluation (Week 4)

### 4.1 Real-time Training Dashboard (Day 1-3)

**Task**: Build live training monitoring interface

**Claude Prompt**:
```
Create a comprehensive real-time training monitoring dashboard:

Dashboard Components:

1. **Training Control Panel**:
   - Model selection dropdown
   - Hyperparameter inputs (batch size, learning rate, epochs)
   - Data augmentation toggles
   - Start/Stop/Pause training buttons
   - Save checkpoint button
   - Load previous checkpoint

2. **Live Training Metrics**:
   - Real-time loss curves (train & validation)
   - Real-time accuracy curves
   - Current epoch progress bar
   - Time remaining estimate
   - Learning rate display
   - GPU/CPU utilization (if available)
   - Memory usage

3. **Live Plots** (updating every N batches):
   - Training loss vs validation loss
   - Training accuracy vs validation accuracy
   - Learning rate schedule
   - Gradient norms (optional)
   - Batch processing time

4. **Training Status Panel**:
   - Current epoch / total epochs
   - Best validation accuracy achieved
   - Best validation loss
   - Time elapsed / estimated remaining
   - Samples processed / total
   - Training state (idle/running/paused/completed/error)

5. **Model Comparison**:
   - Compare multiple training runs
   - Side-by-side metric comparison
   - Best model highlighting
   - Export comparison report

6. **Alert System**:
   - Training completion notification
   - Error alerts
   - Early stopping trigger notification
   - New best model alert

Technical Requirements:
- Asynchronous training execution (don't block UI)
- WebSocket or polling for real-time updates
- Efficient data streaming (don't send all data points)
- Responsive design (works on different screen sizes)
- Export training curves as images

Please provide:
- Frontend component structure
- Backend API endpoints for training control
- Real-time update mechanism
- State management strategy
- Complete implementation

Make it visually appealing with modern UI components and smooth animations.
```

**Deliverables**:
- ✅ Real-time training dashboard
- ✅ Training control interface
- ✅ Live metric visualization

---

### 4.2 Model Evaluation Interface (Day 4-5)

**Task**: Build comprehensive evaluation dashboard

**Claude Prompt**:
```
Develop a complete model evaluation and results visualization interface:

Evaluation Dashboard Features:

1. **Model Selection & Loading**:
   - Dropdown to select trained models
   - Display model metadata (architecture, training date, hyperparameters)
   - Load checkpoint button
   - Model comparison selector (select multiple models)

2. **Test Set Evaluation**:
   - "Run Evaluation" button
   - Progress indicator
   - Display metrics:
     * Accuracy
     * Precision (per class & weighted)
     * Recall (per class & weighted)
     * F1 Score (per class & weighted)
     * Confusion Matrix (interactive heatmap)

3. **Visualizations**:
   - Confusion matrix with hover details
   - ROC curves (for binary/multi-class)
   - Precision-Recall curves
   - Class-wise performance bar charts
   - Prediction confidence distribution

4. **Error Analysis**:
   - Misclassification viewer
   - Show samples where model failed
   - Display: true label, predicted label, confidence
   - Visualize error cases (plot signals)
   - Error patterns analysis
   - Confidence threshold slider

5. **Prediction Interface**:
   - Upload new data for prediction
   - Real-time prediction results
   - Confidence scores
   - Class probabilities visualization
   - Batch prediction mode

6. **Model Comparison**:
   - Side-by-side metrics comparison table
   - Comparative plots
   - Performance difference highlighting
   - Best model recommendation

7. **Export Results**:
   - Export predictions (CSV/JSON)
   - Download confusion matrix
   - Export all plots (PNG/PDF)
   - Generate evaluation report (PDF)

Technical Requirements:
- Efficient loading of evaluation results
- Interactive plots (zoom, hover, click)
- Batch processing for large test sets
- Progress tracking for long evaluations
- Cache evaluation results

Implementation Details:
- Component architecture
- API endpoints for evaluation
- Data flow design
- Performance optimization
- Complete code

Design with professional data science tool aesthetics.
```

**Deliverables**:
- ✅ Evaluation dashboard
- ✅ Error analysis tools
- ✅ Prediction interface

---

## Phase 5: Backend API Development (Week 5)

### 5.1 REST API Implementation (Day 1-3)

**Task**: Build complete backend API

**Claude Prompt**:
```
Create a comprehensive REST API backend for the CNN sensor analysis system:

API Requirements:

**Endpoints Structure**:

1. **Data Management**:
   - POST /api/data/upload - Upload raw data
   - GET /api/data/list - List available datasets
   - GET /api/data/{id}/info - Get dataset metadata
   - POST /api/data/{id}/preprocess - Trigger preprocessing
   - GET /api/data/{id}/statistics - Get data statistics
   - DELETE /api/data/{id} - Delete dataset
   - GET /api/data/{id}/download - Download processed data

2. **Model Management**:
   - GET /api/models/list - List available models
   - GET /api/models/{id}/info - Get model details
   - POST /api/models/train - Start training
   - GET /api/models/train/{job_id}/status - Check training status
   - POST /api/models/train/{job_id}/stop - Stop training
   - DELETE /api/models/{id} - Delete model

3. **Evaluation**:
   - POST /api/evaluate - Run evaluation
   - GET /api/evaluate/{job_id}/status - Check evaluation status
   - GET /api/evaluate/{job_id}/results - Get evaluation results
   - POST /api/predict - Make predictions
   - POST /api/predict/batch - Batch predictions

4. **Monitoring**:
   - GET /api/training/{job_id}/metrics - Get training metrics stream
   - GET /api/system/status - System health check
   - GET /api/system/resources - GPU/CPU/Memory usage

5. **Utilities**:
   - GET /api/config - Get configuration
   - POST /api/config - Update configuration
   - GET /api/export/{type}/{id} - Export results/reports

**Technical Requirements**:

1. Framework: FastAPI (recommended) or Flask
2. Async support for long-running tasks
3. Background task queue (Celery or FastAPI BackgroundTasks)
4. WebSocket support for real-time updates
5. Authentication (JWT tokens) - basic implementation
6. Request validation (Pydantic models)
7. Error handling with proper HTTP status codes
8. API documentation (auto-generated with FastAPI)
9. CORS configuration
10. File upload handling (multipart/form-data)
11. Progress tracking for long operations
12. Rate limiting (optional)

**Additional Components**:

1. Database models (SQLite for simplicity):
   - Datasets table
   - Models table
   - Training jobs table
   - Evaluation results table

2. Task queue manager:
   - Training task handler
   - Preprocessing task handler
   - Evaluation task handler
   - Status tracking

3. File storage manager:
   - Organize uploaded files
   - Manage model checkpoints
   - Store results

Please provide:
- Complete API implementation
- Request/response schemas (Pydantic)
- Database models
- Task queue setup
- Error handling middleware
- API documentation
- Testing examples (curl or Python requests)

Structure code professionally with proper separation of concerns.
```

**Deliverables**:
- ✅ REST API implemented
- ✅ Database schema created
- ✅ Task queue configured
- ✅ API documentation

---

### 5.2 Real-time Communication (Day 4-5)

**Task**: Implement WebSocket for live updates

**Claude Prompt**:
```
Implement WebSocket communication for real-time updates in the sensor analysis system:

Requirements:

1. **WebSocket Server Setup**:
   - WebSocket endpoint configuration
   - Connection management (connect/disconnect)
   - Client identification
   - Connection pool management

2. **Real-time Events**:
   Training Events:
   - training_started
   - epoch_complete (with metrics)
   - batch_complete (periodic updates)
   - training_paused
   - training_resumed
   - training_completed
   - training_failed
   - best_model_updated
   
   Data Processing Events:
   - preprocessing_started
   - preprocessing_progress (percentage)
   - preprocessing_completed
   
   Evaluation Events:
   - evaluation_started
   - evaluation_progress
   - evaluation_completed
   
   System Events:
   - resource_usage_update (GPU/CPU/Memory)

3. **Message Format** (JSON):
   ```
   {
     "event_type": "epoch_complete",
     "job_id": "training_123",
     "timestamp": "2025-10-13T10:30:00Z",
     "data": {
       "epoch": 10,
       "train_loss": 0.234,
       "val_loss": 0.256,
       "train_acc": 92.3,
       "val_acc": 90.1
     }
   }
   ```

4. **Client-side Integration**:
   - WebSocket connection handler
   - Event listeners
   - Automatic reconnection
   - Error handling
   - Message queueing (if connection lost)

5. **Broadcasting Strategy**:
   - Send updates to specific clients (by job_id)
   - Broadcast system-wide updates
   - Throttling (don't send too frequently)
   - Batch updates when appropriate

6. **Features**:
   - Heartbeat/ping-pong for connection health
   - Authentication token validation
   - Subscribe/unsubscribe to specific events
   - Historical data catch-up (if client reconnects)

Please provide:
- WebSocket server implementation
- Event emitter system
- Client connection manager
- Frontend WebSocket client code
- Message serialization/deserialization
- Error recovery mechanisms
- Testing utilities

Ensure low latency and efficient message handling.
```

**Deliverables**:
- ✅ WebSocket server implemented
- ✅ Real-time event system
- ✅ Client integration code

---

## Phase 6: Integration & Optimization (Week 6)

### 6.1 System Integration (Day 1-3)

**Task**: Connect all components

**Claude Prompt**:
```
Create a complete system integration guide and final assembly:

Integration Tasks:

1. **Frontend-Backend Connection**:
   - API client service (axios/fetch wrapper)
   - Error handling and retry logic
   - Loading states management
   - Authentication flow
   - File upload with progress

2. **State Management**:
   - Global state structure (Redux/Zustand/Context)
   - Data flow between components
   - Cache management
   - Optimistic updates

3. **Configuration Management**:
   - Environment variables (.env)
   - Development vs Production configs
   - Feature flags
   - API endpoints configuration

4. **Deployment Setup**:
   - Docker containerization:
     * Backend container (FastAPI + Python environment)
     * Frontend container (if using React)
     * Database container
   - Docker Compose orchestration
   - Volume management (data persistence)
   - Network configuration

5. **Startup Scripts**:
   - Backend startup script
   - Frontend startup script
   - Database initialization
   - One-command startup (docker-compose up)

6. **Health Checks**:
   - API health endpoint
   - Database connection check
   - Model availability check
   - Disk space check
   - GPU availability check

Please provide:
- Complete integration code
- Dockerfile for each service
- docker-compose.yml
- Environment configuration examples
- Startup/shutdown scripts
- Integration testing approach
- Troubleshooting guide

Document the complete system architecture and data flow.
```

**Deliverables**:
- ✅ System fully integrated
- ✅ Docker containers configured
- ✅ Deployment scripts ready

---

### 6.2 Performance Optimization (Day 4-5)

**Task**: Optimize system performance

**Claude Prompt**:
```
Optimize the sensor analysis system for performance and user experience:

Optimization Areas:

1. **Data Loading & Processing**:
   - Implement lazy loading for large datasets
   - Use memory-mapped files for huge data
   - Batch processing strategies
   - Parallel preprocessing (multiprocessing)
   - Caching preprocessed data

2. **Visualization Performance**:
   - Data decimation for plots (show 10k points instead of millions)
   - Progressive rendering
   - Virtual scrolling for large lists
   - Debouncing user inputs
   - Canvas-based rendering for complex plots

3. **Model Training**:
   - DataLoader optimization (num_workers, pin_memory)
   - Mixed precision training
   - Gradient accumulation for large batches
   - Efficient data augmentation
   - Model compilation (torch.compile if using PyTorch 2.0+)

4. **API Performance**:
   - Response caching
   - Database query optimization
   - Connection pooling
   - Async request handling
   - Rate limiting
   - Pagination for large results

5. **Frontend Optimization**:
   - Code splitting
   - Lazy component loading
   - Memoization (React.memo, useMemo)
   - Virtualized lists
   - Image/asset optimization
   - Service worker for offline capability

6. **Memory Management**:
   - Clear unused data from memory
   - Garbage collection triggers
   - Memory profiling tools
   - Memory leak detection

7. **Monitoring & Profiling**:
   - Performance metrics collection
   - Slow query logging
   - Frontend performance monitoring
   - Backend profiling

Please provide:
- Optimization implementations for each area
- Benchmarking scripts
- Performance monitoring setup
- Memory profiling tools
- Optimization checklist
- Before/after performance comparison guide

Include specific PyTorch, NumPy, and web optimization techniques.
```

**Deliverables**:
- ✅ System optimized
- ✅ Performance benchmarks
- ✅ Monitoring tools configured

---

## Phase 7: Testing & Documentation (Week 7)

### 7.1 Comprehensive Testing (Day 1-3)

**Task**: Implement complete test suite

**Claude Prompt**:
```
Create a comprehensive testing suite for the sensor analysis system:

Testing Requirements:

1. **Unit Tests**:
   Backend:
   - Data preprocessing functions
   - Model architectures (forward pass)
   - Training utilities
   - API endpoints
   - Database operations
   
   Frontend:
   - Component rendering
   - User interactions
   - State management
   - API client functions

2. **Integration Tests**:
   - Data pipeline (upload → preprocess → load)
   - Training workflow (setup → train → evaluate)
   - API endpoint chains
   - WebSocket communication
   - File operations

3. **End-to-End Tests**:
   - Complete user workflows:
     * Upload data → Train model → Evaluate → Export results
     * Load existing model → Make predictions
     * Compare multiple models
   - GUI interactions
   - Error scenarios

4. **Performance Tests**:
   - Large dataset handling
   - Concurrent user simulation
   - Memory leak detection
   - Training speed benchmarks

5. **Test Fixtures & Mocks**:
   - Sample datasets (small, medium, large)
   - Mock models
   - Dummy API responses
   - Test configurations

Please provide:
- pytest test suite structure
- Test fixtures and factories
- Mock implementations
- Testing utilities
- Coverage configuration
- CI/CD integration examples (GitHub Actions)
- Test documentation

Aim for >80% code coverage.
Include testing best practices and patterns.
```

**Deliverables**:
- ✅ Test suite implemented
- ✅ Code coverage >80%
- ✅ CI/CD configured

---

### 7.2 Documentation (Day 4-5)

**Task**: Create complete documentation

**Claude Prompt**:
```
Create comprehensive documentation for the sensor analysis system:

Documentation Structure:

1. **User Guide** (for end users):
   - Getting Started
     * System requirements
     * Installation instructions
     * First-time setup
   
   - User Interface Guide
     * Dashboard overview
     * Data import/export
     * Dataset exploration
     * Training models
     * Model evaluation
     * Making predictions
   
   - Tutorials
     * Quick start (5-minute example)
     * Complete workflow walkthrough
     * Advanced features
     * Troubleshooting common issues
   
   - FAQ

2. **Developer Guide** (for developers):
   - Architecture Overview
     * System components
     * Data flow diagrams
     * Technology stack
   
   - Setup & Development
     * Development environment setup
     * Running locally
     * Code organization
     * Coding standards
   
   - API Documentation
     * Endpoint reference
     * Request/response examples
     * Authentication
     * WebSocket events
   
   - Model Development
     * Adding new architectures
     * Custom preprocessing
     * Extending training pipeline
   
   - Testing Guide
     * Running tests
     * Writing new tests
     * CI/CD pipeline

3. **Deployment Guide**:
   - Docker deployment
   - Cloud deployment (AWS/GCP/Azure)
   - Configuration management
   - Scaling strategies
   - Monitoring and logging
   - Backup and recovery

4. **API Reference**:
   - Auto-generated API docs (Swagger/OpenAPI)
   - Code API documentation (docstrings)
   - Configuration reference

5. **Research Documentation**:
   - Model architectures explained
   - Hyperparameter tuning guide
   - Performance benchmarks
   - Comparison with baseline methods
   - Future improvements

Please provide:
- Documentation templates for each section
- Markdown file structure
- Diagram examples (use mermaid or ascii)
- Code snippet formats
- Screenshot guidelines
- Documentation generation setup (Sphinx/MkDocs)

Make it clear, professional, and easy to navigate.
Include visual aids and examples throughout.
```

**Deliverables**:
- ✅ Complete documentation
- ✅ User guide
- ✅ Developer guide
- ✅ API documentation

---

## Phase 8: Final Polish & Deployment (Week 8)

### 8.1 UI/UX Polish (Day 1-3)

**Task**: Final UI/UX improvements

**Claude Prompt**:
```
Finalize UI/UX for production-ready application:

Polish Tasks:

1. **Visual Design**:
   - Consistent color scheme (suggest modern professional palette)
   - Typography standards
   - Icon set (recommend icon library)
   - Logo and branding
   - Loading states and skeletons
   - Empty states with helpful messages
   - Error state designs

2. **User Experience**:
   - Smooth transitions and animations
   - Keyboard shortcuts
   - Tooltips and help text
   - Confirmation dialogs for destructive actions
   - Success/error notifications (toast messages)
   - Undo functionality where appropriate
   - Autosave drafts

3. **Accessibility**:
   - ARIA labels
   - Keyboard navigation
   - Screen reader support
   - Color contrast compliance (WCAG)
   - Focus indicators
   - Alt text for images

4. **Responsive Design**:
   - Mobile view considerations
   - Tablet view
   - Different screen sizes
   - Touch-friendly controls

5. **Performance UX**:
   - Optimistic UI updates
   - Progressive loading
   - Pagination for large datasets
   - Infinite scroll where appropriate
   - Debounced search

6. **Onboarding**:
   - Welcome screen
   - Feature tour
   - Contextual help
   - Sample dataset for demo

7. **Settings & Preferences**:
   - Theme selection (light/dark mode)
   - Notification preferences
   - Display preferences
   - Export/import settings

Please provide:
- UI component library recommendations
- Design system guidelines
- Accessibility checklist
- Responsive breakpoints
- Animation guidelines
- Implementation code for key UX features

Focus on professional, modern, and user-friendly design.
```

**Deliverables**:
- ✅ UI polished and professional
- ✅ UX optimized
- ✅ Accessibility implemented

---

### 8.2 Production Deployment (Day 4-5)

**Task**: Deploy to production

**Claude Prompt**:
```
Create production deployment strategy and implementation:

Deployment Requirements:

1. **Production Configuration**:
   - Environment variables for production
   - Security settings
   - HTTPS/SSL configuration
   - CORS policies
   - Rate limiting
   - Logging configuration

2. **Cloud Deployment Options**:
   
   Option A: AWS
   - EC2 instance setup
   - Load balancer configuration
   - S3 for file storage
   - RDS for database (if needed)
   - CloudWatch monitoring
   
   Option B: Google Cloud
   - Compute Engine / Cloud Run
   - Cloud Storage
   - Cloud SQL
   - Cloud Monitoring
   
   Option C: Azure
   - Virtual Machines / Container Instances
   - Blob Storage
   - Azure Database
   - Azure Monitor
   
   Option D: DigitalOcean (simpler)
   - Droplet setup
   - Spaces for storage
   - Managed database
   
   Option E: Local Server
   - Server requirements
   - Network configuration
   - Firewall rules

3. **Deployment Scripts**:
   - Automated deployment script
   - Database migration script
   - Health check verification
   - Rollback procedure

4. **Monitoring & Logging**:
   - Application logging (structured logs)
   - Error tracking (Sentry or similar)
   - Performance monitoring (APM)
   - Uptime monitoring
   - Alert configuration

5. **Backup & Recovery**:
   - Database backup strategy
   - Model checkpoint backup
   - Configuration backup
   - Disaster recovery plan

6. **Security**:
   - Authentication implementation
   - API key management
   - Secrets management
   - Input validation
   - SQL injection prevention
   - XSS protection
   - CSRF protection

7. **Scaling Strategy**:
   - Horizontal scaling approach
   - Load balancing
   - Caching strategy (Redis)
   - CDN for static assets

Please provide:
- Step-by-step deployment guide for chosen platform
- Production Docker setup
- Kubernetes manifests (if applicable)
- Nginx/Apache configuration
- SSL certificate setup
- Monitoring dashboard setup
- Backup scripts
- Security hardening checklist

Make it production-grade with proper security and reliability.
```

**Deliverables**:
- ✅ Production deployment completed
- ✅ Monitoring configured
- ✅ Security hardened
- ✅ Backup system active

---

## GUI Technology Recommendations

### Option 1: Streamlit (Fastest Development - Recommended for MVP)

**Pros**:
- Pure Python (no frontend coding)
- Rapid development (build in days)
- Built-in components for ML applications
- Easy deployment
- Real-time updates support

**Cons**:
- Less customization flexibility
- Single-page limitations
- Custom components require JavaScript

**Best for**: Quick MVP, Python developers, research/demo applications

---

### Option 2: Gradio (ML-Focused)

**Pros**:
- Designed specifically for ML models
- Extremely fast prototyping
- Auto-generates API
- Easy sharing

**Cons**:
- Limited layout customization
- Best for simple interfaces
- Not suitable for complex dashboards

**Best for**: Simple model demos, quick experiments

---

### Option 3: React + Plotly/D3.js (Most Professional)

**Pros**:
- Maximum customization
- Best performance
- Professional appearance
- Component reusability
- Large ecosystem

**Cons**:
- Requires frontend expertise
- Longer development time
- More complex setup

**Best for**: Production applications, complex dashboards, commercial products

---

### Option 4: Dash (Python + Plotly)

**Pros**:
- Pure Python with reactive components
- Excellent for data visualization
- Good balance of customization and ease
- Built on Plotly

**Cons**:
- Learning curve for callbacks
- Can become complex for large apps

**Best for**: Data-heavy dashboards, analytical tools

---

## Recommended Tech Stack

### Backend:
- **Framework**: FastAPI (modern, fast, auto-docs)
- **Deep Learning**: PyTorch
- **Data Processing**: NumPy, Pandas, SciPy
- **Database**: PostgreSQL (production) / SQLite (development)
- **Task Queue**: Celery + Redis
- **WebSocket**: FastAPI WebSocket support

### Frontend (Choose One):
- **MVP**: Streamlit
- **Production**: React + Plotly + TailwindCSS
- **Alternative**: Dash

### Deployment:
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (for scale) or Docker Swarm
- **Cloud**: AWS / GCP / Azure / DigitalOcean

---

## Hardware Requirements (MVP)

### Development:
- **CPU**: 6+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3060 (12GB) or better
- **Storage**: 100 GB SSD

### Production Server:
- **CPU**: 8+ cores
- **RAM**: 64 GB
- **GPU**: RTX 4070 or better (for training), or cloud GPU
- **Storage**: 500 GB SSD

---

## Project Milestones

### Week 1: Foundation ✅
- [ ] Project structure created
- [ ] Environment configured
- [ ] GUI framework selected
- [ ] Data format standardized

### Week 2: Data Pipeline ✅
- [ ] Preprocessing pipeline working
- [ ] Data import/export GUI functional
- [ ] Dataset insights dashboard complete

### Week 3: Models ✅
- [ ] Three CNN models implemented
- [ ] Training pipeline working
- [ ] Models trainable via CLI

### Week 4: Training GUI ✅
- [ ] Real-time training monitor working
- [ ] Evaluation dashboard complete
- [ ] Prediction interface functional

### Week 5: Backend ✅
- [ ] REST API complete
- [ ] WebSocket working
- [ ] Database configured

### Week 6: Integration ✅
- [ ] All components connected
- [ ] System optimized
- [ ] Docker deployment working

### Week 7: Quality ✅
- [ ] Tests passing (>80% coverage)
- [ ] Documentation complete
- [ ] User guide finished

### Week 8: Launch ✅
- [ ] UI polished
- [ ] Production deployed
- [ ] Monitoring active
- [ ] System validated

---

## Success Criteria

### Technical:
- ✅ Handles 6000+ signals efficiently
- ✅ Training completes successfully
- ✅ Real-time updates work smoothly
- ✅ API response time <500ms (excluding long operations)
- ✅ GUI responsive and intuitive

### Performance:
- ✅ Model accuracy >80% (adjust based on problem)
- ✅ Training time reasonable for dataset size
- ✅ Prediction inference <1 second

### User Experience:
- ✅ Users can upload data without instructions
- ✅ Training status clear and informative
- ✅ Results easy to interpret
- ✅ No crashes or data loss

---

## Risk Mitigation

### Risk 1: Large Data Handling
**Mitigation**: Implement progressive loading, downsampling, memory-mapped files

### Risk 2: Long Training Times
**Mitigation**: Start with small subset, use pretrained models, implement pause/resume

### Risk 3: GPU Availability
**Mitigation**: Support CPU training (slower), cloud GPU options, model optimization

### Risk 4: UI Complexity
**Mitigation**: Start with Streamlit for rapid development, iterate based on feedback

### Risk 5: Performance Issues
**Mitigation**: Profiling from day one, caching strategies, background processing

---

## Next Steps After MVP

1. **Advanced Features**:
   - Attention mechanisms in models
   - AutoML for hyperparameter tuning
   - Model ensemble
   - Feature importance visualization

2. **Production Enhancements**:
   - Multi-user support with authentication
   - User project management
   - Collaborative features
   - Model versioning system

3. **Scaling**:
   - Distributed training
   - Model serving optimization
   - CDN for static assets
   - Database sharding

4. **Advanced Analytics**:
   - Explainability (SHAP, LIME)
   - Uncertainty quantification
   - Anomaly detection
   - Transfer learning from pretrained models

---

## Resources for Claude Interactions

### For Each Development Phase:
1. Start with the provided prompt
2. Ask Claude to implement the component
3. Review the code
4. Ask for modifications if needed
5. Request tests for the component
6. Ask for documentation

### Example Interaction Flow:
```
You: [Paste prompt from Phase 2.1]
Claude: [Provides preprocessing module code]
You: Can you add support for HDF5 files?
Claude: [Adds HDF5 support]
You: Now create unit tests for this module
Claude: [Provides pytest tests]
You: Great! Document the API
Claude: [Provides docstrings and usage examples]
```

---

## Support & Maintenance

### After Deployment:
- Monitor error logs daily
- Track performance metrics weekly
- Update dependencies monthly
- Backup data weekly
- Security patches immediately
- User feedback review bi-weekly

### Version Control:
- Use Git for code
- Tag releases (v1.0.0, v1.1.0, etc.)
- Maintain changelog
- Branch strategy (main, develop, feature/*)

---

**Document Version**: 2.0  
**Last Updated**: October 2025  
**Status**: Ready for Implementation

---

## Quick Start Command Reference

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Development
python -m src.data.preprocessing  # Preprocess data
python train.py --model resnet    # Train model
python evaluate.py --model resnet # Evaluate model
streamlit run app.py              # Run GUI

# Production
docker-compose up -d              # Start all services
docker-compose logs -f            # View logs
docker-compose down               # Stop services
```

---

**Ready to start building! Use each prompt with Claude to implement the corresponding component.**
