# StepReaderCNN Projects

This repository contains CNN-based projects for electrochemical sensor signal processing.

## Project Structure

```
25_StepReaderCNN/
├── MNP_sizer_CNN/          # Magnetic Nanoparticle (MNP) sizing project (published)
│   ├── src/                # Source code
│   ├── models/             # Trained models
│   ├── TestData/           # Test datasets
│   ├── app.py              # Original Streamlit app
│   └── requirements.txt    # Python dependencies
├── app.py                  # Root-level Streamlit entry point (for deployment)
├── requirements.txt        # Root-level dependencies (for deployment)
└── .streamlit/             # Streamlit configuration
```

## MNP Sizer CNN

The MNP_sizer_CNN project is a CNN-based system for analyzing electrochemical sensor signals to determine magnetic nanoparticle sizes. This project has been published and is deployed on Streamlit Cloud.

### Features
- Data exploration and visualization
- CNN model training (SimpleCNN1D, ResNet1D, MultiScaleCNN1D)
- Model evaluation and testing
- Synthetic data generation
- Interactive Streamlit GUI

## Deployment

The repository is configured for Streamlit Cloud deployment:
- The root `app.py` redirects to the MNP_sizer_CNN project
- All dependencies are specified in the root `requirements.txt`
- Configuration is in `.streamlit/config.toml`

## Future Projects

Additional biomarker concentration recognition projects will be added as separate folders alongside the MNP_sizer_CNN project.

## License

See LICENSE file for details.
