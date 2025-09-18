# Bruise Fusion System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.1-green.svg)](https://github.com/artinmajdi/RAP)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red.svg)](https://imagefuse.streamlit.app/)

An advanced image processing application that combines white-light and ALS (Alternate Light Source) images using spatial-frequency fusion techniques to enhance bruise visibility for forensic and medical analysis.

## 🔬 Overview

The Bruise Fusion System provides multiple state-of-the-art fusion algorithms with intelligent image alignment, quality assessment, and an interactive web-based dashboard for real-time processing and visualization. This tool is designed for forensic investigators, medical professionals, and researchers who need to enhance bruise visibility by fusing complementary imaging modalities.

## ✨ Key Features

### 🧠 Advanced Fusion Algorithms
- **Frequency Domain Fusion**: Combines low-frequency components from white-light images with high-frequency details from ALS images
- **Laplacian Pyramid Fusion**: Multi-scale decomposition for optimal detail preservation
- **Wavelet DWT Fusion**: Discrete Wavelet Transform-based fusion with multiple wavelet types
- **Gradient-Based Fusion**: Edge-preserving fusion using gradient information
- **Hybrid Adaptive Fusion**: Intelligent fusion that adapts to local image characteristics

### 🎯 Intelligent Image Alignment
- **Auto-Selection**: Automatically chooses the optimal alignment method based on image characteristics
- **ORB + RANSAC**: Fast feature-based alignment for real-time processing
- **SIFT + RANSAC**: High-quality feature detection for challenging image pairs
- **Multi-Scale Alignment**: Robust alignment across different scales and conditions
- **Hybrid SIFT+ECC**: Premium alignment with Enhanced Correlation Coefficient refinement

### 📊 Quality Assessment
- **Comprehensive Metrics**: SSIM, PSNR, MSE, RMSE, NCC calculations
- **Alignment Quality**: Sub-pixel precision validation
- **Fusion Quality**: Edge preservation and contrast enhancement metrics
- **Real-time Feedback**: Instant quality assessment during processing

### 🖥️ Interactive Dashboard
- **Streamlit-based Interface**: Modern, responsive web interface
- **Drag-and-Drop Upload**: Easy image loading with format validation
- **Real-time Parameter Adjustment**: Interactive controls for all fusion parameters
- **Side-by-side Comparison**: Visual comparison of original and fused images
- **Export Capabilities**: Download results and generate quality reports

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Install from PyPI
```bash
pip install bruise-fusion
```

### Install from Source
```bash
git clone https://github.com/artinmajdi/RAP.git
cd bruise_fusion
pip install -e .
```

### Dependencies
The package automatically installs the following dependencies:
- `opencv-python>=4.12.0` - Computer vision and image processing
- `numpy>=2.2.0` - Numerical computing
- `matplotlib>=3.10.0` - Plotting and visualization
- `streamlit>=1.28.0` - Web dashboard framework
- `Pillow>=9.0.0` - Image processing library
- `pandas>=2.3.0` - Data manipulation
- `seaborn>=0.13.0` - Statistical visualization
- `pywavelets>=1.8.0` - Wavelet transforms
- `scikit-image>=0.25.2` - Image processing algorithms
- `rich>=14.1.0` - Rich text and beautiful formatting
- `rawpy>=0.21.0` - RAW image processing
- `imageio>=2.37.0` - Image I/O operations

## 📖 Usage

### Quick Start with Dashboard

Launch the interactive Streamlit dashboard:

```bash
bfuse
```

This will start the web interface at `http://localhost:8501` where you can:

1. **Upload Images**: Drag and drop your white-light and ALS images
2. **Select Method**: Choose from 5 different fusion algorithms
3. **Configure Parameters**: Adjust method-specific settings
4. **Process Images**: Click "Process Images" to start fusion
5. **Review Results**: Examine fused images and quality metrics
6. **Export Results**: Download processed images and reports

### Programmatic Usage

```python
from src.utils import AdvancedBruiseFusion, FusionConfig, FusionMethod
import cv2

# Load images
white_image = cv2.imread('white_light.jpg')
als_image = cv2.imread('als_image.jpg')

# Create fusion configuration
config = FusionConfig(
    method=FusionMethod.FREQUENCY_DOMAIN,
    alignment_method='auto',
    max_size=2200,
    sigma_low=8.0,
    sigma_high=2.0,
    preserve_color='lab'
)

# Initialize fusion engine
fusion_engine = AdvancedBruiseFusion(config)

# Process images
results = fusion_engine.fuse_images(white_image, als_image)

# Access results
fused_image = results['fused_image']
quality_metrics = results['metrics']
alignment_info = results['alignment']
```

## 🏗️ Project Structure

```
bruise_fusion/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── dashboard.py         # Streamlit web interface
│   ├── utils.py            # Core fusion algorithms
│   └── settings.py         # Configuration and logging
├── tests/
│   └── test_fusion.py      # Unit tests
├── docs/
│   ├── bruise_fusion_prd.md              # Product requirements
│   └── ALIGNMENT_METHODS_DOCUMENTATION.md # Technical documentation
├── pyproject.toml          # Project configuration
├── setup.py               # Package setup
└── README.md              # This file
```

## 🔧 Configuration Options

### Fusion Methods

| Method | Best For | Key Parameters |
|--------|----------|----------------|
| Frequency Domain | General purpose, fast processing | `sigma_low`, `sigma_high`, `w_low`, `w_high` |
| Laplacian Pyramid | Multi-scale detail preservation | `pyramid_levels`, `pyramid_sigma` |
| Wavelet DWT | Texture and edge preservation | `wavelet_type`, `wavelet_levels` |
| Gradient-Based | Edge-rich images | `gradient_sigma`, `edge_threshold` |
| Hybrid Adaptive | Complex scenes, automatic adaptation | `local_window_size`, `contrast_threshold` |

### Alignment Methods

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Auto-Select | Variable | High | Unknown image characteristics |
| ORB + RANSAC | Fast | Good | Real-time processing |
| SIFT + RANSAC | Medium | High | High-quality results |
| Multi-Scale | Slow | Very High | Challenging alignments |
| Hybrid SIFT+ECC | Slowest | Highest | Critical applications |

## 📊 Quality Metrics

The system provides comprehensive quality assessment:

- **SSIM (Structural Similarity Index)**: Measures structural similarity (higher is better)
- **PSNR (Peak Signal-to-Noise Ratio)**: Signal quality measurement (higher is better)
- **MSE (Mean Squared Error)**: Pixel-level differences (lower is better)
- **NCC (Normalized Cross-Correlation)**: Template matching quality (higher is better)
- **Edge Preservation**: Measures how well edges are maintained during fusion

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/artinmajdi/RAP.git
cd bruise_fusion
pip install -e ".[dev]"
pre-commit install
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Product Requirements Document](docs/bruise_fusion_prd.md)
- [Alignment Methods Documentation](docs/ALIGNMENT_METHODS_DOCUMENTATION.md)

## 🐛 Issues and Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/artinmajdi/RAP/issues) page
2. Create a new issue with detailed information
3. Include sample images and error messages when possible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Artin Majdi**
- Email: mmajdi@gmu.edu
- GitHub: [@artinmajdi](https://github.com/artinmajdi)
- Institution: George Mason University

## 🙏 Acknowledgments

- Forensic science community for requirements and feedback
- Medical imaging professionals for validation and testing
- Open source computer vision community for foundational algorithms

## 📈 Roadmap

- [ ] Machine learning-based parameter optimization
- [ ] 3D-aware alignment methods
- [ ] Real-time processing optimization
- [ ] Cloud deployment capabilities
- [ ] API development for system integration
- [ ] Advanced analytics dashboard

---

*For more information about forensic image analysis and bruise detection techniques, please refer to the scientific literature and forensic imaging standards.*