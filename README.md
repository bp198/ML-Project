# 🦴 Trilobite Age Prediction using Machine Learning

> **Predicting fossil age from biological and environmental characteristics using Random Forest regression**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Scientific Insights](#scientific-insights)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🔬 Overview

This project develops a machine learning pipeline to predict trilobite fossil ages using taxonomic, ecological, and geographic characteristics. By analyzing 29,039 trilobite specimens, we achieved **98.1% accuracy (R² = 0.9813)** in age prediction while maintaining scientific rigor and avoiding data leakage.

### 🎯 Objectives
- **Scientific**: Understand which biological/environmental factors best predict trilobite age
- **Technical**: Develop a robust ML pipeline for paleontological age prediction
- **Practical**: Create a deployable model for fossil age estimation

## 📊 Dataset

**Source**: Comprehensive trilobite fossil database  
**Size**: 29,039 specimens  
**Features**: 30 original variables including:
- **Taxonomic**: Order, family, genus, species
- **Temporal**: Early/late intervals, age ranges
- **Geographic**: Longitude, latitude, country, state
- **Geological**: Formation, lithology, environment
- **Biological**: Life habit, vision, diet, preservation mode

**Time Range**: Primarily Paleozoic Era (252-541 Million Years Ago)

## ✨ Key Features

### 🔧 Advanced Feature Engineering
- **Manual Feature Creation**: Geological duration, paleocoordinates, taxonomic diversity
- **Geographic Features**: Hemisphere classification, equatorial distance
- **Categorical Encoding**: Label encoding, one-hot encoding, frequency encoding
- **Data Leakage Prevention**: Careful removal of age-derived features

### 🤖 Machine Learning Pipeline
- **Algorithm**: Random Forest Regressor (100 trees)
- **Validation**: 80/20 train-test split with proper scaling
- **Feature Selection**: 10 scientifically valid predictors
- **Performance**: R² = 0.9813, MAE = 2.35 Mya

### 📈 Comprehensive Analysis
- **Feature Importance**: Gini vs Permutation importance comparison
- **Model Validation**: Residual analysis, cross-validation
- **Manual vs ML Comparison**: Baseline validation against rule-based methods
- **Export System**: Complete model artifacts and reproducible results

## 🏆 Results

### Model Performance
```
📊 Final Model Metrics:
• R² Score: 0.9813 (98.1% variance explained)
• Mean Absolute Error: 2.35 Million Years
• Root Mean Square Error: 6.56 Million Years
• Prediction Range: 254.40 - 538.80 Mya
```

### Top Predictive Features
| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | Family (Taxonomic) | 52.3% | Categorical |
| 2 | Order (Taxonomic) | 20.9% | Categorical |
| 3 | Diet | 10.4% | Categorical |
| 4 | Longitude | 5.7% | Geographic |
| 5 | Latitude | 3.7% | Geographic |

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
Git
```

### Clone Repository
```bash
git clone https://github.com/bp198/trilobite-age-prediction.git
cd trilobite-age-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## 💻 Usage

### Quick Start
```python
import joblib
import pandas as pd

# Load the trained model
model_artifacts = joblib.load('results/model_artifacts.pkl')
model = model_artifacts['model']
scaler = model_artifacts['scaler']
feature_names = model_artifacts['feature_names']

# Prepare your data (ensure features match model expectations)
# your_data should have columns: longitude, latitude, equatorial_distance_manual,
# order_encoded, family_encoded, genus_encoded, environment_encoded, 
# life_habit_encoded, diet_encoded, hemisphere_manual_encoded

# Make predictions
predictions = model.predict(your_data[feature_names])
print(f"Predicted ages: {predictions} Million Years Ago")
```

### Full Pipeline
```bash
# Run complete analysis
python src/main_analysis.py

# Generate results
python src/export_results.py

# Verify exports
python src/verify_exports.py
```

## 📁 Project Structure

```
trilobite-age-prediction/
├── data/
│   └── trilobite.csv                 # Raw dataset
├── src/
│   ├── data_preprocessing.py         # Data cleaning and preparation
│   ├── feature_engineering.py       # Manual feature creation
│   ├── model_training.py            # ML pipeline
│   ├── analysis_visualization.py    # Comprehensive analysis
│   ├── export_results.py            # Results export system
│   └── verify_exports.py            # Verification system
├── results/
│   ├── model_artifacts.pkl          # Trained model + preprocessing
│   ├── predictions_comparison.csv   # All predictions and metrics
│   ├── feature_importance.csv       # Feature rankings
│   ├── comprehensive_results.png    # Key visualizations
│   ├── performance_metrics.json     # Detailed metrics
│   └── report.html                  # Complete analysis report
├── notebooks/
│   └── trilobite_analysis.ipynb     # Jupyter notebook version
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Quality Control**: Missing value analysis and imputation
- **Feature Selection**: Removal of irrelevant and redundant variables
- **Data Validation**: Geological time range verification

### 2. Feature Engineering
- **Manual Features**: Domain-expert derived variables
- **Encoding Strategies**: Appropriate handling of categorical variables
- **Geographic Processing**: Coordinate transformations and regional binning
- **Leakage Prevention**: Careful exclusion of age-derived features

### 3. Model Development
- **Algorithm Selection**: Random Forest for interpretability and performance
- **Hyperparameter Tuning**: Optimized for generalization
- **Validation Strategy**: Rigorous train-test methodology
- **Performance Assessment**: Multiple evaluation metrics

### 4. Scientific Validation
- **Baseline Comparison**: Manual rule-based predictions
- **Feature Interpretation**: Biological/geological relevance
- **Error Analysis**: Systematic examination of prediction failures
- **Reproducibility**: Complete artifact preservation

## 🧬 Scientific Insights

### Key Discoveries
1. **Taxonomic Dominance**: Family and order classifications are the strongest predictors (73% combined importance)
2. **Ecological Significance**: Diet behavior contributes meaningfully to age prediction (10.4% importance)
3. **Geographic Patterns**: Location provides valuable temporal information (9.4% combined importance)
4. **High Predictability**: Trilobite characteristics strongly correlate with geological time

### Paleontological Implications
- **Evolutionary Trends**: Certain taxonomic groups have distinct temporal distributions
- **Ecological Evolution**: Feeding strategies changed predictably through time
- **Biogeography**: Geographic location correlates with age due to continental drift and faunal provinces

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? Let us know!
- 💡 **Feature Requests**: Have ideas for improvements?
- 📊 **New Datasets**: Additional fossil data always welcome
- 📚 **Documentation**: Help improve our docs
- 🔬 **Scientific Review**: Paleontology expertise appreciated

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{trilobite_age_prediction,
  title={Trilobite Age Prediction using Machine Learning},
  author={Babak Pirzadi},
  year={2025},
  url={https://github.com/bp198/trilobite-age-prediction},
  note={Machine learning pipeline for predicting fossil age from biological characteristics}
}
```

## 🙏 Acknowledgments

- **Dataset**: Thanks to the paleontological community for fossil data compilation
- **Scientific Community**: Researchers advancing ML applications in paleontology
- **Open Source**: Built on the shoulders of giants (scikit-learn, pandas, numpy)

## 📬 Contact

- **Author**: Babak Pirzadi
- **Email**: babak.pirzadi@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/babak-pirzadi-0824a59a](https://www.linkedin.com/in/babak-pirzadi-0824a59a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bmgy26nXOQ56Gh3zBMokJ%2FA%3D%3D)
- **GitHub**: [https://github.com/bp198](https://github.com/bp198)

---

## 🔗 Related Projects

- [Fossil Classification ML](link) - Automated fossil identification
- [Paleoclimate Reconstruction](link) - Climate modeling from fossil data
- [Evolutionary Timeline Analysis](link) - Phylogenetic temporal analysis

---

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for paleontology and machine learning*
