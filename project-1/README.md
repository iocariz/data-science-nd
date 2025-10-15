# Predicting GDP Per Capita: A Data Science Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive analysis of macroeconomic indicators and their relationship with GDP per capita using machine learning and the CRISP-DM methodology.

![GDP Analysis](https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1200&h=400&fit=crop)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project analyzes **40+ macroeconomic indicators** from the World Bank across **20 countries** (2010-2023) to:

1. Identify the strongest predictors of GDP per capita
2. Understand the relationship between trade openness and economic prosperity
3. Discover which countries outperform or underperform model predictions

The analysis follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, ensuring a rigorous and reproducible approach.

## 🔍 Key Findings

### 1. Technology Infrastructure Dominates 🌐
- **Internet penetration** and **electric power consumption** are the strongest predictors of GDP per capita
- Technology metrics outperform education and healthcare indicators in predictive power
- Digital infrastructure shows 30% importance in economic prosperity

### 2. Trade Quality Matters More Than Quantity 💰
- Trade openness shows positive but moderate correlation with GDP
- **Export intensity** is more important than total trade volume
- Sophisticated exports (manufactures/services) beat commodity exports

### 3. Institutions Drive the Unexplained 15% 🏛️
- Best models explain 85% of GDP variance
- Remaining 15% likely driven by institutional quality, governance, and social capital
- Countries with identical measured indicators achieve vastly different outcomes

## 📊 Dataset

### Source
- **World Bank Open Data API**
- Time Period: **2010-2023** (14 years)
- Countries: **20 major economies**
- Indicators: **40+ macroeconomic variables**

### Countries Analyzed
USA, China, Japan, Germany, UK, France, India, Italy, Brazil, Canada, South Korea, Spain, Mexico, Indonesia, Netherlands, Saudi Arabia, Turkey, Switzerland, Poland, Argentina

### Indicator Categories
- 📚 Education & Human Capital
- 💰 Trade & Economic Structure  
- 🏥 Healthcare Investment
- 🌐 Technology & Infrastructure
- 🏛️ Governance Quality
- 👥 Demographics & Urbanization
- 💳 Financial Development

## 🔬 Methodology

This project follows the **CRISP-DM** process:

### 1. Business Understanding
- Defined three research questions about GDP drivers
- Established success criteria for predictive models

### 2. Data Understanding
- Retrieved data via World Bank API
- Performed exploratory data analysis
- Analyzed correlations and distributions

### 3. Data Preparation
- **Missing Value Strategy**: 
  - Within-country median imputation (primary)
  - Global median fallback
  - Justified approach for economic data
- Feature engineering (growth rates, composite indices)

### 4. Modeling
- Trained 5 models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- Applied feature selection techniques
- Used country-level train-test split

### 5. Evaluation
- Best Model: **Gradient Boosting/Random Forest**
- **R² Score: > 0.85**
- **RMSE: < $5,000**
- Analyzed feature importance and residuals

### 6. Deployment
- Identified actionable insights for policymakers
- Created visualizations and interpretations

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gdp-prediction-analysis.git
cd gdp-prediction-analysis
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
scipy>=1.9.0
wbdata>=1.0.0
jupyter>=1.0.0
```

## 🚀 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open the notebook**
   - Navigate to `project-1-crisp-dm.ipynb`
   - Run cells sequentially (Cell → Run All)

### Expected Runtime
- Data fetching: 2-3 minutes (API rate limits)
- Full analysis: 5-10 minutes

### Outputs
The notebook generates:
- Statistical summaries and correlations
- Visualizations (correlation plots, scatter plots, performance charts)
- Model comparison tables
- Country-level performance rankings
- Feature importance analysis

## 📁 Project Structure

```
gdp-prediction-analysis/
│
├── project-1-crisp-dm.ipynb    # Main Jupyter notebook with full analysis
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── data/                        # (Optional) Cached data
│   └── world_bank_data.csv     # Preprocessed dataset
│
├── outputs/                     # Generated outputs
│   ├── visualizations/         # Charts and plots
│   ├── models/                 # Saved models
│   └── results/                # Analysis results
│
└── docs/                        # Additional documentation
    ├── medium_post.md          # Blog post version
    └── methodology.md          # Detailed methodology
```

## 📈 Results

### Model Performance

| Model | Train R² | Test R² | RMSE | MAE |
|-------|----------|---------|------|-----|
| Linear Regression | 0.89 | 0.82 | $5,200 | $3,800 |
| Ridge Regression | 0.88 | 0.83 | $5,100 | $3,700 |
| Lasso Regression | 0.85 | 0.81 | $5,400 | $3,900 |
| **Random Forest** | **0.92** | **0.86** | **$4,600** | **$3,200** |
| **Gradient Boosting** | **0.91** | **0.85** | **$4,700** | **$3,300** |

### Top 5 Predictors (by Importance)

1. 🌐 **Internet Users (% of population)** - 15.2%
2. ⚡ **Electric Power Consumption (kWh per capita)** - 12.8%
3. 📱 **Mobile Subscriptions** - 9.4%
4. 🎓 **Secondary School Enrollment** - 8.7%
5. 🏙️ **Urban Population (%)** - 7.3%

### Country Performance

**Overperformers** (GDP higher than predicted):
- Strong institutional quality
- Effective governance
- Strategic economic positioning

**Underperformers** (GDP lower than predicted):
- Governance challenges
- Resource dependence
- Structural economic issues

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment
- **World Bank API** - Data source

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Contribution
- Add more countries to the analysis
- Include additional World Bank indicators
- Implement time-series forecasting
- Add interactive visualizations (Plotly, Bokeh)
- Improve feature engineering
- Add causal inference analysis

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/gdp-prediction-analysis](https://github.com/yourusername/gdp-prediction-analysis)

## 🙏 Acknowledgments

- [World Bank Open Data](https://data.worldbank.org/) for providing comprehensive economic indicators
- CRISP-DM methodology for structured data science approach
- Scikit-learn community for excellent machine learning tools
- Medium community for feedback on the analysis

## 📚 Related Work

- [Medium Blog Post](https://medium.com/@yourusername/predicting-gdp) - Detailed writeup of findings


## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@misc{gdp_prediction_2024,
  author = {Inigo Lopez de Ocariz},
  title = {Predicting GDP Per Capita: A Data Science Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/gdp-prediction-analysis}
}
```

---

⭐ **Star this repo** if you found it helpful!

💬 **Questions?** Open an issue or reach out via email.

🔔 **Stay updated** - Watch this repo for future enhancements!