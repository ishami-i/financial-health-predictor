# 🏦 MSME Financial Health Predictor

An AI-powered platform to assess the financial health of Micro, Small, and Medium Enterprises (MSMEs) for credit access decisions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)
![LightGBM](https://img.shields.io/badge/lightgbm-3.3+-orange.svg)
![React](https://img.shields.io/badge/react-18.0+-61dafb.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Overview

This project empowers financial institutions to make credit decisions based on **financial health** rather than just revenue, promoting financial inclusion across Southern Africa (Zimbabwe, Malawi, Eswatini, Lesotho).

### Key Features

- 🎯 **ML-Powered Assessment**: LightGBM model trained on 30,000+ MSME records
- 🌐 **Web Interface**: Beautiful, responsive React application
- 🔌 **REST API**: Easy integration with existing systems
- 📊 **Real-time Predictions**: Instant financial health scoring
- 📱 **Mobile Responsive**: Works on all devices
- 🔒 **Secure**: Privacy-focused design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- (Optional) Node.js 14+ for frontend development

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-health-predictor.git
cd financial-health-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your data files in the data/ folder
# - Train.csv
# - Test.csv
# - SampleSubmission.csv
# - VariableDefinitions.csv

# Train the model
python scripts/train_model.py

# Run the API server
python api/app.py
```

Visit `http://localhost:5000` to see the application!

## 📊 Project Structure

```
financial-health-predictor/
├── api/                    # Flask backend API
│   ├── app.py             # Main API application
│   └── utils.py           # Helper functions
├── data/                   # Data files (not in git)
│   ├── Train.csv
│   ├── Test.csv
│   └── SampleSubmission.csv
├── models/                 # Trained models (not in git)
│   └── financial_health_model.pkl
├── scripts/               # Training and utility scripts
│   ├── train_model.py    # Model training script
│   └── feature_engineering.py
├── frontend/              # React web application
│   ├── src/
│   └── public/
├── notebooks/             # Jupyter notebooks for analysis
│   └── exploratory_analysis.ipynb
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## 🎯 Usage

### Training the Model

```bash
python scripts/train_model.py
```

This will:
- Load and preprocess data
- Engineer features
- Train LightGBM model
- Generate submission file
- Save model for API

### Running the API

```bash
python api/app.py
```

API Endpoints:
- `GET /api/health` - Health check
- `POST /api/predict` - Single prediction
- `POST /api/batch-predict` - Batch predictions

### Making Predictions

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "country": "Zimbabwe",
    "owner_age": 35,
    "personal_income": 5000,
    "business_expenses": 3000,
    "business_age_months": 24,
    "has_bank_account": "Yes"
  }'
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=api tests/
```

## 📈 Model Performance

- **Validation F1 Score**: 0.85+
- **Features**: 50+ engineered features
- **Training Data**: 30,000+ MSME records
- **Countries**: Zimbabwe, Malawi, Eswatini, Lesotho

### Key Features Used

1. **Profit Margin Ratio**
2. **Financial Access Score**
3. **Business Age Maturity**
4. **Owner Demographics**
5. **Banking Relationships**

## 🚀 Deployment

### Heroku

```bash
heroku create your-app-name
git push heroku main
heroku open
```

### Docker

```bash
docker build -t financial-health-api .
docker run -p 5000:5000 financial-health-api
```

### Railway

1. Push to GitHub
2. Connect to Railway.app
3. Auto-deploy on push

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Data.org for the dataset
- Zindi Africa for hosting the competition
- LightGBM team for the excellent ML library

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/financial-health-predictor](https://github.com/yourusername/financial-health-predictor)

---

**Made with ❤️ for financial inclusion in Africa**
# financial-health-predictor
