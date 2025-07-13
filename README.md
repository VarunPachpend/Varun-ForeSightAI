# UtiliCast - AI-Driven Forecasting for Multi-Purpose Tractor Demand

## 🎯 Project Overview

UtiliCast is an AI-powered forecasting and decision-support system for tractor sales in the U.S. market, specifically focused on multi-purpose utility tractors used for landscaping, hauling, mowing, digging, and general property maintenance.

**Team:** ForeSight AI  
**Tagline:** "Seeing Beyond the Data"  
**Primary Color:** #E31837 (bold red)

## 🚀 Features

### Core Functionality
- **Advanced AI-Driven Demand Forecasting**: Utilizes Prophet, XGBoost, & LSTM for robust multi-purpose tractor sales prediction
- **Dynamic External Indicator Integration**: Incorporates precipitation, housing starts, & economic activity data
- **Interactive Decision-Support Dashboard**: Visualizes forecasts and provides actionable recommendations

### Dashboard Capabilities
- **Timeline Visualization**: View data by Month, Quarter, and Year
- **Product & Brand Analysis**: Filter by Product Name & Brand Name
- **PTO HP Category Analysis**: Analyze data across 0<20, 20<30, 30<40, 40<50, 50<60, and 60<70 HP ranges
- **Multiple Chart Types**: Bar charts, Pie charts, Scatter plots, and more
- **Interactive USA Map**: State-level data visualization with drill-down capabilities
- **Future Trend Forecasting**: Current date visualization with future predictions

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard:**
   Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
utilicast/
├── app.py                          # Main Streamlit application
├── data/                           # Data directory
│   ├── sample_data.py             # Sample data generation
│   └── external_data.py           # External data integration
├── models/                         # ML models
│   ├── prophet_model.py           # Prophet forecasting
│   ├── xgboost_model.py           # XGBoost forecasting
│   └── lstm_model.py              # LSTM forecasting
├── utils/                          # Utility functions
│   ├── data_processor.py          # Data preprocessing
│   ├── visualization.py           # Chart generation
│   └── map_utils.py               # Map visualization
├── pages/                          # Dashboard pages
│   ├── overview.py                # Overview page
│   ├── forecasting.py             # Forecasting page
│   ├── regional_analysis.py       # Regional analysis
│   └── recommendations.py         # Recommendations page
├── requirements.txt                # Python dependencies
└── README.md                      # Project documentation
```

## 🎨 Dashboard Features

### 1. Overview Dashboard
- Executive summary with key metrics
- Current market trends
- Quick insights and alerts

### 2. Forecasting Analysis
- Multi-model forecasting (Prophet, XGBoost, LSTM)
- Interactive timeline selection
- Forecast accuracy metrics

### 3. Regional Analysis
- Interactive USA map with state-level data
- Drill-down capabilities for detailed state analysis
- Regional comparison tools

### 4. Product Analysis
- PTO HP category breakdown
- Brand performance analysis
- Product trend visualization

### 5. Recommendations
- AI-generated actionable insights
- Inventory optimization suggestions
- Strategic recommendations for management

## 🔧 Configuration

The dashboard uses sample data by default. To integrate with real data sources:

1. Update `data/external_data.py` with your API keys
2. Modify data sources in the configuration files
3. Update the data processing pipeline as needed

## 📊 Data Sources

- **Historical Tractor Sales**: PTO HP categorized data (0<20, 20<30, 30<40, 40<50, 50<60, 60<70)
- **External Indicators**: 
  - Precipitation trends (NOAA)
  - Housing starts (U.S. Census Bureau)
  - Regional economic activity (BEA, FRED)

## 🤖 AI Models

- **Prophet**: Baseline seasonality and trend analysis
- **XGBoost**: Feature importance and ensemble predictions
- **LSTM**: Long-term trend forecasting with neural networks

## 📈 Key Performance Indicators

- **Forecast Accuracy (MAPE)**: Target < 5%
- **Inventory Optimization Rate**: 15-20% reduction in holding costs
- **Production Alignment Score**: > 90% alignment
- **Regional Sales Growth**: 10-15% growth in targeted regions

## 🚀 Deployment

The application is designed to run locally with Streamlit. For production deployment:

1. Use Docker containerization
2. Set up scheduled model retraining
3. Configure external data API integrations
4. Implement user authentication if required

## 📝 License

This project is developed for the Ideathon'25 competition under the RecomME – AI for Next Big Move track.

## 👥 Team

**ForeSight AI** - Developing innovative AI solutions for strategic business decisions.

---

*"Seeing Beyond the Data" - UtiliCast* 