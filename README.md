# 🏆 GoldTracker EGP: AI-Powered Gold Analytics

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

An advanced real-time dashboard for tracking, analyzing, and forecasting Egyptian Gold prices. Built for investors and technical analysts.

## 🌟 Features

### 🔍 Real-Time Data Pipeline (ETL)
- **Automated Scraping**: Python script fetches live buy/sell prices for 24k, 21k, and 18k gold every 30 minutes from local market sources.
- **Robust Storage**: Data is historized in a SQLite database, allowing for long-term trend analysis.

### 📈 Interactive Dashboard
- **Advanced Charting**: Pure Line Chart interface with direct price labels, range sliders, and zoom controls (1D, 1W, 1Y, All).
- **Technical Analysis Suite**: Toggleable professional indicators:
  - **SMA 20** (Fair Value / Support)
  - **EMA 50** (Trend Strength)
  - **Bollinger Bands** (Volatility & Breakouts)
  - **RSI** (Momentum & Overbought/Oversold detection)

### 🧠 AI Analyst (Smart Signals)
- **Automated Trading Signals**: The system analyzes technical indicators in real-time to generate **"Strong Buy"**, **"Sell"**, or **"Hold"** recommendations.
- **Dynamic Targets**: Calculates realistic "Fair Value" entry points (based on Mean Reversion) and "Taking Profit" zones.

### 📰 Market Intelligence
- **News Aggregator**: Integrated Google News feed (Arabic) to display the latest local market updates directly in the dashboard.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Visualization**: Plotly Interactive Charts
- **Backend/ETL**: Python (`requests`, `beautifulsoup4`, `pandas`)
- **Database**: SQLite
- **Machine Learning**: `scikit-learn` (Linear Regression for price forecasting)

## 🚀 How to Run Locally

1. **Clone the Repo**
   ```bash
   git clone https://github.com/yourusername/goldtracker-egp.git
   cd goldtracker-egp
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Scraper (Populate Data)**
   ```bash
   python etl.py
   ```

4. **Launch Dashboard**
   ```bash
   streamlit run dashboard.py
   ```

## 📊 Preview
*(Add a screenshot of your dashboard here)*
