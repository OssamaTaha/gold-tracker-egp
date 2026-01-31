# 🏆 GoldTracker EGP

**GoldTracker EGP** is a premium, real-time dashboard for tracking gold and silver prices in Egypt. It combines live global market data with local exchange rates to provide accurate, up-to-the-minute precious metal prices in Egyptian Pounds (EGP).

![GoldTracker Dashboard](https://raw.githubusercontent.com/placeholder/image.png)

## ✨ Features

### 🥇 Gold Tracker (`/`)
- **Live Gold Prices**: Real-time tracking for **24K, 21K, 18K, and 14K** gold in EGP.
- **Global Markets**: Dedicated card for the **Global Ounce (XAU/USD)**.

### 🥈 Silver Tracker (`/silver`)
- **Live Silver Prices**: Real-time tracking for **999, 925, 900, and 800** purity silver in EGP.
- **Egyptian Market Prices**: Includes local market premium for accurate Egyptian pricing.
- **Global Markets**: Dedicated card for the **Global Ounce (XAG/USD)**.

### 📊 Common Features
- **Interactive Charts**: 
  - Switch between different karats/purities.
  - Historical data from 1 Day to All-Time.
  - Zoom, pan, and advanced tooltips.
- **Market News**: Curated feed of the latest precious metals news in Arabic.
- **Premium Design**: Dark theme with luxury aesthetics, glassmorphism effects, and micro-animations.
- **Mobile Optimized**: Fully responsive layout for phones and tablets.
- **Reliable Data**: Uses Gold-API.com for accurate real-time prices, with yfinance as fallback.

## 🛠️ Tech Stack

- **Backend**: Python (FastAPI), SQLite, yfinance, Gold-API.com.
- **Frontend**: Vanilla JavaScript, CSS3 (Variables, Grid, Flexbox), HTML5.
- **Charts**: TradingView Lightweight Charts.

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/gold-tracker-egp.git
    cd gold-tracker-egp
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python api.py
    ```

5.  **Access the Dashboards:**
    - **Gold Tracker**: `http://localhost:8000`
    - **Silver Tracker**: `http://localhost:8000/silver`

## 🌐 Live Demo / Hosting

To share the application with others over the internet, we recommend using **ngrok**:

```bash
# Install ngrok and authenticate
ngrok config add-authtoken <YOUR_TOKEN>

# Expose port 8000
ngrok http 8000
```
Then share the generated HTTPS link.

## 📁 Project Structure

- `api.py`: Main FastAPI application and background workers.
- `gold_prices.db`: SQLite database storing historical price data.
- `frontend/`: Static assets.
    - `index.html`: Gold dashboard UI.
    - `silver.html`: Silver dashboard UI.
    - `css/styles.css`: Gold theme styles.
    - `css/silver.css`: Silver theme styles.
    - `js/app.js`: Gold dashboard logic.
    - `js/silver-app.js`: Silver dashboard logic.

## 🔗 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/prices/current` | Live gold prices (all karats) |
| `GET /api/prices/ohlc` | Gold historical OHLC data |
| `GET /api/silver/prices/current` | Live silver prices (all purities) |
| `GET /api/silver/prices/ohlc` | Silver historical OHLC data |
| `GET /api/news` | Gold market news |
| `GET /api/silver/news` | Silver market news |

## 📄 License
MIT License.

