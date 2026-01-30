# 🏆 GoldTracker EGP

**GoldTracker EGP** is a premium, real-time dashboard for tracking gold prices in Egypt. It combines live global market data with local exchange rates to provide accurate, up-to-the-minute gold prices in Egyptian Pounds (EGP).

![GoldTracker Dashboard](https://raw.githubusercontent.com/placeholder/image.png)

## ✨ Features

- **Live Gold Prices**: Real-time tracking for **24K, 21K, 18K, and 14K** gold in EGP.
- **Global Markets**: Dedicated card for the **Global Ounce (XAU/USD)**.
- **Interactive Charts**: 
  - Switch between **Local (EGP)** and **Global (USD)** views.
  - Historical data from 1 Day to All-Time.
  - Zoom, pan, and advanced tooltips.
- **Market News**: Curated feed of the latest gold news in Arabic.
- **Premium Design**: Dark theme with "Gold Luxury" aesthetics, glassmorphism effects, and micro-animations.
- **Mobile Optimized**: Fully responsive layout for phones and tablets.
- **Reliable Data**: Automatic background collection, crash recovery, and data gap backfilling.

## 🛠️ Tech Stack

- **Backend**: Python (FastAPI), SQLite, yfinance (Yahoo Finance API).
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

5.  **Access the Dashboard:**
    Open your browser to `http://localhost:8000`.

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
    - `index.html`: Main dashboard UI.
    - `css/styles.css`: Premium dark theme styles.
    - `js/app.js`: Frontend logic and chart rendering.

## 📄 License
MIT License.
