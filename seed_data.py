import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime
import time

DB_NAME = "gold_prices.db"

def seed_history():
    print("Fetching historical market data...")
    
    # Fetch last 30 days of data
    # GC=F is Gold Futures (USD per Ounce)
    # EGP=X is USD to EGP Exchange Rate
    gold = yf.Ticker("GC=F")
    egp = yf.Ticker("EGP=X")
    
    # Get historical data
    gold_hist = gold.history(period="1mo")
    egp_hist = egp.history(period="1mo")
    
    # Ensure indices are timezone-naive or matching
    gold_hist.index = gold_hist.index.tz_localize(None)
    egp_hist.index = egp_hist.index.tz_localize(None)
    
    # Merge on Date
    df = pd.merge(gold_hist[['Close']], egp_hist[['Close']], left_index=True, right_index=True, suffixes=('_gold', '_egp'))
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    print(f"Found {len(df)} days of historical data.")
    
    for date, row in df.iterrows():
        usd_price_oz = row['Close_gold']
        usd_egp_rate = row['Close_egp']
        
        # Calculation:
        # 1 Ounce = 31.1035 grams
        # Price per gram 24k (USD) = Price / 31.1035
        # Price per gram 24k (EGP) = (Price / 31.1035) * Exchange Rate
        
        price_gram_24_usd = usd_price_oz / 31.1035
        price_gram_24_egp = price_gram_24_usd * usd_egp_rate
        
        # Adjust for local market premium (approx 5% overhead/customs/market diff in Egypt)
        # This is a heuristic to match current street prices roughly
        market_premium = 1.02 
        
        p24 = round(price_gram_24_egp * market_premium, 2)
        p21 = round(p24 * 0.875, 2) # 21k is 87.5% gold
        p18 = round(p24 * 0.750, 2) # 18k is 75.0% gold
        
        # Check if date already exists to avoid duplicates
        # We use the date at 12:00:00 as a placeholder timestamp
        # Fix: Convert pandas Timestamp to python string or datetime object explicitly
        ts = date.to_pydatetime().replace(hour=12, minute=0, second=0)
        
        print(f"[{ts}] Inserting: 24k={p24}, 21k={p21}, 18k={p18}")
        
        # INSERT
        c.execute('''
            INSERT INTO prices (timestamp, karat_24, karat_21, karat_18)
            VALUES (?, ?, ?, ?)
        ''', (ts, p24, p21, p18))
        
    conn.commit()
    conn.close()
    print("Seeding complete.")

if __name__ == "__main__":
    seed_history()
