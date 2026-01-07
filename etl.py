import requests
from bs4 import BeautifulSoup
import sqlite3
import re
from datetime import datetime, timedelta
import time
import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
URL = "https://egypt.gold-price-today.com/"
DB_NAME = "gold_prices.db"

def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME UNIQUE,
            karat_24 REAL,
            karat_21 REAL,
            karat_18 REAL
        )
    ''')
    conn.commit()
    conn.close()

def get_min_timestamp():
    """Get the oldest timestamp currently in the DB."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT MIN(timestamp) FROM prices")
        result = c.fetchone()
        conn.close()
        if result and result[0]:
            return pd.to_datetime(result[0])
        return None
    except Exception:
        return None

def backfill_step():
    """Fetches ONE chunk of older history (going backwards from current oldest)."""
    try:
        oldest_date = get_min_timestamp()
        
        # If DB empty, start from today
        if oldest_date is None:
            end_date = datetime.now()
        else:
            end_date = oldest_date
            
        target_year = 2010
        if end_date.year < target_year:
            print(f"Info: History reached {target_year}. No more backfilling needed.")
            return

        # Go back 1 year (or until target)
        start_date = end_date - timedelta(days=365)
        
        print(f"⌛ Usage: Current oldest is {end_date.date()}. Extending back to {start_date.date()}...")

        # Fetch Data
        gold = yf.Ticker("GC=F")
        egp = yf.Ticker("EGP=X")
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        gold_hist = gold.history(start=start_str, end=end_str)
        egp_hist = egp.history(start=start_str, end=end_str)
        
        if gold_hist.empty or egp_hist.empty:
            print("⚠️ No market data found for this specific range. Skipping.")
            return

        gold_hist.index = gold_hist.index.tz_localize(None)
        egp_hist.index = egp_hist.index.tz_localize(None)
        
        df = pd.merge(gold_hist[['Close']], egp_hist[['Close']], left_index=True, right_index=True, suffixes=('_gold', '_egp'))
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        new_records = 0
        for date, row in df.iterrows():
            ts = date.to_pydatetime().replace(hour=12, minute=0, second=0)
            timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S")

            # Duplicate check
            c.execute("SELECT 1 FROM prices WHERE timestamp = ?", (timestamp_str,))
            if c.fetchone():
                continue

            usd_price_oz = row['Close_gold']
            usd_egp_rate = row['Close_egp']
            
            # --- CURRENCY CORRECTION HEURISTIC ---
            # During 2022-2023, Official Rate != Parallel Rate
            # We apply a rough multiplier if the rate looks suspiciously low compared to street history
            # This is a basic patch to avoid the "U" shape graph
            year = date.year
            if 2022 <= year <= 2023:
                # Basic correction for parallel market divergence
                # In late 2023 divergence was ~60 vs 30 (2.0x)
                # In 2022 it was ~30 vs 24 (1.25x)
                if usd_egp_rate < 35: 
                    usd_egp_rate *= 1.4 # Average correction factor
            
            # Calculation
            price_gram_24_usd = usd_price_oz / 31.1035
            price_gram_24_egp = price_gram_24_usd * usd_egp_rate
            market_premium = 1.02
            
            p24 = round(price_gram_24_egp * market_premium, 2)
            p21 = round(p24 * 0.875, 2)
            p18 = round(p24 * 0.750, 2)
            
            c.execute('''
                INSERT INTO prices (timestamp, karat_24, karat_21, karat_18)
                VALUES (?, ?, ?, ?)
            ''', (timestamp_str, p24, p21, p18))
            new_records += 1
            
        conn.commit()
        conn.close()
        print(f"✅ Backfill Batch Success: Added {new_records} older records.")
            
    except Exception as e:
        print(f"Backfill error: {e}")

def get_gold_prices():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(URL, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        prices = {}
        
        rows = soup.find_all('tr')
        for row in rows:
            header = row.find('th')
            if not header: continue
            header_text = header.get_text().strip()
            
            cells = row.find_all('td')
            if not cells: continue
            price_text = cells[0].get_text().strip()
            
            if "عيار 24" in header_text: prices['24'] = extract_price(price_text)
            elif "عيار 21" in header_text: prices['21'] = extract_price(price_text)
            elif "عيار 18" in header_text: prices['18'] = extract_price(price_text)
                
        return prices
    except Exception as e:
        print(f"Error scraping: {e}")
        return None

def extract_price(text):
    match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        num_str = match.group(0).replace(',', '')
        return float(num_str)
    return 0.0

def save_to_db(prices):
    if not prices or len(prices) < 3: return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    ts = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Saving Live Data: 24k={prices.get('24')}, 21k={prices.get('21')}, 18k={prices.get('18')}")
    
    c.execute('''
        INSERT INTO prices (timestamp, karat_24, karat_21, karat_18)
        VALUES (?, ?, ?, ?)
    ''', (ts, prices.get('24'), prices.get('21'), prices.get('18')))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_db()
    print("Starting GoldTracker ETL Service (Live + Aggressive Backfill)...")
    
    LIVE_INTERVAL = 1800 # 30 Minutes
    
    while True:
        cycle_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Live Cycle Started.")
        
        # 1. Scrape Live (Priority)
        print("Fetching live prices...")
        data = get_gold_prices()
        if data:
            save_to_db(data)
        else:
            print("Failed to fetch live data.")
            
        # 2. Backfill while waiting for the next live cycle
        print(f"Backfilling history active for the next {LIVE_INTERVAL/60} minutes...")
        
        while (time.time() - cycle_start_time) < LIVE_INTERVAL:
            # Check if we are done (e.g. reached 2010)
            oldest = get_min_timestamp()
            if oldest and oldest.year <= 2010:
                remaining = LIVE_INTERVAL - (time.time() - cycle_start_time)
                print(f"History fully backfilled to 2010. Sleeping remaining {int(remaining)}s...")
                time.sleep(remaining)
                break
            
            # Perform one backfill step
            backfill_step()
            
            # Small pause to avoid rate limits, but keep going
            time.sleep(10)
            
            # Check time again to ensure we don't overrun live update
            if (time.time() - cycle_start_time) >= LIVE_INTERVAL:
                print("Time for next live update.")
                break
