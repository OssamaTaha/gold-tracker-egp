"""
GoldTracker EGP - FastAPI Backend
Premium Gold Price Tracking for Egypt

Features:
- Live gold prices from yfinance (GC=F × USDEGP=X)
- Background data collection every 5 minutes
- Crash recovery with timestamp tracking
- Historical data storage in SQLite
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
import re
from typing import Optional
import os
import pytz
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Egypt timezone
EGYPT_TZ = pytz.timezone('Africa/Cairo')

DB_NAME = "gold_prices.db"
STATE_FILE = "collector_state.json"
COLLECTION_INTERVAL = 5 * 60  # 5 minutes in seconds

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_database():
    """Initialize fresh database with proper schema."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create prices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            gold_usd_oz REAL NOT NULL,
            usd_egp_rate REAL NOT NULL,
            karat_24 REAL NOT NULL,
            karat_21 REAL NOT NULL,
            karat_18 REAL NOT NULL,
            karat_14 REAL NOT NULL,
            source TEXT DEFAULT 'yfinance',
            UNIQUE(timestamp)
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prices(timestamp DESC)')
    
    # Create state table for crash recovery
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS collector_state (
            id INTEGER PRIMARY KEY,
            last_collection DATETIME NOT NULL,
            status TEXT DEFAULT 'running'
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully")


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# LIVE DATA FETCHING
# ============================================================================

def fetch_gold_from_api():
    """Fetch live gold price from Gold-API (more accurate real-time data)."""
    try:
        response = requests.get("https://api.gold-api.com/price/XAU", timeout=10)
        response.raise_for_status()
        data = response.json()
        gold_usd_oz = float(data.get("price", 0))
        if gold_usd_oz > 0:
            logger.info(f"🥇 Gold-API: ${gold_usd_oz:.2f}/oz")
            return gold_usd_oz
    except Exception as e:
        logger.warning(f"⚠️ Gold-API failed: {e}")
    return None


def fetch_gold_from_yfinance():
    """Fallback: Fetch gold price from yfinance futures."""
    try:
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(period="1d")
        if not gold_data.empty:
            price = float(gold_data['Close'].iloc[-1])
            logger.info(f"📊 yfinance fallback: ${price:.2f}/oz")
            return price
    except Exception as e:
        logger.warning(f"⚠️ yfinance gold failed: {e}")
    return None


def fetch_live_gold_price():
    """Fetch live gold price using Gold-API (primary) or yfinance (fallback)."""
    try:
        # Try Gold-API first (more accurate real-time spot price)
        gold_usd_oz = fetch_gold_from_api()
        
        # Fallback to yfinance if Gold-API fails
        if gold_usd_oz is None:
            gold_usd_oz = fetch_gold_from_yfinance()
        
        if gold_usd_oz is None:
            logger.warning("⚠️ No gold data from any source")
            return None
        
        # Get USD/EGP exchange rate from yfinance
        egp = yf.Ticker("USDEGP=X")
        egp_data = egp.history(period="1d")
        
        if egp_data.empty:
            logger.warning("⚠️ No USD/EGP data from yfinance")
            return None
        
        usd_egp_rate = float(egp_data['Close'].iloc[-1])
        
        # Calculate 24K price in EGP per gram (31.1035 grams per troy ounce)
        karat_24 = (gold_usd_oz / 31.1035) * usd_egp_rate
        
        return {
            "timestamp": datetime.now(EGYPT_TZ),
            "gold_usd_oz": round(gold_usd_oz, 2),
            "usd_egp_rate": round(usd_egp_rate, 4),
            "karat_24": round(karat_24, 2),
            "karat_21": round(karat_24 * (21/24), 2),
            "karat_18": round(karat_24 * (18/24), 2),
            "karat_14": round(karat_24 * (14/24), 2)
        }
    except Exception as e:
        logger.error(f"❌ Error fetching live price: {e}")
        return None


# ============================================================================
# SILVER PRICE FETCHING
# ============================================================================

def fetch_silver_from_api():
    """Fetch live silver price from Gold-API (XAG)."""
    try:
        response = requests.get("https://api.gold-api.com/price/XAG", timeout=10)
        response.raise_for_status()
        data = response.json()
        silver_usd_oz = float(data.get("price", 0))
        if silver_usd_oz > 0:
            logger.info(f"🥈 Silver-API: ${silver_usd_oz:.2f}/oz")
            return silver_usd_oz
    except Exception as e:
        logger.warning(f"⚠️ Silver-API failed: {e}")
    return None


def fetch_silver_from_yfinance():
    """Fallback: Fetch silver price from yfinance futures."""
    try:
        silver = yf.Ticker("SI=F")
        silver_data = silver.history(period="1d")
        if not silver_data.empty:
            price = float(silver_data['Close'].iloc[-1])
            logger.info(f"📊 yfinance silver fallback: ${price:.2f}/oz")
            return price
    except Exception as e:
        logger.warning(f"⚠️ yfinance silver failed: {e}")
    return None


def fetch_live_silver_price():
    """Fetch live silver price using Gold-API (primary) or yfinance (fallback)."""
    try:
        # Try Gold-API first for accurate real-time spot price
        silver_usd_oz = fetch_silver_from_api()
        
        # Fallback to yfinance if Gold-API fails
        if silver_usd_oz is None:
            silver_usd_oz = fetch_silver_from_yfinance()
        
        if silver_usd_oz is None:
            logger.warning("⚠️ No silver data from any source")
            return None
        
        # Get USD/EGP exchange rate from yfinance
        egp = yf.Ticker("USDEGP=X")
        egp_data = egp.history(period="1d")
        
        if egp_data.empty:
            logger.warning("⚠️ No USD/EGP data from yfinance")
            return None
        
        usd_egp_rate = float(egp_data['Close'].iloc[-1])
        
        # Calculate silver purity prices in EGP per gram (31.1035 grams per troy ounce)
        # Apply Egyptian market premium (~11%) to match local silver prices
        EGYPT_SILVER_PREMIUM = 1.11  # Egyptian market premium over global spot
        silver_999_spot = (silver_usd_oz / 31.1035) * usd_egp_rate
        silver_999 = silver_999_spot * EGYPT_SILVER_PREMIUM
        
        return {
            "timestamp": datetime.now(EGYPT_TZ),
            "silver_usd_oz": round(silver_usd_oz, 2),
            "usd_egp_rate": round(usd_egp_rate, 4),
            "purity_999": round(silver_999, 2),           # 99.9% pure
            "purity_925": round(silver_999 * (925/999), 2),   # Sterling silver
            "purity_900": round(silver_999 * (900/999), 2),   # Coin silver
            "purity_800": round(silver_999 * (800/999), 2)    # European silver
        }
    except Exception as e:
        logger.error(f"❌ Error fetching live silver price: {e}")
        return None


def save_price_to_db(price_data: dict):
    """Save price data to database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO prices 
            (timestamp, gold_usd_oz, usd_egp_rate, karat_24, karat_21, karat_18, karat_14, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'yfinance')
        ''', (
            price_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            price_data['gold_usd_oz'],
            price_data['usd_egp_rate'],
            price_data['karat_24'],
            price_data['karat_21'],
            price_data['karat_18'],
            price_data['karat_14']
        ))
        
        # Update collector state
        cursor.execute('''
            INSERT OR REPLACE INTO collector_state (id, last_collection, status)
            VALUES (1, ?, 'running')
        ''', (price_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),))
        
        conn.commit()
        conn.close()
        logger.info(f"💰 Price saved: 24K = {price_data['karat_24']} EGP @ {price_data['timestamp'].strftime('%H:%M:%S')}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving price: {e}")
        return False


def get_last_collection_time():
    """Get the last collection timestamp for crash recovery."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT last_collection FROM collector_state WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=EGYPT_TZ)
        return None
    except Exception as e:
        logger.warning(f"⚠️ Could not get last collection time: {e}")
        return None


async def seed_historical_data():
    """
    Populate database with historical data if empty.
    Fetches:
    1. Daily data for max history (2010-now)
    2. Hourly data for last 730 days (2 years)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if we have significant history (more than 100 records)
        cursor.execute("SELECT COUNT(*) FROM prices")
        count = cursor.fetchone()[0]
        
        if count > 100:
            logger.info(f"✅ Database already has {count} records, skipping seed.")
            conn.close()
            return

        logger.info("📉 Seeding historical data from yfinance...")
        
        # 1. Fetch Daily Data (Max History)
        logger.info("   Fetching daily data (2010-present)...")
        gold = yf.Ticker("GC=F")
        egp = yf.Ticker("USDEGP=X")
        
        gold_daily = gold.history(period="max")
        egp_daily = egp.history(period="max")
        
        # Align indexes (normalize timezone to ensure match)
        gold_daily.index = gold_daily.index.tz_localize(None)
        egp_daily.index = egp_daily.index.tz_localize(None)
        
        daily_idx = gold_daily.index.intersection(egp_daily.index)
        
        # Prepare daily records
        daily_records = []
        for ts in daily_idx:
            # Skip if older than 2010
            if ts.year < 2010:
                continue
                
            gold_price = float(gold_daily.loc[ts, 'Close'])
            egp_rate = float(egp_daily.loc[ts, 'Close'])
            karat_24 = (gold_price / 31.1035) * egp_rate
            
            daily_records.append((
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                round(gold_price, 2),
                round(egp_rate, 4),
                round(karat_24, 2),
                round(karat_24 * (21/24), 2),
                round(karat_24 * (18/24), 2),
                round(karat_24 * (14/24), 2),
                'yfinance-history-daily'
            ))
            
        # 2. Fetch Hourly Data (Last 2 Years)
        logger.info("   Fetching hourly data (last 2 years)...")
        gold_hourly = gold.history(period="2y", interval="1h")
        egp_hourly = egp.history(period="2y", interval="1h")
        
        # Normalize timezones for hourly too
        gold_hourly.index = gold_hourly.index.tz_localize(None)
        egp_hourly.index = egp_hourly.index.tz_localize(None)
        
        hourly_idx = gold_hourly.index.intersection(egp_hourly.index)
        
        hourly_records = []
        for ts in hourly_idx:
            gold_price = float(gold_hourly.loc[ts, 'Close'])
            egp_rate = float(egp_hourly.loc[ts, 'Close'])
            karat_24 = (gold_price / 31.1035) * egp_rate
            
            hourly_records.append((
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                round(gold_price, 2),
                round(egp_rate, 4),
                round(karat_24, 2),
                round(karat_24 * (21/24), 2),
                round(karat_24 * (18/24), 2),
                round(karat_24 * (14/24), 2),
                'yfinance-history-hourly'
            ))
            
        # 3. Bulk Insert (Use INSERT OR REPLACE to prefer hourly over daily for same timestamp)
        logger.info(f"   Inserting {len(daily_records)} daily records...")
        cursor.executemany('''
            INSERT OR IGNORE INTO prices 
            (timestamp, gold_usd_oz, usd_egp_rate, karat_24, karat_21, karat_18, karat_14, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', daily_records)
        
        logger.info(f"   Inserting {len(hourly_records)} hourly records...")
        # Hourly data provides better granularity, so we insert it. 
        # Since timestamps include time, they won't conflict with daily (usually 00:00 or market close)
        # except purely by coincidence, which is fine.
        cursor.executemany('''
            INSERT OR IGNORE INTO prices 
            (timestamp, gold_usd_oz, usd_egp_rate, karat_24, karat_21, karat_18, karat_14, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', hourly_records)
        
        conn.commit()
        conn.close()
        logger.info("✅ Historical data dump complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to seed history: {e}")

async def backfill_missing_data():
    """Backfill any missing data since last collection (crash recovery)."""
    # First, run the full historical seed if needed
    await seed_historical_data()
    
    last_collection = get_last_collection_time()
    
    if last_collection is None:
        logger.info("📊 No previous data - starting fresh collection")
        return
    
    now = datetime.now(EGYPT_TZ)
    gap_minutes = (now - last_collection).total_seconds() / 60
    
    if gap_minutes > 10:  # More than 10 minutes gap
        logger.info(f"🔄 Detected {gap_minutes:.0f} minute gap since last collection. Backfilling...")
        
        try:
            # Fetch historical data to fill the gap
            gold = yf.Ticker("GC=F")
            egp = yf.Ticker("USDEGP=X")
            
            # Determine appropriate period for yfinance
            if gap_minutes <= 60 * 24:
                period = "1d"
            elif gap_minutes <= 60 * 24 * 5:
                period = "5d"
            elif gap_minutes <= 60 * 24 * 30:
                period = "1mo"
            else:
                period = "3mo"
            
            logger.info(f"📅 Fetching backfill data with period='{period}'")
            gold_data = gold.history(period=period, interval="5m")
            egp_data = egp.history(period=period, interval="5m")
            
            if not gold_data.empty and not egp_data.empty:
                # Align the data
                common_times = gold_data.index.intersection(egp_data.index)
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                backfilled = 0
                for ts in common_times:
                    if ts.tz_localize(None) > last_collection.replace(tzinfo=None):
                        gold_price = float(gold_data.loc[ts, 'Close'])
                        egp_rate = float(egp_data.loc[ts, 'Close'])
                        karat_24 = (gold_price / 31.1035) * egp_rate
                        
                        try:
                            cursor.execute('''
                                INSERT OR IGNORE INTO prices 
                                (timestamp, gold_usd_oz, usd_egp_rate, karat_24, karat_21, karat_18, karat_14, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, 'yfinance-backfill')
                            ''', (
                                ts.strftime("%Y-%m-%d %H:%M:%S"),
                                round(gold_price, 2),
                                round(egp_rate, 4),
                                round(karat_24, 2),
                                round(karat_24 * (21/24), 2),
                                round(karat_24 * (18/24), 2),
                                round(karat_24 * (14/24), 2)
                            ))
                            backfilled += 1
                        except:
                            pass
                
                conn.commit()
                conn.close()
                logger.info(f"✅ Backfilled {backfilled} data points")
        except Exception as e:
            logger.error(f"❌ Backfill error: {e}")
    else:
        logger.info(f"✅ Last collection was {gap_minutes:.0f} minutes ago - no backfill needed")


# ============================================================================
# BACKGROUND COLLECTOR
# ============================================================================

async def price_collector():
    """Background task to collect prices every 5 minutes."""
    logger.info("🚀 Starting price collector (every 5 minutes)")
    
    # Initial collection
    price = fetch_live_gold_price()
    if price:
        save_price_to_db(price)
    
    while True:
        await asyncio.sleep(COLLECTION_INTERVAL)
        try:
            price = fetch_live_gold_price()
            if price:
                save_price_to_db(price)
        except Exception as e:
            logger.error(f"❌ Collection error: {e}")


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("🏆 Starting GoldTracker EGP API Server...")
    init_database()
    await backfill_missing_data()
    
    # Start background collector
    collector_task = asyncio.create_task(price_collector())
    
    logger.info("📊 Dashboard: http://localhost:8000")
    logger.info("📚 API Docs:  http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    collector_task.cancel()
    try:
        await collector_task
    except asyncio.CancelledError:
        pass
    logger.info("👋 Server shutdown complete")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="GoldTracker EGP API",
    description="Real-time Gold Price Tracking for Egypt",
    version="2.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_karat_prices(price_24k: float) -> dict:
    """Calculate all karat prices from 24K base price."""
    return {
        "karat_24": round(price_24k, 2),
        "karat_21": round(price_24k * (21/24), 2),
        "karat_18": round(price_24k * (18/24), 2),
        "karat_14": round(price_24k * (14/24), 2),
        "ounce": round(price_24k * 31.1035, 2)
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/prices/current")
async def get_current_prices():
    """
    Get LIVE gold prices for all karats.
    Uses Gold-API for accurate spot prices, yfinance as fallback.
    """
    try:
        # Try Gold-API first for accurate real-time spot price
        current_gold_usd = fetch_gold_from_api()
        source = "Gold-API (XAU/USD)"
        
        # Fallback to yfinance if Gold-API fails
        if current_gold_usd is None:
            current_gold_usd = fetch_gold_from_yfinance()
            source = "yfinance (GC=F)"
        
        if current_gold_usd is None:
            raise HTTPException(status_code=503, detail="Unable to fetch gold prices")
        
        # Fetch USD/EGP exchange rate from yfinance
        egp = yf.Ticker("USDEGP=X")
        egp_data = egp.history(period="5d")
        
        if egp_data.empty:
            raise HTTPException(status_code=503, detail="Unable to fetch exchange rate")
        
        current_usd_egp = float(egp_data['Close'].iloc[-1])
        
        # Get previous gold price from yfinance for change calculation
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(period="5d")
        prev_gold_usd = float(gold_data['Close'].iloc[-2]) if len(gold_data) > 1 else current_gold_usd
        prev_usd_egp = float(egp_data['Close'].iloc[-2]) if len(egp_data) > 1 else current_usd_egp
        
        # Calculate 24K gold price in EGP per gram
        current_24k = (current_gold_usd / 31.1035) * current_usd_egp
        prev_24k = (prev_gold_usd / 31.1035) * prev_usd_egp
        
        # Calculate all karat prices
        prices = calculate_karat_prices(current_24k)
        prev_prices = calculate_karat_prices(prev_24k)
        
        # Calculate Ounce change
        oz_change = current_gold_usd - prev_gold_usd
        oz_change_pct = (oz_change / prev_gold_usd) * 100 if prev_gold_usd else 0
        
        # Get current Egypt time
        now_egypt = datetime.now(EGYPT_TZ)
        
        # Build response
        result = {
            "timestamp": now_egypt.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "gold_usd_oz": {
                "value": round(current_gold_usd, 2),
                "change": round(oz_change, 2),
                "change_pct": round(oz_change_pct, 2),
                "direction": "up" if oz_change > 0 else "down" if oz_change < 0 else "neutral"
            },
            "usd_egp_rate": round(current_usd_egp, 2),
            "prices": {}
        }
        
        for karat in ['karat_24', 'karat_21', 'karat_18', 'karat_14']:
            change = prices[karat] - prev_prices[karat]
            change_pct = (change / prev_prices[karat]) * 100 if prev_prices[karat] else 0
            
            result["prices"][karat] = {
                "value": prices[karat],
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "direction": "up" if change > 0 else "down" if change < 0 else "neutral"
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live prices: {str(e)}")


# ============================================================================
# SILVER API ENDPOINTS
# ============================================================================

def calculate_silver_purity_prices(price_999: float) -> dict:
    """Calculate all silver purity prices from 999 base price."""
    return {
        "purity_999": round(price_999, 2),
        "purity_925": round(price_999 * (925/999), 2),  # Sterling silver
        "purity_900": round(price_999 * (900/999), 2),  # Coin silver
        "purity_800": round(price_999 * (800/999), 2),  # European silver
        "ounce": round(price_999 * 31.1035, 2)
    }


@app.get("/api/silver/prices/current")
async def get_silver_current_prices():
    """
    Get LIVE silver prices for all purities.
    Uses Gold-API for accurate spot prices, yfinance as fallback.
    """
    try:
        # Try Gold-API first for accurate real-time spot price
        current_silver_usd = fetch_silver_from_api()
        source = "Gold-API (XAG/USD)"
        
        # Fallback to yfinance if Gold-API fails
        if current_silver_usd is None:
            current_silver_usd = fetch_silver_from_yfinance()
            source = "yfinance (SI=F)"
        
        if current_silver_usd is None:
            raise HTTPException(status_code=503, detail="Unable to fetch silver prices")
        
        # Fetch USD/EGP exchange rate from yfinance
        egp = yf.Ticker("USDEGP=X")
        egp_data = egp.history(period="5d")
        
        if egp_data.empty:
            raise HTTPException(status_code=503, detail="Unable to fetch exchange rate")
        
        current_usd_egp = float(egp_data['Close'].iloc[-1])
        
        # Get previous silver price from yfinance for change calculation
        silver = yf.Ticker("SI=F")
        silver_data = silver.history(period="5d")
        prev_silver_usd = float(silver_data['Close'].iloc[-2]) if len(silver_data) > 1 else current_silver_usd
        prev_usd_egp = float(egp_data['Close'].iloc[-2]) if len(egp_data) > 1 else current_usd_egp
        
        # Calculate 999 silver price in EGP per gram with Egyptian market premium
        EGYPT_SILVER_PREMIUM = 1.11  # Egyptian market premium (~11%) over global spot
        current_999_spot = (current_silver_usd / 31.1035) * current_usd_egp
        prev_999_spot = (prev_silver_usd / 31.1035) * prev_usd_egp
        current_999 = current_999_spot * EGYPT_SILVER_PREMIUM
        prev_999 = prev_999_spot * EGYPT_SILVER_PREMIUM
        
        # Calculate all purity prices
        prices = calculate_silver_purity_prices(current_999)
        prev_prices = calculate_silver_purity_prices(prev_999)
        
        # Calculate Ounce change
        oz_change = current_silver_usd - prev_silver_usd
        oz_change_pct = (oz_change / prev_silver_usd) * 100 if prev_silver_usd else 0
        
        # Get current Egypt time
        now_egypt = datetime.now(EGYPT_TZ)
        
        # Build response
        result = {
            "timestamp": now_egypt.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "silver_usd_oz": {
                "value": round(current_silver_usd, 2),
                "change": round(oz_change, 2),
                "change_pct": round(oz_change_pct, 2),
                "direction": "up" if oz_change > 0 else "down" if oz_change < 0 else "neutral"
            },
            "usd_egp_rate": round(current_usd_egp, 2),
            "prices": {}
        }
        
        for purity in ['purity_999', 'purity_925', 'purity_900', 'purity_800']:
            change = prices[purity] - prev_prices[purity]
            change_pct = (change / prev_prices[purity]) * 100 if prev_prices[purity] else 0
            
            result["prices"][purity] = {
                "value": prices[purity],
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "direction": "up" if change > 0 else "down" if change < 0 else "neutral"
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live silver prices: {str(e)}")


@app.get("/api/silver/prices/ohlc")
async def get_silver_ohlc_data(
    period: str = Query("1m", description="Period: 1d, 1w, 1m, 3m, 6m, 1y, max"),
    type: str = Query("999", description="Type: 999, 925, 900, 800, oz")
):
    """
    Get historical silver price data for chart.
    Fetches from yfinance (SI=F) and calculates purity prices.
    """
    try:
        period_map = {
            "1d": ("1d", "5m"), "1w": ("5d", "1h"), "1m": ("1mo", "1d"),
            "3m": ("3mo", "1d"), "6m": ("6mo", "1d"), "1y": ("1y", "1wk"),
            "max": ("max", "1mo")
        }
        yf_period, yf_interval = period_map.get(period, ("1mo", "1d"))
        
        # Get silver and EGP data
        silver = yf.Ticker("SI=F")
        egp = yf.Ticker("USDEGP=X")
        
        silver_data = silver.history(period=yf_period, interval=yf_interval)
        egp_data = egp.history(period=yf_period, interval=yf_interval)
        
        if silver_data.empty or egp_data.empty:
            return {"period": period, "data": []}
        
        # Use a fixed exchange rate for historical calculation
        latest_egp = float(egp_data['Close'].iloc[-1])
        
        # Purity multipliers
        purity_map = {
            "999": 1.0, "925": 0.925, "900": 0.900, "800": 0.800, "oz": None
        }
        purity_mult = purity_map.get(type.lower(), 1.0)
        
        data = []
        for idx, row in silver_data.iterrows():
            silver_price = float(row['Close'])
            
            if type.lower() == 'oz':
                # Return USD price directly
                close_val = silver_price
                open_val = float(row['Open'])
                high_val = float(row['High'])
                low_val = float(row['Low'])
            else:
                # Calculate EGP per gram for purity
                purity_999 = (silver_price / 31.1035) * latest_egp
                close_val = purity_999 * purity_mult
                open_val = (float(row['Open']) / 31.1035) * latest_egp * purity_mult
                high_val = (float(row['High']) / 31.1035) * latest_egp * purity_mult
                low_val = (float(row['Low']) / 31.1035) * latest_egp * purity_mult
            
            data.append({
                "time": int(idx.timestamp()),
                "open": round(open_val, 2),
                "high": round(high_val, 2),
                "low": round(low_val, 2),
                "close": round(close_val, 2)
            })
        
        return {
            "period": period, 
            "interval": yf_interval, 
            "data": data, 
            "currency": "USD" if type.lower() == "oz" else "EGP"
        }
        
    except Exception as e:
        return {"period": period, "data": [], "error": str(e)}


@app.get("/api/silver/news")
async def get_silver_news(limit: int = Query(6, description="Number of news items")):
    """Fetch latest silver news from Google News RSS (Arabic)."""
    try:
        # Google News RSS for silver prices in Egypt (Arabic)
        rss_url = "https://news.google.com/rss/search?q=اسعار+الفضة+مصر&hl=ar&gl=EG&ceid=EG:ar"
        
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        
        # Parse RSS feed
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        news_items = []
        for item in root.findall('.//item')[:limit]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            source = item.find('source').text if item.find('source') is not None else "Google News"
            
            # Parse date
            try:
                from email.utils import parsedate_to_datetime
                parsed_date = parsedate_to_datetime(pub_date)
                time_ago = get_time_ago(parsed_date)
            except:
                time_ago = pub_date
            
            news_items.append({
                "title": title,
                "link": link,
                "source": source,
                "time_ago": time_ago
            })
        
        return {"news": news_items}
        
    except Exception as e:
        return {"news": [], "error": str(e)}

@app.get("/api/prices/ohlc")
async def get_ohlc_data(
    period: str = Query("1m", description="Period: 1d, 1w, 1m, 3m, 6m, 1y, max"),
    type: str = Query("24k", description="Type: 24k, 21k, 18k, 14k, oz")
):
    """
    Get historical price data for chart.
    Supports different karats and USD Ounce.
    """
    try:
        conn = get_db_connection()
        
        # Map type to database column
        column_map = {
            "24k": "karat_24",
            "21k": "karat_21",
            "18k": "karat_18",
            "14k": "karat_14",
            "oz": "gold_usd_oz"
        }
        db_col = column_map.get(type.lower(), "karat_24")
        
        # Calculate date range
        period_days = {
            "1d": 1, "1w": 7, "1m": 30, "3m": 90, 
            "6m": 180, "1y": 365, "max": 3650
        }
        days = period_days.get(period, 30)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        df = pd.read_sql_query(
            f"SELECT timestamp, {db_col} as close FROM prices WHERE timestamp >= '{cutoff}' ORDER BY timestamp ASC",
            conn
        )
        conn.close()
        
        if df.empty:
            return {"period": period, "interval": "1D", "data": []}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample based on period
        resample_map = {
            "1d": "5min", "1w": "1h", "1m": "4h",
            "3m": "1D", "6m": "1D", "1y": "1W", "max": "1W"
        }
        resample_freq = resample_map.get(period, "1D")
        
        df.set_index('timestamp', inplace=True)
        ohlc = df['close'].resample(resample_freq).agg(['first', 'max', 'min', 'last']).dropna()
        ohlc.columns = ['open', 'high', 'low', 'close']
        
        data = []
        for timestamp, row in ohlc.iterrows():
            data.append({
                "time": int(timestamp.timestamp()),
                "open": round(row['open'], 2),
                "high": round(row['high'], 2),
                "low": round(row['low'], 2),
                "close": round(row['close'], 2)
            })
            
        # LIVE SYNC: Append the very latest live price
        try:
            live_price = await get_current_prices()
            current_time = int(datetime.now(EGYPT_TZ).timestamp())
            
            # Get correct value based on type
            if type.lower() == 'oz':
                current_val = live_price['gold_usd_oz']['value']
            else:
                karat_key = f"karat_{type.replace('k', '')}"
                current_val = live_price['prices'].get(karat_key, {}).get('value', 0)
            
            if data and current_val > 0:
                last_candle = data[-1]
                last_time = last_candle['time']
                
                if current_time - last_time > 300: 
                     data.append({
                        "time": current_time,
                        "open": current_val,
                        "high": current_val,
                        "low": current_val,
                        "close": current_val
                    })
                else:
                    last_candle['close'] = current_val
                    last_candle['high'] = max(last_candle['high'], current_val)
                    last_candle['low'] = min(last_candle['low'], current_val)
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not sync live price to chart: {e}")
        
        return {"period": period, "interval": resample_freq, "data": data, "currency": "USD" if type == "oz" else "EGP"}
        
    except Exception as e:
        # Fallback to yfinance
        return await get_ohlc_from_yfinance(period)


async def get_ohlc_from_yfinance(period: str):
    """Fallback: Get OHLC data directly from yfinance."""
    try:
        period_map = {
            "1d": ("1d", "5m"), "1w": ("5d", "1h"), "1m": ("1mo", "1d"),
            "3m": ("3mo", "1d"), "6m": ("6mo", "1d"), "1y": ("1y", "1wk"),
            "max": ("max", "1mo")
        }
        yf_period, yf_interval = period_map.get(period, ("1mo", "1d"))
        
        # Get gold and EGP data
        gold = yf.Ticker("GC=F")
        egp = yf.Ticker("USDEGP=X")
        
        gold_data = gold.history(period=yf_period, interval=yf_interval)
        egp_data = egp.history(period=yf_period, interval=yf_interval)
        
        if gold_data.empty or egp_data.empty:
            return {"period": period, "data": []}
        
        # Use a fixed exchange rate for historical calculation
        latest_egp = float(egp_data['Close'].iloc[-1])
        
        data = []
        for idx, row in gold_data.iterrows():
            gold_price = float(row['Close'])
            karat_24 = (gold_price / 31.1035) * latest_egp
            
            data.append({
                "time": int(idx.timestamp()),
                "open": round((float(row['Open']) / 31.1035) * latest_egp, 2),
                "high": round((float(row['High']) / 31.1035) * latest_egp, 2),
                "low": round((float(row['Low']) / 31.1035) * latest_egp, 2),
                "close": round(karat_24, 2)
            })
        
        return {"period": period, "interval": yf_interval, "data": data}
        
    except Exception as e:
        return {"period": period, "data": [], "error": str(e)}


@app.get("/api/prices/history")
async def get_price_history(limit: int = Query(100, description="Number of records")):
    """Get recent price history from database."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(
            f"SELECT * FROM prices ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        conn.close()
        
        return {
            "count": len(df),
            "data": df.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collector/status")
async def get_collector_status():
    """Get the status of the data collector."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get last collection time
        cursor.execute('SELECT last_collection, status FROM collector_state WHERE id = 1')
        state_row = cursor.fetchone()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) as count FROM prices')
        count_row = cursor.fetchone()
        
        # Get date range
        cursor.execute('SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM prices')
        range_row = cursor.fetchone()
        
        conn.close()
        
        return {
            "status": state_row['status'] if state_row else "not started",
            "last_collection": state_row['last_collection'] if state_row else None,
            "total_records": count_row['count'] if count_row else 0,
            "first_record": range_row['first'] if range_row else None,
            "last_record": range_row['last'] if range_row else None,
            "collection_interval_seconds": COLLECTION_INTERVAL
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/news")
async def get_news(limit: int = Query(6, description="Number of news items")):
    """Fetch latest gold news from Google News RSS (Arabic)."""
    try:
        # Google News RSS for gold prices in Egypt (Arabic)
        rss_url = "https://news.google.com/rss/search?q=اسعار+الذهب+مصر&hl=ar&gl=EG&ceid=EG:ar"
        
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        
        # Parse RSS feed
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        news_items = []
        for item in root.findall('.//item')[:limit]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            source = item.find('source').text if item.find('source') is not None else "Google News"
            
            # Parse date
            try:
                from email.utils import parsedate_to_datetime
                parsed_date = parsedate_to_datetime(pub_date)
                time_ago = get_time_ago(parsed_date)
            except:
                time_ago = pub_date
            
            news_items.append({
                "title": title,
                "link": link,
                "source": source,
                "time_ago": time_ago
            })
        
        return {"news": news_items}
        
    except Exception as e:
        return {"news": [], "error": str(e)}


def get_time_ago(dt):
    """Convert datetime to 'X hours ago' format."""
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    hours = diff.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    minutes = diff.seconds // 60
    return f"{minutes} minute{'s' if minutes > 1 else ''} ago"


# ============================================================================
# STATIC FILES AND FRONTEND
# ============================================================================

# Mount static files
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")


@app.get("/")
async def serve_frontend():
    """Serve the main dashboard."""
    return FileResponse("frontend/index.html")


@app.get("/silver")
async def serve_silver_frontend():
    """Serve the silver dashboard."""
    return FileResponse("frontend/silver.html")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
