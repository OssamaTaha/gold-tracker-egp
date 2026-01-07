import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

import re
import html

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_gold_news(start_date=None, end_date=None):
    """Fetches news using Google News RSS (Arabic) with date filtering."""
    # Base Arabic Query: "أسعار الذهب"
    query = "%D8%A3%D8%B3%D8%B9%D8%A7%D8%B1+%D8%A7%D9%84%D8%B0%D9%87%D8%A8"
    
    # Append date filters if provided
    if start_date and end_date:
        # Google Search operators: after:YYYY-MM-DD before:YYYY-MM-DD
        # We assume dates are datetime.date objects
        query += f"+after:{start_date}+before:{end_date}"
        
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=ar-EG&gl=EG&ceid=EG:ar"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(rss_url, headers=headers, timeout=5)
        content = response.text
        
        # Regex to find items
        items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
        news_list = []
        
        for item_xml in items:
            # Helper to extract tag content
            def get_tag(tag, text):
                match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
                return match.group(1).strip() if match else ""

            title = get_tag('title', item_xml)
            link = get_tag('link', item_xml)
            pub_date = get_tag('pubDate', item_xml)
            
            if title and link:
                news_list.append({
                    'title': title, 
                    'link': link, 
                    'date': pub_date
                })
            
            if len(news_list) >= 15: break
                
        return news_list
    except Exception as e:
        print(f"News Fetch Error: {e}")
        return []

st.set_page_config(page_title="GoldTracker EGP", layout="wide")

st.title("🏆 Egyptian Gold Price Tracker & AI Predictor")

def load_data():
    try:
        conn = sqlite3.connect("gold_prices.db")
        df = pd.read_sql_query("SELECT * FROM prices ORDER BY timestamp ASC", conn)
        conn.close()
        
        # Convert timestamp to datetime (Fix for AttributeError)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # Calculate derived units
        df['karat_14'] = df['karat_24'] * (14/24)
        df['ounce'] = df['karat_24'] * 31.1035
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Auto-refresh logic
if st.button('Refresh Data'):
    st.rerun()

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filters")

if not df.empty:
    last_ts = df['timestamp'].max()
    st.sidebar.caption(f"Last Updated: {last_ts.strftime('%Y-%m-%d %H:%M')}")

# 1. Date Filter
if not df.empty:
    # --- CUSTOM CSS (Mobile Polish) ---
    st.markdown("""
    <style>
        /* Card Style for Metrics */
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
        /* Mobile: Hide Chart Modebar & Compact Headers */
        .modebar-container { display: none !important; }
        h1, h2, h3 { padding-top: 5px !important; }
    </style>
    """, unsafe_allow_html=True)

    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[],
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = []

# 2. Karat Filter
available_karats = ['karat_24', 'karat_21', 'karat_18', 'karat_14', 'ounce']
selected_karats = st.sidebar.multiselect(
    "Select Karats/Units", 
    available_karats, 
    default=[]
)

# Apply Filters
if not df.empty and len(date_range) == 2:
    start_date, end_date = date_range
    # Convert to datetime for comparison
    cutoff_start = pd.Timestamp(start_date)
    cutoff_end = pd.Timestamp(end_date) + timedelta(days=1) - timedelta(seconds=1) # End of day
    
    df_filtered = df[(df['timestamp'] >= cutoff_start) & (df['timestamp'] <= cutoff_end)]
else:
    df_filtered = df

if not df.empty:
    # Most recent price (Global)
    latest = df.iloc[-1]
    
    # Dynamic Metric Display
    # If no filter selected, show ALL (User requirement)
    if not selected_karats:
        selected_karats = available_karats
        
    # Dynamic Metric Display (Responsive Grid)
    # If no filter selected, show ALL (User requirement)
    if not selected_karats:
        selected_karats = available_karats
    
    # --- SMART DELTA LOGIC ---
    # Fix: If NO filter is selected (Default View), compare Today vs YESTERDAY.
    # If Filter IS selected, compare End vs Start of that period.
    
    start_of_period = None
    is_custom_filter = True if (not df.empty and len(date_range) > 0) else False # date_range is [] by default now

    if is_custom_filter and len(df_filtered) > 1:
         # Compare End of Range vs Start of Range
         start_of_period = df_filtered.iloc[0]
    elif not is_custom_filter and len(df) >= 2:
        # Default View: Show DAILY Change (Today vs Yesterday)
        # We use iloc[-2] because iloc[-1] is "Latest"
        start_of_period = df.iloc[-2]
    else:
        start_of_period = None
        
    # Grid System: Max 3 columns per row to avoid squeezing
    MAX_COLS = 3
    for i in range(0, len(selected_karats), MAX_COLS):
        row_chunk = selected_karats[i:i+MAX_COLS]
        cols = st.columns(len(row_chunk))
        
        for idx, karat in enumerate(row_chunk):
            label = karat.replace('_', ' ').title()
            if karat == 'ounce': label = "Ounce (31.1g)"
            
            current_val = latest[karat]
            
            if start_of_period is not None:
                delta_val = current_val - start_of_period[karat]
                pct_val = (delta_val / start_of_period[karat]) * 100
                delta_str = f"{delta_val:,.2f} EGP ({pct_val:.2f}%)"
                
                # Context Label
                if is_custom_filter:
                    label += " (Period)"
                else:
                    label += " (Daily)"
            else:
                delta_str = "0.00 EGP (0.00%)"
                
            with cols[idx]:
                st.metric(label, f"{current_val:,.2f} EGP", delta_str)

    st.markdown("---")
    
    # 1. Historical Trend
    st.markdown("---")
    
    # 1. Advanced Trading Chart (Line Only)
    st.subheader(f"📈 Advanced Market Chart")
    
    # Defaults for Analysis
    freq = "1d"
    target_asset = selected_karats[0] if selected_karats else 'karat_24'

    # --- SHARED TECHNICAL ANALYSIS ---
    # Resample Data to OHLC (Needed for Indicators)
    df_asset = df_filtered[['timestamp', target_asset]].copy()
    df_asset.set_index('timestamp', inplace=True)
    
    # Resample
    ohlc = df_asset[target_asset].resample(freq).agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['Open', 'High', 'Low', 'Close']
    ohlc.dropna(inplace=True)
    
    # --- TECHNICAL INDICATORS CALCULATIONS ---
    # 1. SMA 20
    ohlc['SMA20'] = ohlc['Close'].rolling(window=20).mean()
    # 2. EMA 50
    ohlc['EMA50'] = ohlc['Close'].ewm(span=50, adjust=False).mean()
    # 3. Bollinger Bands
    ohlc['BB_Middle'] = ohlc['Close'].rolling(window=20).mean()
    ohlc['BB_Std'] = ohlc['Close'].rolling(window=20).std()
    ohlc['BB_Upper'] = ohlc['BB_Middle'] + (2 * ohlc['BB_Std'])
    ohlc['BB_Lower'] = ohlc['BB_Middle'] - (2 * ohlc['BB_Std'])
    # 4. RSI
    delta = ohlc['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    ohlc['RSI'] = 100 - (100 / (1 + rs))

    # --- INDICATOR CONTROLS ---
    st.write("### 🛠️ Technical Indicators")
    c_ind1, c_ind2, c_ind3, c_ind4 = st.columns(4)
    
    show_sma = c_ind1.checkbox("SMA 20", value=True, help="Avg Price (20d). Support level in uptrends.")
    show_ema = c_ind2.checkbox("EMA 50", value=False, help="Avg Price (50d). Medium-term trend.")
    show_bb = c_ind3.checkbox("Bollinger Bands", value=False, help="Volatility Bands. Buy at Lower, Sell at Upper.")
    show_rsi = c_ind4.checkbox("Show RSI Panel", value=True, help="Overbought (>70) / Oversold (<30) Gauge.")

    # --- PLOTTING ---
    from plotly.subplots import make_subplots
    
    rows = 2 if show_rsi else 1
    row_heights = [0.7, 0.3] if show_rsi else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=row_heights,
                        subplot_titles=("Price Action", "RSI (14)" if show_rsi else None))
    
    # Draw all selected assets as lines
    if selected_karats:
        for k in selected_karats:
            asset_data = df_filtered[['timestamp', k]].dropna()
            
            # Main Line
            fig.add_trace(go.Scatter(
                x=asset_data['timestamp'], y=asset_data[k],
                mode='lines', name=k.replace('_',' ').title()
            ), row=1, col=1)
            
            # Latest Price Label (Text at the end)
            if not asset_data.empty:
                last_pt = asset_data.iloc[-1]
                fig.add_trace(go.Scatter(
                    x=[last_pt['timestamp']], 
                    y=[last_pt[k]],
                    mode='markers+text',
                    text=[f"{last_pt[k]:,.0f}"],
                    textposition="middle right",
                    marker=dict(size=8),
                    showlegend=False,
                    name=f"Current {k}"
                ), row=1, col=1)

    else:
        st.warning("Select Karats to view.")

    # --- OVERLAYS (SHARED) ---
    if show_sma:
        fig.add_trace(go.Scatter(x=ohlc.index, y=ohlc['SMA20'], mode='lines', name=f'SMA 20 ({target_asset})', line=dict(color='orange', width=1)), row=1, col=1)
    if show_ema:
        fig.add_trace(go.Scatter(x=ohlc.index, y=ohlc['EMA50'], mode='lines', name=f'EMA 50 ({target_asset})', line=dict(color='cyan', width=1)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=ohlc.index, y=ohlc['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc.index, y=ohlc['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'), row=1, col=1)

    # --- RSI SUBPLOT ---
    if show_rsi:
        fig.add_trace(go.Scatter(x=ohlc.index, y=ohlc['RSI'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # Layout
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        height=700 if show_rsi else 500,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            type="date"
        )
    )
    
    # Default Zoom to last 30 days
    # User requested FULL history by default
    # if not df.empty:
    #     last_date = df['timestamp'].max()
    #     start_zoom = last_date - timedelta(days=30)
    #     fig.update_xaxes(range=[start_zoom, last_date])
    
    # MOBILE CONFIG: Disable Scroll Zoom to prevent page blocking
    st.plotly_chart(fig, width="stretch", config={'scrollZoom': False, 'displayModeBar': False, 'staticPlot': False})

    # --- AI ANALYST (SHARED) ---
    if not ohlc.empty and len(ohlc) > 20:
            last_close = ohlc['Close'].iloc[-1]
            last_rsi = ohlc['RSI'].iloc[-1]
            
            # BB Values
            bb_upper = ohlc['BB_Upper'].iloc[-1]
            bb_lower = ohlc['BB_Lower'].iloc[-1]
            sma_20 = ohlc['SMA20'].iloc[-1]

            # Signal Logic
            signal = "HOLD"
            color = "blue"
            reason = "Market is ranging."
            
            # Buy Conditions
            if last_rsi < 35:
                signal = "STRONG BUY"
                color = "green"
                reason = "RSI indicates Oversold conditions."
            elif last_close <= sma_20 * 1.005: 
                signal = "BUY"
                color = "green" 
                reason = "Price near Fair Value (SMA 20)."
                
            # Sell Conditions 
            elif last_rsi > 65:
                signal = "STRONG SELL"
                color = "red"
                reason = "RSI indicates Overbought conditions."
            elif last_close >= bb_upper * 0.99: 
                signal = "SELL"
                color = "red"
                reason = "Price touched Resistance (Upper Band)."

            # Targets
            buy_target = sma_20 
            sell_target = bb_upper

            st.markdown("### 🧠 AI Trading Signal")
            
            sig_col1, sig_col2, sig_col3 = st.columns(3)
            sig_col1.metric("Recommendation", signal, reason, delta_color="off" if color=="blue" else "normal")
            
            diff_buy = last_close - buy_target
            buy_delta = f"{diff_buy:,.0f} EGP premium" if diff_buy > 0 else "Below Fair Value!"
            sig_col2.metric("📉 Smart Entry Price", f"{buy_target:,.0f} EGP", "Fair Value (SMA 20)")
            
            diff_sell = sell_target - last_close
            sell_delta = f"{diff_sell:,.0f} EGP potential" if diff_sell > 0 else "At Resistance!"
            sig_col3.metric("📈 Take Profit Price", f"{sell_target:,.0f} EGP", "Resistance Level (Upper Band)")
            
    else:
        st.info("Insufficient data for technical analysis (Need > 20 periods).")

    # 2. AI Prediction Section
    st.subheader("🤖 AI Price Prediction (Beta)")
    
    if len(df) < 5:
        st.info("Not enough data to train the AI model. Please run the scraper more times to build history.")
    else:
        # Prepare data for ML
        # Logic Update: Train only on RECENT history (e.g. last 6 months) 
        # to capture current market momentum and avoid skewing with 10-year old data.
        
        subset_size = 180 # Approx 6 months
        if len(df) > subset_size:
            df_train = df.tail(subset_size).copy()
            st.caption(f"ℹ️ Model trained on the last {subset_size} data points (Recent Trend) to ensure accuracy.")
        else:
            df_train = df.copy()
            st.caption("ℹ️ Model trained on available history.")

        target_karat = 'karat_21'
        df_train['timestamp_ordinal'] = df_train['timestamp'].map(datetime.toordinal)
        
        X = df_train[['timestamp_ordinal']]
        y = df_train[target_karat]
        
        # Simple Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 7 days
        last_date = df['timestamp'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        
        predictions = model.predict(future_ordinal)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': future_dates,
            'Predicted Price (21k)': predictions
        })
        
        # Visualize Prediction
        fig_pred = go.Figure()
        
        # Actual History
        fig_pred.add_trace(go.Scatter(x=df['timestamp'], y=df[target_karat], mode='lines+markers', name='Actual History'))
        
        # Forecast
        fig_pred.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['Predicted Price (21k)'], 
                                     mode='lines+markers', name='AI Forecast', line=dict(dash='dash', color='green')))
        
        fig_pred.update_layout(title="Gold Price Forecast (Next 5 Days)", xaxis_title="Date", yaxis_title="Price (EGP)")
        st.plotly_chart(fig_pred, width="stretch")
        
        st.write("Predicted Prices:")
        st.dataframe(forecast_df)



    st.markdown("---")
    
    # 3. News Section
    st.subheader("📰 Market News (Google News - Egypt/Arabic)")
    
    # Pass date filter if available
    if not df.empty and len(date_range) == 2:
        # Extend end_date by 1 day to cover the full range in search
        news_items = fetch_gold_news(date_range[0], date_range[1] + timedelta(days=1))
        st.caption(f"Showing news from {date_range[0]} to {date_range[1]}")
    else:
        news_items = fetch_gold_news()
    
    if news_items:
        for news in news_items:
            st.markdown(f"**[{news['title']}]({news['link']})**")
            st.caption(f"📅 {news['date']}")
            st.markdown("---")
    else:
        st.info("No recent specific gold/economy news found in the feed.")

    with st.expander("Week Data Table (Last 7 Days)"):
        # Explicitly show last 7 days of raw data regardless of filter selection (as requested)
        last_week_date = df['timestamp'].max() - timedelta(days=7)
        week_df = df[df['timestamp'] >= last_week_date]
        st.dataframe(week_df.sort_values(by='timestamp', ascending=False))

else:
    st.warning("No data found! Run the scraper script (etl.py) first.")
