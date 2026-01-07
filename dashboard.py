import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
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
                # Parse date for sorting
                try:
                    dt_obj = pd.to_datetime(pub_date)
                except:
                    dt_obj = datetime.min

                news_list.append({
                    'title': title, 
                    'link': link, 
                    'date': pub_date,
                    'dt': dt_obj
                })
            
            if len(news_list) >= 15: break
                
        # Sort by date descending (Latest first)
        news_list.sort(key=lambda x: x['dt'], reverse=True)
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
    # --- CUSTOM CSS (Adaptive Polish) ---
    st.markdown("""
    <style>
        /* Card Style for Metrics - Adaptive to Theme */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(250, 250, 250, 0.1);
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
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
    # --- PLOTTING (ALTAIR) ---
    import altair as alt

    # Prepare Data for Altair (Long Format)
    chart_karats = selected_karats if selected_karats else ['karat_24', 'karat_21', 'karat_18']
    
    # Filter and Melt
    df_plot = df_filtered[['timestamp'] + chart_karats].melt('timestamp', var_name='Karat', value_name='Price')
    
    # Define Default Zoom (Last 30 Days)
    last_date = df['timestamp'].max()
    start_zoom = last_date - timedelta(days=30)
    
    # Add buffer to the right so the Text Label isn't cut off
    domain_end = last_date + timedelta(days=4)
    
    # Configure X-Axis Scale with Default Domain
    x_scale = alt.Scale(domain=(start_zoom, domain_end))
    
    # Base Chart
    base = alt.Chart(df_plot).encode(
        x=alt.X('timestamp:T', title=None, axis=alt.Axis(format='%d %b'), scale=x_scale),
        y=alt.Y('Price:Q', title='EGP', scale=alt.Scale(zero=False)),
        color=alt.Color('Karat:N', legend=alt.Legend(orient='bottom')),
        tooltip=['timestamp', 'Karat', alt.Tooltip('Price', format=',.0f')]
    )
    
    # Simple Line + Points (Similar to AI Chart style)
    lines = base.mark_line(point=True).encode(
        color=alt.Color('Karat:N')
    )
    
    # Text Labels at the End (Latest Price)
    last_points = df_plot.sort_values('timestamp').groupby('Karat').tail(1)
    
    text_labels = alt.Chart(last_points).mark_text(
        align='left', dx=5, dy=-5, fontSize=12, fontWeight='bold'
    ).encode(
        x='timestamp:T',
        y='Price:Q',
        text=alt.Text('Price:Q', format=',.0f'),
        color='Karat:N'
    )
    
    # Combined (No Area layer)
    final_chart = lines + text_labels
    
    text_labels = alt.Chart(last_points).mark_text(
        align='left', dx=5, dy=-5, fontSize=12, fontWeight='bold'
    ).encode(
        x='timestamp:T',
        y='Price:Q',
        text=alt.Text('Price:Q', format=',.0f'),
        color='Karat:N'
    )
    
    final_chart = areas + lines + text_labels

    # --- TECHNICAL OVERLAYS (ALTAIR) ---
    ohlc_reset = ohlc.reset_index()
    
    tech_layers = []
    
    if show_sma:
        sma_line = alt.Chart(ohlc_reset).mark_line(color='orange', strokeDash=[5,5]).encode(
            x='timestamp:T',
            y='SMA20:Q',
            tooltip=['timestamp', alt.Tooltip('SMA20', format=',.0f', title=f'SMA 20 ({target_asset})')]
        )
        tech_layers.append(sma_line)

    if show_ema:
        ema_line = alt.Chart(ohlc_reset).mark_line(color='cyan', strokeDash=[2,2]).encode(
            x='timestamp:T',
            y='EMA50:Q',
            tooltip=['timestamp', alt.Tooltip('EMA50', format=',.0f', title=f'EMA 50 ({target_asset})')]
        )
        tech_layers.append(ema_line)

    if show_bb:
        # Band (Area)
        bb_band = alt.Chart(ohlc_reset).mark_area(opacity=0.1, color='gray').encode(
            x='timestamp:T',
            y='BB_Lower:Q',
            y2='BB_Upper:Q'
        )
        # Edges
        bb_upper = alt.Chart(ohlc_reset).mark_line(color='gray', size=0.5).encode(x='timestamp:T', y='BB_Upper:Q')
        bb_lower = alt.Chart(ohlc_reset).mark_line(color='gray', size=0.5).encode(x='timestamp:T', y='BB_Lower:Q')
        
        tech_layers.extend([bb_band, bb_upper, bb_lower])

    # Combine all layers
    for layer in tech_layers:
        final_chart += layer

    # Display Main Chart with Horizontal Interaction ONLY
    # bind_y=False ensures vertical swiping scrolls the page, NOT the chart
    st.altair_chart(final_chart.interactive(bind_y=False), use_container_width=True)

    # --- RSI SUBPLOT (ALTAIR) ---
    if show_rsi:
        # RSI Chart
        rsi_base = alt.Chart(ohlc_reset).encode(x='timestamp:T')
        
        rsi_line = rsi_base.mark_line(color='purple').encode(
            y=alt.Y('RSI:Q', scale=alt.Scale(domain=[0, 100]), title='RSI')
        )
        
        # Guidelines (70/30)
        rule_70 = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(color='red', strokeDash=[2,2]).encode(y='y')
        rule_30 = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='green', strokeDash=[2,2]).encode(y='y')
        
        rsi_chart = (rsi_line + rule_70 + rule_30).properties(height=150)
        
        st.altair_chart(rsi_chart.interactive(), use_container_width=True)


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
        future_prices = predictions.flatten()
        forecast_df = pd.DataFrame({
            'timestamp': future_dates,
            'Price': future_prices,
            'Type': 'Forecast'
        })
        
        # Historical Data (Last 30 days) for context
        hist_df = df.tail(30).copy()
        hist_df['Price'] = hist_df[target_karat]
        hist_df['Type'] = 'Historical'
        
        # Combine for Plotting
        plot_df = pd.concat([hist_df[['timestamp', 'Price', 'Type']], forecast_df])

        # --- PREDICTION VISUALIZATION (ALTAIR) ---
        pred_chart = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title='Date', axis=alt.Axis(format='%d %b')),
            y=alt.Y('Price:Q', title='Price (EGP)', scale=alt.Scale(zero=False)),
            color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['#1f77b4', '#ff7f0e'])),
            tooltip=['timestamp', alt.Tooltip('Price', format=',.2f'), 'Type']
        ).properties(
            title="Gold Price Forecast (Next 7 Days)"
        )
        
        st.altair_chart(pred_chart.interactive(), use_container_width=True)
        
        st.write("Predicted Prices:")
        st.dataframe(forecast_df[['timestamp', 'Price']])



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
