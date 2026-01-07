import streamlit as st
import pandas as pd
import sqlite3
import streamlit as st
import pandas as pd
import sqlite3
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, LabelSet, HoverTool, DatetimeTickFormatter, Label, NumeralTickFormatter
from bokeh.palettes import Category10
from streamlit_bokeh import streamlit_bokeh as st_bokeh_chart
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
    st.subheader(f"📈 Price History")
    
    st.subheader(f"📈 Price History")
    
    # Defaults
    chart_karats = selected_karats if selected_karats else ['karat_24', 'karat_21', 'karat_18']
    
    if not df_filtered.empty:
        # Prepare Bokeh Figure
        p = figure(x_axis_type="datetime", title="", height=300, toolbar_location="right", tools="pan,wheel_zoom,reset,save")
        p.grid.grid_line_alpha = 0.3
        p.background_fill_color = "#0E1117" # Match Dark Theme
        p.border_fill_color = "#0E1117"
        p.xaxis.axis_label = "Date"
        p.yaxis.formatter = NumeralTickFormatter(format="0,0")
        
        # Colors
        colors = Category10[10]
        
        # Labels Data
        labels_data = []

        for i, karat in enumerate(chart_karats):
            # Data for this karat
            k_df = df_filtered[['timestamp', karat]].dropna().sort_values('timestamp')
            
            if not k_df.empty:
                # Add Line
                source = ColumnDataSource(k_df)
                p.line(x='timestamp', y=karat, source=source, line_width=2, color=colors[i % 10], legend_label=karat)
                p.circle(x='timestamp', y=karat, source=source, size=4, color=colors[i % 10])
                
                # Store last point for label
                last_pt = k_df.iloc[-1]
                labels_data.append({
                    'timestamp': last_pt['timestamp'],
                    'price': last_pt[karat],
                    'text': f"{last_pt[karat]:,.0f}",
                    'color': colors[i % 10]
                })

        # Add Labels
        if labels_data:
            label_df = pd.DataFrame(labels_data)
            label_source = ColumnDataSource(label_df)
            labels = LabelSet(x='timestamp', y='price', text='text', level='glyph',
                              x_offset=8, y_offset=-8, source=label_source, 
                              text_color='color', text_font_style='bold')
            p.add_layout(labels)
            
            # Hover Tool
            hover = HoverTool(tooltips=[("Date", "@timestamp{%F}"), ("Price", "@y{0,0} EGP")],
                              formatters={'@timestamp': 'datetime'})
            p.add_tools(hover)

            # CENTER THE VIEW: Extension logic
            min_ts = df_filtered['timestamp'].min()
            max_ts = df_filtered['timestamp'].max()
            duration = max_ts - min_ts
            
            # If we have duration, add it to the right
            if duration.total_seconds() > 0:
                p.x_range = Range1d(start=min_ts, end=max_ts + (duration / 2))
            else:
                p.x_range = Range1d(start=max_ts - timedelta(days=4), end=max_ts + timedelta(days=4))

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.0
        p.legend.label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.title.text_color = "white"
        
        st_bokeh_chart(p, use_container_width=True)
    else:
        st.info("No data available for the selected range.")


    # --- AI ANALYST (SHARED) ---
    # Need to calculate technicals for AI logic (since we removed the chart-based calc)
    if not df_filtered.empty:
        # Defaults for Analysis
        target_asset = selected_karats[0] if selected_karats else 'karat_24'
        df_asset = df_filtered[['timestamp', target_asset]].copy()
        df_asset.set_index('timestamp', inplace=True)
        
        # Resample
        ohlc = df_asset[target_asset].resample('1D').agg(['first', 'max', 'min', 'last'])
        ohlc.columns = ['Open', 'High', 'Low', 'Close']
        ohlc.dropna(inplace=True)
        
        # Calculate Indicators
        ohlc['SMA20'] = ohlc['Close'].rolling(window=20).mean()
        ohlc['BB_Middle'] = ohlc['Close'].rolling(window=20).mean()
        ohlc['BB_Std'] = ohlc['Close'].rolling(window=20).std()
        ohlc['BB_Upper'] = ohlc['BB_Middle'] + (2 * ohlc['BB_Std'])
        ohlc['BB_Lower'] = ohlc['BB_Middle'] - (2 * ohlc['BB_Std'])
        
        delta = ohlc['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        ohlc['RSI'] = 100 - (100 / (1 + rs))

        if not ohlc.empty and len(ohlc) > 20:
            last_close = ohlc['Close'].iloc[-1]
            last_rsi = ohlc['RSI'].iloc[-1]
            # ... rest of logic uses already defined vars ...
            
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
        
        # Create Forecast DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': future_dates,
            'Price': predictions
        })
        
        # --- PREDICTION VISUALIZATION (BOKEH) ---
        p_pred = figure(x_axis_type="datetime", title="Gold Price Forecast (Next 7 Days)", 
                        height=300, toolbar_location="right", tools="pan,wheel_zoom,reset,save")
        p_pred.grid.grid_line_alpha = 0.1 # Fainter grid
        p_pred.background_fill_color = "#0E1117"
        p_pred.yaxis.formatter = NumeralTickFormatter(format="0,0")
        
        # 1. Historical
        hist_df = df.tail(30)
        source_hist = ColumnDataSource(hist_df)
        p_pred.line(x='timestamp', y=target_karat, source=source_hist, line_width=2, color='#1f77b4', legend_label="Historical")
        p_pred.circle(x='timestamp', y=target_karat, source=source_hist, size=4, color='#1f77b4')
        
        # 2. Forecast
        source_pred = ColumnDataSource(forecast_df)
        p_pred.line(x='timestamp', y='Price', source=source_pred, line_width=2, color='#ff7f0e', line_dash="dashed", legend_label="Forecast")
        p_pred.circle(x='timestamp', y='Price', source=source_pred, size=4, color='#ff7f0e')
        
        # Label for final prediction
        if not forecast_df.empty:
            last_pred = forecast_df.iloc[-1]
            label_pred = Label(x=last_pred['timestamp'], y=last_pred['Price'], 
                               text=f"{last_pred['Price']:,.0f}", text_color='#ff7f0e', 
                               x_offset=8, y_offset=-8, text_font_style='bold')
            p_pred.add_layout(label_pred)

        # Styling
        p_pred.legend.location = "top_left"
        p_pred.legend.background_fill_alpha = 0.0
        p_pred.legend.label_text_color = "white"
        p_pred.xaxis.major_label_text_color = "white"
        p_pred.yaxis.major_label_text_color = "white"
        p_pred.title.text_color = "white"
        p_pred.min_border_left = 0
        p_pred.min_border_right = 0
        
        # Hover
        hover_pred = HoverTool(tooltips=[("Date", "@timestamp{%F}"), ("Price", "@y{0,0}")],
                               formatters={'@timestamp': 'datetime'})
        p_pred.add_tools(hover_pred)

        st_bokeh_chart(p_pred, use_container_width=True)
        
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
