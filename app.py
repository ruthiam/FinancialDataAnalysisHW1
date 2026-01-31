import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import sys
from datetime import datetime, timedelta

# --- CONSTANTS ---
TICKERS = {
    "SP500": "SPY",
    "WTI": "CL=F",
    "GOLD": "GC=F"
}
DATA_FILE = "data/market_data.csv"

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(tickers, start_date, end_date):
    """Downloads adjusted close prices for given tickers."""
    ticker_list = list(tickers.values())
    
    # Download all at once (more efficient and easier to align)
    # Use auto_adjust=False to ensure 'Adj Close' is available if possible
    df = yf.download(ticker_list, start=start_date, end=end_date, auto_adjust=False)
    
    if df.empty:
        return pd.DataFrame()
        
    # Handle MultiIndex columns (common in newer yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            price_df = df['Adj Close']
        else:
            price_df = df['Close']
    else:
        # Single ticker might not have MultiIndex depending on version/args
        if 'Adj Close' in df.columns:
            price_df = df[['Adj Close']]
        elif 'Close' in df.columns:
            price_df = df[['Close']]
        else:
            return pd.DataFrame()

    # Map ticker back to names (SP500, WTI, GOLD)
    reverse_tickers = {v: k for k, v in tickers.items()}
    price_df = price_df.rename(columns=reverse_tickers)
    
    # If it's a single ticker result and price_df became a Series, convert back
    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame()
        
    price_df = price_df.ffill().dropna()
    return price_df

def compute_returns(price_df):
    """Computes daily log returns."""
    # Ensure no zero or negative prices to avoid log errors
    price_df = price_df.clip(lower=1e-8)
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return log_returns

def save_csv(price_df, log_returns):
    """Saves data in tidy/long format to CSV."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    # Melt price data
    price_long = price_df.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Price')
    
    # Melt returns data
    returns_long = log_returns.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Log_Return')
    
    # Merge
    tidy_df = pd.merge(price_long, returns_long, on=['Date', 'Asset'], how='left')
    tidy_df.to_csv(DATA_FILE, index=False)
    return tidy_df

def detect_outliers(series):
    """Detects outliers using z-score > 3 and 1%/99% percentiles."""
    z_scores = np.abs(stats.zscore(series.dropna()))
    z_outliers = z_scores > 3
    
    p1 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    p_outliers = (series < p1) | (series > p99)
    
    return z_outliers | p_outliers

def process_uploaded_file(uploaded_file):
    """Reads and parses the uploaded CSV in tidy format."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Basic validation: check for required columns
        required_cols = ['Date', 'Asset', 'Price']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Invalid format! CSV must contain columns: {', '.join(required_cols)}")
            return None, None
            
        # Parse Dates
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pivot from Tidy to Wide format
        price_wide = df.pivot(index='Date', columns='Asset', values='Price')
        price_wide = price_wide.ffill().dropna()
        
        # If Log_Return exists in CSV, pivot it too, else calculate
        if 'Log_Return' in df.columns:
            returns_wide = df.pivot(index='Date', columns='Asset', values='Log_Return').dropna()
        else:
            returns_wide = compute_returns(price_wide)
            
        return price_wide, returns_wide
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="Market Risk Monitoring - Phase 1", layout="wide")
    st.title("🛡️ Market Risk Monitoring Dashboard")
    st.markdown("### Phase 1: Data Preparation & Exploratory Risk Analysis")

    # --- SIDEBAR ---
    st.sidebar.header("Settings")
    
    # Data Source Selection
    data_source = st.sidebar.radio("Select Data Source", ["Yahoo Finance (Live)", "Upload CSV"])
    
    if data_source == "Yahoo Finance (Live)":
        # Date Range
        default_start = datetime.now() - timedelta(days=10*365)
        default_end = datetime.now()
        start_date = st.sidebar.date_input("Start Date", default_start)
        end_date = st.sidebar.date_input("End Date", default_end)
        
        # Internal loading
        with st.spinner("Fetching data from Yahoo Finance..."):
            price_df = load_data(TICKERS, start_date, end_date)
            log_returns = compute_returns(price_df) if not price_df.empty else pd.DataFrame()
    else:
        # File Uploader
        uploaded_file = st.sidebar.file_uploader("Upload your tidy CSV", type=["csv"])
        if uploaded_file is not None:
            price_df, log_returns = process_uploaded_file(uploaded_file)
        else:
            st.info("Please upload a CSV file to continue.")
            price_df, log_returns = pd.DataFrame(), pd.DataFrame()

    if price_df.empty:
        if data_source == "Yahoo Finance (Live)":
            st.warning("No data found for the selected range.")
        return

    asset_options = ["All"] + list(price_df.columns)
    selected_asset = st.sidebar.selectbox("Select Asset", asset_options)
    
    rolling_window = st.sidebar.slider("Rolling Window (Returns)", 5, 252, 20)
    annualize = st.sidebar.checkbox("Annualize Volatility", value=True)
    corr_window = st.sidebar.number_input("Rolling Correlation Window", 30, 252, 60)

    # --- DATA EXPORT ---
    # Export Button (Only relevant for Live data)
    if data_source == "Yahoo Finance (Live)":
        if st.sidebar.button("Export Cleaned Data"):
            save_csv(price_df, log_returns)
            st.sidebar.success(f"Data saved to {DATA_FILE}")

        # Auto-save on first load if file doesn't exist
        if not os.path.exists(DATA_FILE):
            save_csv(price_df, log_returns)

    # --- TABS FOR SECTIONS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Quality", 
        "📈 Price & Returns", 
        "📉 Distribution & Tail Risk", 
        "🎲 Rolling Risk", 
        "🔗 Relationships"
    ])

    # Filter data for display if not "All"
    filtered_price = price_df if selected_asset == "All" else price_df[[selected_asset]]
    filtered_returns = log_returns if selected_asset == "All" else log_returns[[selected_asset]]

    # 1. Data Quality & Overview
    with tab1:
        st.header("Section 1: Data Quality & Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Date Range", f"{price_df.index[0].date()} to {price_df.index[-1].date()}")
        with col2:
            st.metric("Total Observations", len(price_df))
        with col3:
            st.metric("Missing Values", price_df.isna().sum().sum())
            
        st.subheader("Outlier Detection (Z > 3 or < 1% / > 99%)")
        outlier_summary = {}
        for col in log_returns.columns:
            outliers = detect_outliers(log_returns[col])
            outlier_summary[col] = outliers.sum()
        
        st.write("Number of detected outliers in returns:")
        st.dataframe(pd.Series(outlier_summary, name="Outlier Count"))
        st.info("Note: Outliers may represent real market shocks (e.g., 2020 COVID crash).")

    # 2. Price & Returns
    with tab2:
        st.header("Section 2: Price & Returns")
        
        # Normalize prices for easier comparison if "All" is selected
        if selected_asset == "All":
            norm_price = price_df / price_df.iloc[0] * 100
            fig_price = px.line(norm_price, title="Normalized Price Time Series (Base=100)")
            fig_price.update_layout(yaxis_title="Index Level")
        else:
            fig_price = px.line(filtered_price, title=f"{selected_asset} Price Time Series")
            fig_price.update_layout(yaxis_title="Price (USD)")
            
        st.plotly_chart(fig_price, width="stretch")
        
        fig_returns = px.line(filtered_returns, title=f"Daily Log Returns - {selected_asset}")
        fig_returns.update_layout(yaxis_title="Log Return")
        st.plotly_chart(fig_returns, width="stretch")

    # 3. Return Distribution & Tail Risk
    with tab3:
        st.header("Section 3: Return Distribution & Tail Risk")
        
        if selected_asset == "All":
            st.warning("Please select a specific asset in the sidebar to view distribution analysis.")
        else:
            asset_returns = log_returns[selected_asset].dropna()
            
            # Stats Table
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
                "Value": [
                    asset_returns.mean(),
                    asset_returns.std(),
                    asset_returns.min(),
                    asset_returns.max(),
                    asset_returns.skew(),
                    asset_returns.kurtosis()
                ]
            })
            st.table(stats_df)
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                fig_hist = px.histogram(asset_returns, nbins=50, marginal="box", 
                                        title=f"Histogram of {selected_asset} Returns")
                st.plotly_chart(fig_hist, width="stretch")
                
            with col_right:
                # Q-Q Plot
                qq = stats.probplot(asset_returns, dist="norm")
                qq_df = pd.DataFrame({
                    "Theoretical Quantiles": qq[0][0],
                    "Sample Quantiles": qq[0][1]
                })
                fig_qq = px.scatter(qq_df, x="Theoretical Quantiles", y="Sample Quantiles", 
                                    title=f"Q-Q Plot vs Normal - {selected_asset}")
                fig_qq.add_shape(type="line", x0=qq_df["Theoretical Quantiles"].min(), y0=qq_df["Theoretical Quantiles"].min(),
                                 x1=qq_df["Theoretical Quantiles"].max(), y1=qq_df["Theoretical Quantiles"].max(), line=dict(color="red", dash="dash"))
                st.plotly_chart(fig_qq, width="stretch")

    # 4. Rolling Risk Measures
    with tab4:
        st.header("Section 4: Rolling Risk Measures")
        if selected_asset == "All":
            st.warning("Select a specific asset to see rolling statistics.")
        else:
            r_mean = log_returns[selected_asset].rolling(window=rolling_window).mean()
            r_std = log_returns[selected_asset].rolling(window=rolling_window).std()
            
            if annualize:
                r_std = r_std * np.sqrt(252)
                vol_label = "Annualized Volatility"
            else:
                vol_label = "Daily Volatility"
                
            # Rolling Vol Calculation
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=r_std.index, y=r_std, name=vol_label))
            fig_vol.update_layout(title=f"Rolling {vol_label} ({rolling_window} days) - {selected_asset}", yaxis_title="Volatility")
            st.plotly_chart(fig_vol, width="stretch")
            
            # Rolling Mean with Bands
            show_bands = st.checkbox("Show ±2 Std Bands on Mean", value=True)
            fig_mean = go.Figure()
            fig_mean.add_trace(go.Scatter(x=r_mean.index, y=r_mean, name="Rolling Mean"))
            if show_bands:
                raw_std = log_returns[selected_asset].rolling(window=rolling_window).std()
                fig_mean.add_trace(go.Scatter(x=r_mean.index, y=r_mean + 2*raw_std, name="+2 Std", line=dict(dash='dot', color='gray')))
                fig_mean.add_trace(go.Scatter(x=r_mean.index, y=r_mean - 2*raw_std, name="-2 Std", line=dict(dash='dot', color='gray')))
            
            fig_mean.update_layout(title=f"Rolling Mean & Bands ({rolling_window} days) - {selected_asset}", yaxis_title="Return")
            st.plotly_chart(fig_mean, width="stretch")
            
            st.info("Rolling volatility provides a baseline. Subsequent phases will introduce more reactive models like EWMA and GARCH.")

    # 5. Cross-Asset Relationships
    with tab5:
        st.header("Section 5: Cross-Asset Relationships")
        
        # Heatmap
        corr = log_returns.corr()
        fig_heat = px.imshow(corr, text_auto=True, title="Return Correlation Matrix")
        st.plotly_chart(fig_heat, width="stretch")
        
        # Rolling Correlation
        st.subheader("Rolling Correlation")
        pair = st.multiselect("Select two assets for rolling correlation", options=list(log_returns.columns), default=list(log_returns.columns)[:2])
        if len(pair) == 2:
            roll_corr = log_returns[pair[0]].rolling(window=corr_window).corr(log_returns[pair[1]])
            fig_roll_corr = px.line(roll_corr, title=f"Rolling {corr_window}-day Correlation: {pair[0]} vs {pair[1]}")
            st.plotly_chart(fig_roll_corr, width="stretch")
        else:
            st.write("Select exactly two assets to see rolling correlation.")

if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        # If run directly with 'python app.py', this block triggers 'streamlit run'
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
