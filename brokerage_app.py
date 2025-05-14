import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import yfinance as yf
from datetime import datetime

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Brokerage Account Analyzer Pro")

# --- Helper Functions ---
def clean_monetary_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        symbols_to_remove = ['$', '€', 'â‚¬', 'USD', 'EUR', ' ', ','] # Added comma
        for symbol in symbols_to_remove:
            value = value.replace(symbol, '')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan

def normalize_to_usd(df):
    df_copy = df.copy()

    # Preserve original amount and currency for other calculations (e.g., EUR investment)
    if 'Total Amount' in df_copy.columns:
        df_copy['Original_Amount'] = df_copy['Total Amount'].apply(clean_monetary_value)
    else:
        st.error("Critical Error: 'Total Amount' column is missing.")
        return None
    if 'Currency' in df_copy.columns:
        df_copy['Original_Currency'] = df_copy['Currency'] # Preserve original currency string

    df_copy['Amount_Cleaned'] = df_copy['Original_Amount'] # Already cleaned

    if 'Price per share' in df_copy.columns:
        df_copy['Price_per_share_Cleaned'] = df_copy['Price per share'].apply(clean_monetary_value)
    else:
        df_copy['Price_per_share_Cleaned'] = np.nan

    df_copy['Amount_USD'] = df_copy['Amount_Cleaned']
    df_copy['Price_per_share_USD'] = df_copy['Price_per_share_Cleaned']

    if 'Original_Currency' in df_copy.columns and 'FX Rate' in df_copy.columns:
        df_copy['FX Rate'] = pd.to_numeric(df_copy['FX Rate'], errors='coerce').fillna(1.0)
        # Ensure 'Original_Currency' is treated as string for comparison
        usd_conversion_needed_mask = (df_copy['Original_Currency'].astype(str).str.upper() != 'USD') & \
                                     (df_copy['FX Rate'] != 1.0) & \
                                     (df_copy['FX Rate'].notna())

        df_copy.loc[usd_conversion_needed_mask, 'Amount_USD'] = \
            df_copy.loc[usd_conversion_needed_mask, 'Amount_Cleaned'] * df_copy.loc[usd_conversion_needed_mask, 'FX Rate']
        df_copy.loc[usd_conversion_needed_mask, 'Price_per_share_USD'] = \
            df_copy.loc[usd_conversion_needed_mask, 'Price_per_share_Cleaned'] * df_copy.loc[usd_conversion_needed_mask, 'FX Rate']
    else:
        st.warning("Warning: 'Currency' or 'FX Rate' column not found or improperly formatted. Assuming all amounts are in USD or a consistent currency for USD conversion. Live prices will be fetched assuming USD markets.")

    df_copy.rename(columns={'Amount_USD': 'Amount'}, inplace=True) # 'Amount' will now refer to Amount_USD
    return df_copy


def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        if 'Date' not in df.columns:
            st.error("Critical Error: 'Date' column is missing.")
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        # Attempt to sort by Date and potentially a sub-second column if it exists, to ensure order
        # If you have a 'Time' column or your 'Date' column has time precise enough:
        # df = df.sort_values(by=['Date', 'Time_Column_If_Any'], ascending=[True, True])
        # For now, just date:
        df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)


        df = normalize_to_usd(df)
        if df is None: return None
        df.dropna(subset=['Amount', 'Type'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading or parsing the Excel file: {e}")
        return None

def get_live_price(ticker_symbol, last_known_price_usd_from_data):
    if ticker_symbol.endswith('Q'):
        return 0.0, "Delisted/Bankrupt (Value $0)"
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="2d")
        if not hist.empty and 'Close' in hist.columns:
            live_price = hist['Close'].iloc[-1]
            return live_price, "Live Market"
        st.warning(f"Could not fetch recent history for {ticker_symbol} via yfinance. Using last transaction price as fallback.")
        return last_known_price_usd_from_data, "Last Txn Price (Fallback)"
    except Exception: # Catch broad exceptions from yf
        # st.warning(f"Failed to fetch live price for {ticker_symbol} from yfinance: {str(e)[:100]}. Using last transaction price.")
        return last_known_price_usd_from_data, "Last Txn Price (API Error)"

def calculate_total_invested_in_currency(df, target_currency="EUR"):
    """Calculates total amount invested (buys) in a specific original currency."""
    if df is None or not all(col in df.columns for col in ['Original_Amount', 'Original_Currency', 'Type']):
        return 0
    
    # Filter for buy transactions in the target currency
    # Ensure 'Original_Currency' is compared as uppercase string
    buy_transactions_in_target_currency = df[
        (df['Type'].isin(['BUY - MARKET'])) &
        (df['Original_Currency'].astype(str).str.upper() == target_currency.upper())
    ]
    total_invested = buy_transactions_in_target_currency['Original_Amount'].sum()
    return total_invested


def calculate_total_deposits(df):
    if 'CASH TOP-UP' in df['Type'].unique():
        deposits = df[df['Type'] == 'CASH TOP-UP']['Amount'].sum() # Amount is already USD
        return deposits
    return 0

def calculate_portfolio_metrics(df_transactions):
    if df_transactions is None or df_transactions.empty:
        return 0, 0, pd.DataFrame(), 0, 0, 0

    # These are always in USD after normalization
    buy_types = ['BUY - MARKET']
    sell_types = ['SELL - MARKET']
    fee_types = ['CUSTODY FEE', 'FEE', 'SERVICE FEE'] # Ensure 'Amount' for fees is positive debit
    dividend_types = ['DIVIDEND'] # Ensure 'Amount' for dividends is positive credit

    # Overall sums for P&L calculation (all in USD)
    grand_total_cost_of_all_buys_usd = df_transactions[df_transactions['Type'].isin(buy_types)]['Amount'].sum()
    grand_total_proceeds_from_all_sells_usd = df_transactions[df_transactions['Type'].isin(sell_types)]['Amount'].sum()
    grand_total_dividends_received_usd = df_transactions[df_transactions['Type'].isin(dividend_types)]['Amount'].sum()
    grand_total_fees_paid_usd = df_transactions[df_transactions['Type'].isin(fee_types)]['Amount'].sum()


    # For tracking holdings and average costs (all in USD)
    # This dictionary will store: quantity, total_cost_of_held_shares, weighted_avg_buy_price_of_held
    live_holdings = {}
    total_realized_pl_from_sales_usd = 0.0 # Accumulator for P/L from actual sales

    # Get last transaction prices (USD) as a base for fallback if yfinance fails
    last_transaction_prices_usd_from_data = {}
    if 'Price_per_share_USD' in df_transactions.columns and 'Ticker' in df_transactions.columns:
        last_transaction_prices_usd_from_data = df_transactions.dropna(subset=['Price_per_share_USD', 'Ticker'])\
                                               .groupby('Ticker')['Price_per_share_USD'].last().to_dict()

    # Process transactions chronologically (already sorted in load_data)
    for index, row in df_transactions.iterrows():
        ticker = row.get('Ticker')
        tx_type = row.get('Type')
        quantity = row.get('Quantity', 0) # Default to 0 if missing
        amount_usd = row.get('Amount', 0) # This is Amount_USD
        price_per_share_usd = row.get('Price_per_share_USD', 0) #This is Price_per_share_USD

        if pd.isna(ticker) and tx_type not in fee_types + dividend_types + ['CASH TOP-UP']: # Ticker needed for buy/sell
            continue
        if pd.isna(quantity) and tx_type in buy_types + sell_types: # Quantity needed for buy/sell
            continue


        if tx_type in buy_types:
            if ticker not in live_holdings:
                live_holdings[ticker] = {'quantity': 0, 'cost_basis_held_usd': 0.0, 'avg_buy_price_held_usd': 0.0}

            current_qty = live_holdings[ticker]['quantity']
            current_cost_basis = live_holdings[ticker]['cost_basis_held_usd']

            # Update holdings
            new_quantity = current_qty + quantity
            new_cost_basis = current_cost_basis + amount_usd # amount_usd is total cost of this buy txn

            live_holdings[ticker]['quantity'] = new_quantity
            live_holdings[ticker]['cost_basis_held_usd'] = new_cost_basis
            if new_quantity > 0: # Avoid division by zero
                live_holdings[ticker]['avg_buy_price_held_usd'] = new_cost_basis / new_quantity
            else: # Should not happen if quantity > 0 for a buy
                 live_holdings[ticker]['avg_buy_price_held_usd'] = 0


        elif tx_type in sell_types:
            if ticker in live_holdings and live_holdings[ticker]['quantity'] > 0:
                avg_buy_price_at_sale = live_holdings[ticker]['avg_buy_price_held_usd']
                sold_quantity = min(quantity, live_holdings[ticker]['quantity']) # Cannot sell more than held

                cost_of_goods_sold = sold_quantity * avg_buy_price_at_sale
                proceeds_this_sale = amount_usd # amount_usd is total proceeds of this sell txn
                realized_pl_this_sale = proceeds_this_sale - cost_of_goods_sold
                total_realized_pl_from_sales_usd += realized_pl_this_sale

                # Update holdings
                live_holdings[ticker]['quantity'] -= sold_quantity
                live_holdings[ticker]['cost_basis_held_usd'] -= cost_of_goods_sold
                # avg_buy_price_held_usd remains the same for the shares still held

                if live_holdings[ticker]['quantity'] <= 0.00001: # Handle float precision
                    del live_holdings[ticker] # Remove if all sold
            # else: sell of untracked stock or more than available - can be warning

    # --- Calculate Current Portfolio Value using live prices ---
    current_portfolio_market_value_usd = 0
    holdings_df_data = []
    live_prices_fetched_count = 0
    
    total_investment_in_q_stocks_usd = df_transactions[
        (df_transactions['Type'].isin(buy_types)) &
        (df_transactions['Ticker'].astype(str).str.endswith('Q'))
    ]['Amount'].sum()


    if live_holdings: # Check if there are any holdings left
        st.write("Fetching live prices for current holdings (this may take a moment)...")
        progress_bar = st.progress(0)
        num_holdings_to_fetch = len(live_holdings)
        fetched_count = 0

        for ticker, data in live_holdings.items():
            if data['quantity'] > 0.00001: # If shares are held
                last_known_price_usd = last_transaction_prices_usd_from_data.get(ticker, data['avg_buy_price_held_usd'])

                live_price_usd, price_source = get_live_price(ticker, last_known_price_usd)
                if price_source == "Live Market":
                    live_prices_fetched_count += 1

                market_value_of_holding = data['quantity'] * live_price_usd
                current_portfolio_market_value_usd += market_value_of_holding
                unrealized_pl_holding = market_value_of_holding - data['cost_basis_held_usd']
                percent_unrealized_pl = (unrealized_pl_holding / data['cost_basis_held_usd']) * 100 if data['cost_basis_held_usd'] != 0 else 0

                holdings_df_data.append({
                    'Ticker': ticker,
                    'Quantity': data['quantity'],
                    'Avg. Buy Price (USD)': data['avg_buy_price_held_usd'],
                    'Cost Basis (USD)': data['cost_basis_held_usd'],
                    'Current Price (USD)': live_price_usd,
                    'Price Source': price_source,
                    'Market Value (USD)': market_value_of_holding,
                    'Unrealized P/L (USD)': unrealized_pl_holding,
                    '% Unrealized P/L': percent_unrealized_pl
                })
            fetched_count += 1
            progress_bar.progress(fetched_count / num_holdings_to_fetch)
        progress_bar.empty()

    holdings_df = pd.DataFrame(holdings_df_data)

    # --- Final Overall P&L ---
    # P/L = (Current Portfolio Value + Total Sells + Total Dividends) - (Total Buys + Total Fees)
    overall_net_pl_usd = (current_portfolio_market_value_usd + grand_total_proceeds_from_all_sells_usd + grand_total_dividends_received_usd) - \
                         (grand_total_cost_of_all_buys_usd + grand_total_fees_paid_usd)

    if st.sidebar.checkbox("Show P&L Calculation Breakdown", False):
        st.sidebar.markdown("---")
        st.sidebar.subheader("P&L Calculation (USD):")
        st.sidebar.markdown(f"""
        **Income Side:**
        - Est. Current Holdings Market Value: ${current_portfolio_market_value_usd:,.2f}
        - Total Proceeds from All Sells: ${grand_total_proceeds_from_all_sells_usd:,.2f}
        - Total Dividends Received: ${grand_total_dividends_received_usd:,.2f}
        ---
        **Subtotal (Income Side):** ${(current_portfolio_market_value_usd + grand_total_proceeds_from_all_sells_usd + grand_total_dividends_received_usd):,.2f}
        
        **Cost Side:**
        - Total Cost of All Buys: ${grand_total_cost_of_all_buys_usd:,.2f}
        - Total Fees Paid: ${grand_total_fees_paid_usd:,.2f}
        ---
        **Subtotal (Cost Side):** ${(grand_total_cost_of_all_buys_usd + grand_total_fees_paid_usd):,.2f}
        ---
        **Estimated Overall Net P/L:** ${overall_net_pl_usd:,.2f}
        
        *Internal check: Total Realized P/L from sales (before dividends/fees): ${total_realized_pl_from_sales_usd:,.2f}*
        """)
        st.sidebar.info(f"_{live_prices_fetched_count} ticker(s) updated with live market prices._ Tickers ending in 'Q' are valued at $0.")

    return overall_net_pl_usd, current_portfolio_market_value_usd, holdings_df, live_prices_fetched_count, total_investment_in_q_stocks_usd


# --- Main Application ---
st.title("📈 Brokerage Account Analyzer Pro")
st.markdown(f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Live prices fetched on data load)")

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel transactions file (.xlsx, .xls)", type=["xlsx", "xls"])

if uploaded_file:
    df_transactions_loaded = load_data(uploaded_file)

    if df_transactions_loaded is not None and not df_transactions_loaded.empty:
        st.success("File processed! Monetary values below are in USD (converted via FX rates if provided).")

        # --- Key Metrics ---
        total_deposited_usd = calculate_total_deposits(df_transactions_loaded)
        total_invested_eur = calculate_total_invested_in_currency(df_transactions_loaded, "EUR")
        # Add other currencies if needed: total_invested_gbp = calculate_total_invested_in_currency(df_transactions_loaded, "GBP")


        overall_pl_usd, current_mkt_val_usd, df_holdings, live_cnt, investment_q_usd = \
            calculate_portfolio_metrics(df_transactions_loaded)

        st.subheader("📊 Key Financial Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Deposited (Cash Top-ups, USD)", f"${total_deposited_usd:,.2f}")
        col2.metric("Est. Current Portfolio Market Value (USD)", f"${current_mkt_val_usd:,.2f}",
                    help="Based on live market prices (or $0 for 'Q' stocks, fallback to last txn price).")
        col3.metric("Est. Overall Net Profit/Loss (USD)", f"${overall_pl_usd:,.2f}",
                     delta_color="normal" if overall_pl_usd >=0 else "inverse")
        
        if total_invested_eur > 0:
             st.metric("Total Invested (Buys) in EUR", f"€{total_invested_eur:,.2f}",
                       help="Sum of 'Total Amount' for BUY transactions where currency was EUR, before FX conversion.")

        if investment_q_usd > 0:
            st.warning(f"⚠️ Total investment (cost of buys) in 'Q' (delisted/bankrupt) stocks: ${investment_q_usd:,.2f} USD. These are currently valued at $0 in your portfolio.")

        if live_cnt > 0:
            st.info(f"💡 Successfully fetched live market prices for {live_cnt} ticker(s).")
        else:
            st.warning("⚠️ Could not fetch any live market prices. Portfolio valuation uses last transaction prices or $0 for 'Q' stocks.")

        # --- Detailed Holdings View ---
        st.markdown("---")
        st.subheader("💼 Current Holdings (Unsold Shares)")
        st.write("Detailed breakdown of shares you currently own, valued using live market prices (or fallback).")
        if not df_holdings.empty:
            # Sort by Market Value Descending for better view
            df_holdings_sorted = df_holdings.sort_values(by="Market Value (USD)", ascending=False).reset_index(drop=True)
            st.dataframe(df_holdings_sorted.style.format({
                "Quantity": "{:.2f}",
                "Avg. Buy Price (USD)": "${:,.2f}",
                "Cost Basis (USD)": "${:,.2f}",
                "Current Price (USD)": "${:,.2f}",
                "Market Value (USD)": "${:,.2f}",
                "Unrealized P/L (USD)": "${:,.2f}",
                "% Unrealized P/L": "{:.2f}%"
            }).applymap(lambda val: 'color: red' if isinstance(val, (float, int)) and val < 0 and '%' in str(val) else ('color: green' if isinstance(val, (float, int)) and val > 0 and '%' in str(val) else ''), subset=['% Unrealized P/L'])
              .applymap(lambda val: 'color: red' if isinstance(val, (float, int)) and val < 0 else ('color: green' if isinstance(val, (float, int)) and val > 0 else ''), subset=['Unrealized P/L (USD)']))
        else:
            st.info("No current active holdings to display based on processed transactions.")

        # --- Charts ---
        st.markdown("---")
        st.subheader("📈 Transaction Charts (USD)")
        # (Charts code remains largely the same, ensure 'Amount' used is the USD converted one)
        if 'CASH TOP-UP' in df_transactions_loaded['Type'].unique():
            # Charting logic as before
            deposits_over_time = df_transactions_loaded[df_transactions_loaded['Type'] == 'CASH TOP-UP'].groupby(pd.Grouper(key='Date', freq='M'))['Amount'].sum().reset_index()
            deposits_over_time['DateStr'] = deposits_over_time['Date'].dt.strftime('%Y-%m')
            chart_deposits = alt.Chart(deposits_over_time).mark_bar().encode(
                x=alt.X('DateStr:O', title='Month', sort=None),
                y=alt.Y('Amount:Q', title='Total Deposits ($ USD)'),
                tooltip=[alt.Tooltip('DateStr', title='Month'), alt.Tooltip('Amount', title='Deposits ($ USD)', format='$,.2f')]
            ).properties(title='Monthly Cash Top-ups (USD)')
            st.altair_chart(chart_deposits, use_container_width=True)

        buys_sells_df = df_transactions_loaded[df_transactions_loaded['Type'].isin(['BUY - MARKET', 'SELL - MARKET'])].copy()
        if not buys_sells_df.empty:
            # Charting logic as before
            buys_sells_over_time = buys_sells_df.groupby([pd.Grouper(key='Date', freq='M'), 'Type'])['Amount'].sum().unstack(fill_value=0).reset_index()
            buys_sells_over_time['DateStr'] = buys_sells_over_time['Date'].dt.strftime('%Y-%m')
            buys_sells_over_time_melted = buys_sells_over_time.melt(id_vars=['DateStr', 'Date'], value_vars=['BUY - MARKET', 'SELL - MARKET'], var_name='Transaction Type', value_name='Total Amount ($ USD)')
            chart_buys_sells = alt.Chart(buys_sells_over_time_melted).mark_line(point=True).encode(
                x=alt.X('DateStr:O', title='Month', sort=alt.EncodingSortField(field="Date", op="min", order='ascending')),
                y=alt.Y('Total Amount ($ USD):Q', title='Total Transaction Amount ($ USD)'),
                color='Transaction Type:N',
                tooltip=[alt.Tooltip('DateStr', title='Month'), 'Transaction Type', alt.Tooltip('Total Amount ($ USD)', format='$,.2f')]
            ).properties(title='Monthly Buy vs. Sell Activity (Value in USD)')
            st.altair_chart(chart_buys_sells, use_container_width=True)
        
        # Cumulative chart (Optional enhancement: Add cumulative P&L line)
        # ... (existing cumulative chart code can largely remain)


        st.markdown("---")
        st.subheader("📋 Processed Transaction Data")
        st.write("View your uploaded transactions after processing (monetary values converted to USD where applicable, original amounts preserved).")
        display_cols = ['Date', 'Ticker', 'Type', 'Quantity', 'Price per share', 'Original_Currency', 'Original_Amount', 'FX Rate', 'Amount', 'Price_per_share_USD']
        display_cols = [col for col in display_cols if col in df_transactions_loaded.columns]
        st.dataframe(df_transactions_loaded[display_cols].sort_values(by='Date', ascending=False).style.format({
            "Original_Amount": "{:,.2f}", "Amount": "${:,.2f}", # Amount is USD
            "Price per share": "{:,.2f}", "Price_per_share_USD": "${:,.2f}",
            "FX Rate": "{:.4f}", "Quantity": "{:.4f}" # Increased precision for quantity
        }))

    elif df_transactions_loaded is not None and df_transactions_loaded.empty:
        st.warning("The uploaded Excel file resulted in no processable transaction data (e.g., all rows had critical missing values).")
    elif uploaded_file and df_transactions_loaded is None: # Error during load_data
        st.error("Failed to load or process the Excel file. Check console for specific errors if running locally, or ensure file format and essential columns like 'Date', 'Total Amount', 'Type' are correct.")
else:
    st.info("👈 Upload your brokerage transaction Excel file to begin analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Disclaimers & Notes:**
* This app uses the **Average Cost Basis** method for P&L calculations on sales and for valuing current holdings.
* Transactions are processed chronologically by 'Date'. Ensure your data is accurate.
* Live market prices from Yahoo Finance are subject to delays. Tickers ending in 'Q' are assumed to have $0 value.
* Currency conversions rely on 'FX Rate' in your file. 'Total Invested in EUR' (etc.) refers to original buy amounts in that currency.
* This tool is for informational purposes only, not financial advice.
""")