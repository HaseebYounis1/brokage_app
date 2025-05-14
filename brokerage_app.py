import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Brokerage Account Analyzer Pro")

# --- Session State Initialization ---
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = pd.DataFrame()
if 'manual_entries' not in st.session_state:
    st.session_state.manual_entries = []
if 'eur_usd_rate' not in st.session_state:
    st.session_state.eur_usd_rate = None
if 'last_fx_fetch_time' not in st.session_state:
    st.session_state.last_fx_fetch_time = None

# --- Helper Functions (assuming these are the same as before) ---
def clean_monetary_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        symbols_to_remove = ['$', '€', 'â‚¬', 'USD', 'EUR', ' ', ',']
        for symbol in symbols_to_remove:
            value = value.replace(symbol, '')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan

def fetch_eur_usd_rate():
    """Fetches the latest EUR/USD exchange rate."""
    now = datetime.now()
    if st.session_state.eur_usd_rate and st.session_state.last_fx_fetch_time and \
       (now - st.session_state.last_fx_fetch_time < timedelta(hours=1)):
        return st.session_state.eur_usd_rate
    try:
        ticker = yf.Ticker("EURUSD=X")
        hist = ticker.history(period="2d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            st.session_state.eur_usd_rate = rate
            st.session_state.last_fx_fetch_time = now
            return rate
        return None
    except Exception:
        st.warning("Failed to fetch EUR/USD rate from yfinance. EUR conversions might be affected.", icon="⚠️")
        return None


def convert_usd_to_display_currency(amount_usd, display_currency, eur_usd_rate):
    if display_currency == "EUR" and eur_usd_rate is not None and eur_usd_rate != 0:
        return amount_usd / eur_usd_rate
    return amount_usd

def format_currency(amount, display_currency, eur_usd_rate):
    target_amount = convert_usd_to_display_currency(amount, display_currency, eur_usd_rate)
    symbol = "€" if display_currency == "EUR" else "$"
    return f"{symbol}{target_amount:,.2f}"


def normalize_to_usd(df):
    df_copy = df.copy()
    required_cols = ['Total Amount', 'Currency']
    if not all(col in df_copy.columns for col in required_cols):
        st.error(f"Critical Error: One or more required columns ({', '.join(required_cols)}) are missing for currency normalization.")
        return None

    df_copy['Original_Amount'] = df_copy['Total Amount'].apply(clean_monetary_value)
    df_copy['Original_Currency'] = df_copy['Currency'].astype(str).str.upper()

    df_copy['Price_per_share_Cleaned'] = np.nan
    if 'Price per share' in df_copy.columns:
        df_copy['Price_per_share_Cleaned'] = df_copy['Price per share'].apply(clean_monetary_value)

    df_copy['Amount_USD'] = df_copy['Original_Amount']
    df_copy['Price_per_share_USD'] = df_copy['Price_per_share_Cleaned']

    if 'FX Rate' in df_copy.columns:
        df_copy['FX Rate'] = pd.to_numeric(df_copy['FX Rate'], errors='coerce')
        conversion_needed_mask = (df_copy['Original_Currency'] != 'USD') & (df_copy['FX Rate'].notna()) & (df_copy['FX Rate'] != 0)
        df_copy.loc[conversion_needed_mask, 'Amount_USD'] = \
            df_copy.loc[conversion_needed_mask, 'Original_Amount'] * df_copy.loc[conversion_needed_mask, 'FX Rate']
        df_copy.loc[conversion_needed_mask, 'Price_per_share_USD'] = \
            df_copy.loc[conversion_needed_mask, 'Price_per_share_Cleaned'] * df_copy.loc[conversion_needed_mask, 'FX Rate']
        df_copy['FX Rate'].fillna(1.0, inplace=True)
        df_copy.loc[df_copy['Original_Currency'] == 'USD', 'FX Rate'] = 1.0
    else:
        st.warning("Warning: 'FX Rate' column not found. Assuming amounts in 'USD' currency are already USD. Other currencies might not be converted correctly without an FX rate.")
        df_copy['FX Rate'] = 1.0

    df_copy.rename(columns={'Amount_USD': 'Amount'}, inplace=True)
    return df_copy

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                # Read the raw content to check for delimiter
                content_peek = uploaded_file.read(1024).decode('utf-8', errors='ignore') # Read first 1KB
                uploaded_file.seek(0) # Reset buffer
                if ';' in content_peek.splitlines()[0]: # Check header line for semicolon
                    df = pd.read_csv(uploaded_file, delimiter=';')
                else:
                    df = pd.read_csv(uploaded_file)
            except Exception as e_csv:
                 st.error(f"Error reading CSV: {e_csv}. Ensure it's a standard comma or semicolon-separated file.")
                 return None
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file type. Please upload .csv, .xlsx, or .xls")
            return None

        # --- Date Parsing Start ---
        if 'Date' not in df.columns:
            st.error("Critical Error: 'Date' column is missing in the uploaded file.")
            return None
        try:
            # Make a copy of original date strings for robust error reporting
            original_date_strings = df['Date'].astype(str).copy()

            # Use format='mixed' to allow pandas to infer the format for each element individually.
            # This is robust for columns with slight variations in ISO8601 date string format
            # (e.g., with/without fractional seconds, different timezone representations if any beyond 'Z').
            # errors='coerce' will turn unparseable dates into NaT (Not a Time).
            df['Date'] = pd.to_datetime(original_date_strings, format='mixed', errors='coerce')

            if df['Date'].isnull().any():
                num_failed = df['Date'].isnull().sum()
                failed_mask = df['Date'].isnull()
                # Get unique original problematic strings
                example_failed_values = original_date_strings[failed_mask].unique()[:3]

                st.error(
                    f"Error parsing 'Date' column: {num_failed} date(s) could not be converted. "
                    f"Please ensure all dates are in a recognizable format (e.g., YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, or ISO8601 variants like '2024-06-07T16:52:58.164724Z' or '2025-04-03T14:27:24Z'). "
                    f"Example(s) of problematic date strings from your file: {', '.join(example_failed_values)}. "
                    "Rows with unparseable dates cannot be processed correctly."
                )
                return None # Stop processing if dates are bad
        except Exception as e_date: # Catch other potential errors during conversion
            st.error(f"An unexpected error occurred while parsing 'Date' column: {e_date}. Please check date formats in your file.")
            return None
        # --- Date Parsing End ---
        
        df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

        # Standardize common column name variations BEFORE checking for required columns
        column_name_map = {
            "Total Amo": "Total Amount",
            "Price per s": "Price per share",
            # Add other common variations if needed
        }
        df.rename(columns=column_name_map, inplace=True)


        if 'Type' not in df.columns:
            st.error("Critical Error: 'Type' column (transaction type) is missing.")
            return None
        df['Type'] = df['Type'].astype(str).str.strip().str.upper()

        df = normalize_to_usd(df) # This function now expects "Total Amount"
        if df is None: return None

        df.dropna(subset=['Amount', 'Type'], inplace=True) # Amount is Amount_USD
        return df
    except Exception as e:
        st.error(f"Error loading or parsing the file: {e}")
        return None

def get_live_price(ticker_symbol, last_known_price_usd_from_data):
    if pd.isna(ticker_symbol) or not isinstance(ticker_symbol, str):
        return 0.0, "Invalid Ticker"
    if ticker_symbol.endswith('Q'):
        return 0.0, "Delisted/Bankrupt (Value $0)"
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")
        if not hist.empty and 'Close' in hist.columns:
            live_price = hist['Close'].iloc[-1]
            if hist.index[-1].date() >= (datetime.now() - timedelta(days=7)).date():
                 return live_price, "Live Market"
            else:
                 st.warning(f"Price for {ticker_symbol} from yfinance is older than 7 days. Using last transaction price as fallback.", icon="⚠️")
                 return last_known_price_usd_from_data, "Last Txn Price (Stale Live Data)"
        st.warning(f"Could not fetch recent history for {ticker_symbol}. Using last transaction price.", icon="⚠️")
        return last_known_price_usd_from_data, "Last Txn Price (No Live Data)"
    except Exception:
        return last_known_price_usd_from_data, "Last Txn Price (API Error)"

def calculate_total_invested_in_currency(df, target_currency="EUR"):
    if df is None or not all(col in df.columns for col in ['Original_Amount', 'Original_Currency', 'Type']):
        return 0
    buy_transactions_in_target_currency = df[
        (df['Type'].str.contains('BUY', case=False)) &
        (df['Original_Currency'].astype(str).str.upper() == target_currency.upper())
    ]
    return buy_transactions_in_target_currency['Original_Amount'].sum()

def calculate_total_deposits(df):
    if df is None or 'Type' not in df.columns or 'Amount' not in df.columns:
        return 0
    deposit_types = ['CASH TOP-UP', 'DEPOSIT', 'WIRE IN']
    deposits = df[df['Type'].str.upper().isin(deposit_types)]['Amount'].sum()
    return deposits

def calculate_portfolio_metrics(df_transactions):
    if df_transactions is None or df_transactions.empty:
        # Return default zero values for all expected metrics
        return 0, 0, pd.DataFrame(), pd.DataFrame(), 0, 0, 0, 0, 0, 0, 0

    BUY_TYPES = ['BUY - MARKET', 'BUY']
    SELL_TYPES = ['SELL - MARKET', 'SELL']
    FEE_TYPES = ['CUSTODY FEE', 'FEE', 'SERVICE FEE', 'COMMISSION']
    DIVIDEND_TYPES = ['DIVIDEND', 'DIVIDEND INCOME']
    CASH_TOP_UP_TYPES = ['CASH TOP-UP', 'DEPOSIT']

    df_transactions['Quantity'] = pd.to_numeric(df_transactions['Quantity'], errors='coerce').fillna(0)

    grand_total_cost_of_all_buys_usd = df_transactions[df_transactions['Type'].isin(BUY_TYPES)]['Amount'].sum()
    grand_total_proceeds_from_all_sells_usd = df_transactions[df_transactions['Type'].isin(SELL_TYPES)]['Amount'].sum()
    grand_total_dividends_received_usd = df_transactions[df_transactions['Type'].isin(DIVIDEND_TYPES)]['Amount'].sum()
    grand_total_fees_paid_usd = df_transactions[df_transactions['Type'].isin(FEE_TYPES)]['Amount'].sum()

    live_holdings = {}
    completed_transactions_data = []

    last_transaction_prices_usd_from_data = {}
    if 'Price_per_share_USD' in df_transactions.columns and 'Ticker' in df_transactions.columns:
        last_transaction_prices_usd_from_data = df_transactions.dropna(subset=['Price_per_share_USD', 'Ticker'])\
                                               .groupby('Ticker')['Price_per_share_USD'].last().to_dict()

    for index, row in df_transactions.iterrows():
        ticker = row.get('Ticker')
        tx_type = row.get('Type')
        quantity = row.get('Quantity', 0)
        amount_usd = row.get('Amount', 0)
        price_per_share_usd = row.get('Price_per_share_USD', 0)
        tx_date = row.get('Date')

        if pd.isna(ticker) and tx_type not in FEE_TYPES + DIVIDEND_TYPES + CASH_TOP_UP_TYPES:
            continue
        if pd.isna(quantity) and tx_type in BUY_TYPES + SELL_TYPES:
            continue

        if tx_type in BUY_TYPES:
            if ticker not in live_holdings:
                live_holdings[ticker] = {'quantity': 0, 'cost_basis_held_usd': 0.0, 'avg_buy_price_held_usd': 0.0}
            
            current_qty = live_holdings[ticker]['quantity']
            current_cost_basis = live_holdings[ticker]['cost_basis_held_usd']
            
            new_quantity = current_qty + quantity
            new_cost_basis = current_cost_basis + amount_usd

            live_holdings[ticker]['quantity'] = new_quantity
            live_holdings[ticker]['cost_basis_held_usd'] = new_cost_basis
            if new_quantity > 0.000001:
                live_holdings[ticker]['avg_buy_price_held_usd'] = new_cost_basis / new_quantity
            else:
                live_holdings[ticker]['avg_buy_price_held_usd'] = 0

        elif tx_type in SELL_TYPES:
            if ticker in live_holdings and live_holdings[ticker]['quantity'] > 0.000001:
                avg_buy_price_at_sale_usd = live_holdings[ticker]['avg_buy_price_held_usd']
                sold_quantity = min(quantity, live_holdings[ticker]['quantity'])
                cost_of_goods_sold_usd = sold_quantity * avg_buy_price_at_sale_usd
                proceeds_this_sale_usd = amount_usd
                actual_sell_price_per_share_usd = price_per_share_usd
                if (pd.isna(actual_sell_price_per_share_usd) or actual_sell_price_per_share_usd == 0) and sold_quantity > 0:
                    actual_sell_price_per_share_usd = proceeds_this_sale_usd / sold_quantity
                realized_pl_this_sale_usd = proceeds_this_sale_usd - cost_of_goods_sold_usd

                completed_transactions_data.append({
                    'Date': tx_date, 'Ticker': ticker, 'Quantity Sold': sold_quantity,
                    'Avg. Buy Price (USD)': avg_buy_price_at_sale_usd,
                    'Sell Price p.s. (USD)': actual_sell_price_per_share_usd,
                    'Cost Basis Sold (USD)': cost_of_goods_sold_usd,
                    'Total Proceeds (USD)': proceeds_this_sale_usd,
                    'Realized P/L (USD)': realized_pl_this_sale_usd
                })
                live_holdings[ticker]['quantity'] -= sold_quantity
                live_holdings[ticker]['cost_basis_held_usd'] -= cost_of_goods_sold_usd
                if live_holdings[ticker]['quantity'] <= 0.00001:
                    del live_holdings[ticker]

    current_portfolio_market_value_usd = 0
    holdings_df_data = []
    live_prices_fetched_count = 0
    
    total_investment_in_q_stocks_usd = df_transactions[
        (df_transactions['Type'].isin(BUY_TYPES)) &
        (df_transactions['Ticker'].astype(str).str.endswith('Q'))
    ]['Amount'].sum()

    if live_holdings:
        num_holdings_to_fetch = len(live_holdings)
        fetched_count = 0
        for ticker, data in live_holdings.items():
            if data['quantity'] > 0.00001:
                last_known_price_usd = last_transaction_prices_usd_from_data.get(ticker, data['avg_buy_price_held_usd'])
                live_price_usd, price_source = get_live_price(ticker, last_known_price_usd)
                if price_source == "Live Market":
                    live_prices_fetched_count += 1
                market_value_of_holding = data['quantity'] * live_price_usd
                current_portfolio_market_value_usd += market_value_of_holding
                unrealized_pl_holding = market_value_of_holding - data['cost_basis_held_usd']
                percent_unrealized_pl = (unrealized_pl_holding / data['cost_basis_held_usd'] * 100) if data['cost_basis_held_usd'] != 0 else 0
                holdings_df_data.append({
                    'Ticker': ticker, 'Quantity': data['quantity'],
                    'Avg. Buy Price (USD)': data['avg_buy_price_held_usd'],
                    'Cost Basis (USD)': data['cost_basis_held_usd'],
                    'Current Price (USD)': live_price_usd, 'Price Source': price_source,
                    'Market Value (USD)': market_value_of_holding,
                    'Unrealized P/L (USD)': unrealized_pl_holding,
                    '% Unrealized P/L': percent_unrealized_pl
                })
            fetched_count += 1

    holdings_df = pd.DataFrame(holdings_df_data)
    completed_transactions_df = pd.DataFrame(completed_transactions_data)
    if not completed_transactions_df.empty:
        completed_transactions_df = completed_transactions_df.sort_values(by="Date", ascending=False)

    total_realized_pl_from_sales_usd = 0
    if not completed_transactions_df.empty:
        total_realized_pl_from_sales_usd = completed_transactions_df['Realized P/L (USD)'].sum()
    
    overall_net_pl_usd = (current_portfolio_market_value_usd + grand_total_proceeds_from_all_sells_usd + grand_total_dividends_received_usd) - \
                         (grand_total_cost_of_all_buys_usd + grand_total_fees_paid_usd)
    
    # This is an alternative way to calculate overall P/L, good for cross-checking:
    # total_unrealized_pl_usd = holdings_df['Unrealized P/L (USD)'].sum() if not holdings_df.empty else 0
    # check_overall_pl = total_unrealized_pl_usd + total_realized_pl_from_sales_usd + grand_total_dividends_received_usd - grand_total_fees_paid_usd
    # if abs(overall_net_pl_usd - check_overall_pl) > 0.01: # Small tolerance for float precision
    #     st.warning(f"P/L Mismatch: Method1 ${overall_net_pl_usd:.2f}, Method2 ${check_overall_pl:.2f}", icon="⚠️")

    return (overall_net_pl_usd, current_portfolio_market_value_usd, holdings_df, completed_transactions_df,
            live_prices_fetched_count, total_investment_in_q_stocks_usd,
            grand_total_cost_of_all_buys_usd, grand_total_proceeds_from_all_sells_usd,
            grand_total_dividends_received_usd, grand_total_fees_paid_usd, total_realized_pl_from_sales_usd)


# --- Main Application ---
st.title("📈 Brokerage Account Analyzer Pro")

# --- Sidebar for Upload, Manual Entry, and Settings ---
st.sidebar.header("⚙️ Settings & Data Input")
display_currency = st.sidebar.selectbox("Display Monetary Values In:", ["USD", "EUR"], key="display_currency_select")
eur_usd_live_rate = fetch_eur_usd_rate()

if display_currency == "EUR":
    if eur_usd_live_rate:
        st.sidebar.caption(f"Using EUR/USD: {eur_usd_live_rate:.4f} (live rate)")
    # Warning for failed fetch is now inside fetch_eur_usd_rate

uploaded_file = st.sidebar.file_uploader("Upload Transactions File (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])

st.sidebar.subheader("✍️ Manual Transaction Entry")
with st.sidebar.expander("Add a Transaction"):
    with st.form("manual_transaction_form", clear_on_submit=True):
        mt_date = st.date_input("Date", value=datetime.now())
        mt_ticker = st.text_input("Ticker")
        mt_type = st.selectbox("Type", ["BUY - MARKET", "SELL - MARKET", "DIVIDEND", "CASH TOP-UP", "FEE", "CUSTODY FEE", "SERVICE FEE"])
        mt_quantity = st.number_input("Quantity", min_value=0.0, step=0.0001, format="%.4f")
        mt_price_per_share = st.number_input("Price per Share (in transaction currency)", min_value=0.0, format="%.4f")
        mt_total_amount = st.number_input("Total Amount (in transaction currency, overrides PxQ)", min_value=0.0, format="%.2f")
        mt_currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "Other"], index=0)
        mt_fx_rate = st.number_input("FX Rate (to USD, e.g., EUR to USD rate if currency is EUR)", value=1.0, min_value=0.0, format="%.6f",
                                     help="If currency is USD, FX rate is 1. If EUR, provide EUR/USD rate at time of transaction.")
        submitted = st.form_submit_button("Add Transaction")
        if submitted:
            if not mt_ticker and mt_type not in ["CASH TOP-UP", "FEE", "CUSTODY FEE", "SERVICE FEE"]:
                st.error("Ticker is required for Buy/Sell/Dividend transactions.")
            else:
                final_total_amount = mt_total_amount if mt_total_amount > 0 else mt_quantity * mt_price_per_share
                final_price_per_share = mt_price_per_share
                if mt_type in ["DIVIDEND", "FEE", "CUSTODY FEE", "SERVICE FEE"] and mt_quantity == 0 and final_total_amount > 0:
                    final_price_per_share = np.nan
                new_entry = {
                    'Date': pd.to_datetime(mt_date), 'Ticker': mt_ticker.upper() if mt_ticker else None,
                    'Type': mt_type.upper(), 'Quantity': mt_quantity, 'Price per share': final_price_per_share,
                    'Total Amount': final_total_amount, 'Currency': mt_currency, 'FX Rate': mt_fx_rate
                }
                st.session_state.manual_entries.append(new_entry)
                st.success(f"Added {mt_type} for {mt_ticker or 'N/A'} to temporary list.")

if st.session_state.manual_entries:
    if st.sidebar.button("Clear Manually Added Transactions"):
        st.session_state.manual_entries = []
        st.sidebar.success("Manually added transactions cleared.")
        st.experimental_rerun()

# --- Data Loading and Processing ---
if uploaded_file:
    df_loaded = load_data(uploaded_file)
    if df_loaded is not None:
        st.session_state.transactions_df = df_loaded
        st.session_state.manual_entries = []
        st.success("File uploaded and processed successfully!")
elif st.session_state.manual_entries and st.session_state.transactions_df.empty:
    st.session_state.transactions_df = pd.DataFrame(st.session_state.manual_entries)
    st.session_state.transactions_df = normalize_to_usd(st.session_state.transactions_df)
    if st.session_state.transactions_df is not None:
         st.session_state.transactions_df['Date'] = pd.to_datetime(st.session_state.transactions_df['Date'])
         st.session_state.transactions_df = st.session_state.transactions_df.sort_values(by='Date', ascending=True).reset_index(drop=True)

if not st.session_state.transactions_df.empty and st.session_state.manual_entries:
    manual_df = pd.DataFrame(st.session_state.manual_entries)
    manual_df = normalize_to_usd(manual_df)
    if manual_df is not None:
        manual_df['Date'] = pd.to_datetime(manual_df['Date'])
        st.session_state.transactions_df = pd.concat([st.session_state.transactions_df, manual_df], ignore_index=True)
        st.session_state.transactions_df = st.session_state.transactions_df.sort_values(by='Date', ascending=True).reset_index(drop=True)
        st.session_state.manual_entries = []
        st.info("Manually added transactions have been merged with the uploaded data.")

# --- Main Analysis Area ---
if 'transactions_df' in st.session_state and st.session_state.transactions_df is not None and not st.session_state.transactions_df.empty:
    df_analysis = st.session_state.transactions_df.copy()

    (overall_pl_usd, current_mkt_val_usd, df_holdings, df_completed_transactions,
     live_cnt, investment_q_usd, total_buys_usd, total_sells_usd, total_dividends_usd,
     total_fees_usd, total_realized_pl_sales_usd) = calculate_portfolio_metrics(df_analysis) # total_realized_pl_sales_usd is from completed txns

    total_deposited_usd = calculate_total_deposits(df_analysis)
    total_invested_eur_orig = calculate_total_invested_in_currency(df_analysis, "EUR")

    # Calculate Total Unrealized P/L from current holdings
    total_unrealized_pl_usd = 0
    if not df_holdings.empty and 'Unrealized P/L (USD)' in df_holdings.columns:
        total_unrealized_pl_usd = df_holdings['Unrealized P/L (USD)'].sum()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💼 Current Holdings", "🤝 Completed Transactions", "📋 All Transactions", "📈 Charts"])

    with tab1:
        st.subheader(" Key Financial Metrics")
        st.markdown(f"_(Monetary values displayed in {display_currency})_")

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric(f"Est. Current Portfolio Value ({display_currency})",
                     format_currency(current_mkt_val_usd, display_currency, eur_usd_live_rate),
                     help="Based on live market prices (or $0 for 'Q' stocks, fallback to last txn price).")
        
        mcol2.metric(f"Total Deposited (Cash, {display_currency})",
                     format_currency(total_deposited_usd, display_currency, eur_usd_live_rate))
        
        overall_pl_color = "normal" if overall_pl_usd >=0 else "inverse"
        mcol3.metric(f"Est. Overall Net P/L ({display_currency})",
                     format_currency(overall_pl_usd, display_currency, eur_usd_live_rate),
                     delta_color=overall_pl_color)
        
        st.markdown("---") # Separator

        mcol4, mcol5 = st.columns(2)
        realized_pl_color = "normal" if total_realized_pl_sales_usd >=0 else "inverse"
        mcol4.metric(f"Total Realized P/L (Sales, {display_currency})",
                     format_currency(total_realized_pl_sales_usd, display_currency, eur_usd_live_rate),
                     delta_color=realized_pl_color,
                     help="Sum of profit/loss from all completed sell transactions.")

        unrealized_pl_color = "normal" if total_unrealized_pl_usd >=0 else "inverse"
        mcol5.metric(f"Total Unrealized P/L (Holdings, {display_currency})",
                     format_currency(total_unrealized_pl_usd, display_currency, eur_usd_live_rate),
                     delta_color=unrealized_pl_color,
                     help="Current profit/loss on shares still held, if sold at market price.")


        if total_invested_eur_orig > 0:
             st.metric("Total Invested (Original EUR Buys)", f"€{total_invested_eur_orig:,.2f}",
                       help="Sum of 'Total Amount' for BUY transactions where original currency was EUR, before any FX conversion to USD.")

        if investment_q_usd > 0:
            st.warning(f"⚠️ Total investment (cost of buys) in 'Q' (delisted/bankrupt) stocks: {format_currency(investment_q_usd, display_currency, eur_usd_live_rate)}. These are valued at $0.")

        if live_cnt > 0:
            st.info(f"💡 Successfully fetched live market prices for {live_cnt} ticker(s).")
        else:
            st.warning("⚠️ Could not fetch any live market prices. Portfolio valuation uses last transaction prices or $0 for 'Q' stocks.")

        st.markdown("---")
        st.subheader(f"💰 P&L Calculation Breakdown ({display_currency})")
        st.markdown(f"""
        **Income Components:**
        - Est. Current Holdings Market Value: {format_currency(current_mkt_val_usd, display_currency, eur_usd_live_rate)}
          - _(Cost Basis of Holdings: {format_currency(current_mkt_val_usd - total_unrealized_pl_usd, display_currency, eur_usd_live_rate)})_
          - _(Unrealized P/L on Holdings: {format_currency(total_unrealized_pl_usd, display_currency, eur_usd_live_rate)})_
        - Total Proceeds from All Sells: {format_currency(total_sells_usd, display_currency, eur_usd_live_rate)}
          - _(Cost Basis of Sold Shares: {format_currency(total_sells_usd - total_realized_pl_sales_usd, display_currency, eur_usd_live_rate)})_
          - _(Realized P/L from Sales: {format_currency(total_realized_pl_sales_usd, display_currency, eur_usd_live_rate)})_
        - Total Dividends Received: {format_currency(total_dividends_usd, display_currency, eur_usd_live_rate)}
        ---
        **Subtotal (Income Side):** {format_currency(current_mkt_val_usd + total_sells_usd + total_dividends_usd, display_currency, eur_usd_live_rate)}
        
        **Cost Components:**
        - Total Cost of All Buys: {format_currency(total_buys_usd, display_currency, eur_usd_live_rate)}
        - Total Fees Paid: {format_currency(total_fees_usd, display_currency, eur_usd_live_rate)}
        ---
        **Subtotal (Cost Side):** {format_currency(total_buys_usd + total_fees_usd, display_currency, eur_usd_live_rate)}
        ---
        **Estimated Overall Net P/L:** {format_currency(overall_pl_usd, display_currency, eur_usd_live_rate)}
        
        *Overall Net P/L can also be seen as: Unrealized P/L + Realized P/L (Sales) + Dividends - Fees*
        *({format_currency(total_unrealized_pl_usd, display_currency, eur_usd_live_rate)} + {format_currency(total_realized_pl_sales_usd, display_currency, eur_usd_live_rate)} + {format_currency(total_dividends_usd, display_currency, eur_usd_live_rate)} - {format_currency(total_fees_usd, display_currency, eur_usd_live_rate)})*
        """)

    with tab2:
        st.subheader(" Current Holdings (Unsold Shares)")
        st.markdown(f"_(Monetary values displayed in {display_currency})_")
        if not df_holdings.empty:
            df_holdings_display = df_holdings.copy()
            usd_cols_holdings = ['Avg. Buy Price (USD)', 'Cost Basis (USD)', 'Current Price (USD)', 'Market Value (USD)', 'Unrealized P/L (USD)']
            for col_usd in usd_cols_holdings:
                new_col_name = col_usd.replace(" (USD)", f" ({display_currency})")
                df_holdings_display[new_col_name] = df_holdings_display[col_usd].apply(
                    lambda x: convert_usd_to_display_currency(x, display_currency, eur_usd_live_rate)
                )
            
            display_cols_holdings = ['Ticker', 'Quantity', f'Avg. Buy Price ({display_currency})', 
                                     f'Cost Basis ({display_currency})', f'Current Price ({display_currency})', 
                                     'Price Source', f'Market Value ({display_currency})',
                                     f'Unrealized P/L ({display_currency})', '% Unrealized P/L']
            
            df_holdings_sorted = df_holdings_display.sort_values(by=f"Market Value ({display_currency})", ascending=False).reset_index(drop=True)
            
            st.dataframe(df_holdings_sorted[display_cols_holdings].style.format({
                "Quantity": "{:.4f}",
                f"Avg. Buy Price ({display_currency})": "{:,.2f}",
                f"Cost Basis ({display_currency})": "{:,.2f}",
                f"Current Price ({display_currency})": "{:,.2f}",
                f"Market Value ({display_currency})": "{:,.2f}",
                f"Unrealized P/L ({display_currency})": "{:,.2f}",
                "% Unrealized P/L": "{:.2f}%"
            }).applymap(lambda val: 'color: red' if isinstance(val, (float, int)) and val < 0 and isinstance(val, (float, int)) else ('color: green' if isinstance(val, (float, int)) and val > 0 else ''),
                        subset=[f'Unrealized P/L ({display_currency})', '% Unrealized P/L']))
        else:
            st.info("No current active holdings to display.")

    with tab3:
        st.subheader(" Completed Transactions (Sales)")
        st.markdown(f"_(Monetary values displayed in {display_currency})_")
        if not df_completed_transactions.empty:
            df_completed_display = df_completed_transactions.copy()
            usd_cols_completed = ['Avg. Buy Price (USD)', 'Sell Price p.s. (USD)', 
                                  'Cost Basis Sold (USD)', 'Total Proceeds (USD)', 'Realized P/L (USD)']
            for col_usd in usd_cols_completed:
                new_col_name = col_usd.replace(" (USD)", f" ({display_currency})")
                df_completed_display[new_col_name] = df_completed_display[col_usd].apply(
                    lambda x: convert_usd_to_display_currency(x, display_currency, eur_usd_live_rate)
                )
            display_cols_completed = ['Date', 'Ticker', 'Quantity Sold', f'Avg. Buy Price ({display_currency})',
                                      f'Sell Price p.s. ({display_currency})', f'Cost Basis Sold ({display_currency})',
                                      f'Total Proceeds ({display_currency})', f'Realized P/L ({display_currency})']
            st.dataframe(df_completed_display[display_cols_completed].style.format({
                "Quantity Sold": "{:.4f}",
                f"Avg. Buy Price ({display_currency})": "{:,.2f}",
                f"Sell Price p.s. ({display_currency})": "{:,.2f}",
                f"Cost Basis Sold ({display_currency})": "{:,.2f}",
                f"Total Proceeds ({display_currency})": "{:,.2f}",
                f"Realized P/L ({display_currency})": "{:,.2f}",
                "Date": "{:%Y-%m-%d}"
            }).applymap(lambda val: 'color: red' if isinstance(val, (float, int)) and val < 0 else ('color: green' if isinstance(val, (float, int)) and val > 0 else ''),
                        subset=[f'Realized P/L ({display_currency})']))
        else:
            st.info("No completed (sell) transactions to display.")

    with tab4:
        st.subheader(" Processed Transaction Data")
        st.markdown("View your transactions after processing. 'Amount' and 'Price_per_share_USD' are normalized to USD. Original values are preserved.")
        display_cols_all_tx = ['Date', 'Ticker', 'Type', 'Quantity', 'Price per share',
                               'Original_Currency', 'Original_Amount', 'FX Rate',
                               'Amount', 'Price_per_share_USD']
        display_cols_all_tx_filtered = [col for col in display_cols_all_tx if col in df_analysis.columns]
        # Create a copy for display to avoid modifying df_analysis directly if needed elsewhere
        df_display_all_tx = df_analysis[display_cols_all_tx_filtered].sort_values(by='Date', ascending=False).copy()

        # Convert Date to string, NaT will become 'NaT'
        if 'Date' in df_display_all_tx.columns:
            df_display_all_tx['Date'] = df_display_all_tx['Date'].astype(str) 

        st.dataframe(df_display_all_tx.style.format({
            "Original_Amount": "{:,.2f}", "Amount": "${:,.2f}",
            "Price per share": "{:,.2f}", "Price_per_share_USD": "${:,.2f}",
            "FX Rate": "{:.4f}", "Quantity": "{:.4f}"
            # No specific "Date" formatter here, as it's already string
        }))

    with tab5:
        st.subheader(" Visualizations")
        st.markdown(f"_(Charts display values in USD)_")

        if not df_holdings.empty and 'Market Value (USD)' in df_holdings.columns:
            st.markdown("#### Asset Allocation by Market Value (USD)")
            df_holdings_chart = df_holdings[df_holdings['Market Value (USD)'] > 0].copy()
            if not df_holdings_chart.empty:
                pie_chart = alt.Chart(df_holdings_chart).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Market Value (USD)", type="quantitative", stack=True),
                    color=alt.Color(field="Ticker", type="nominal"),
                    tooltip=['Ticker', alt.Tooltip('Market Value (USD):Q', format='$,.2f')]
                ).properties(title='Current Holdings by Market Value (USD)')
                st.altair_chart(pie_chart, use_container_width=True)
            else:
                st.info("No holdings with positive market value to display in pie chart.")
        
        if not df_completed_transactions.empty and 'Realized P/L (USD)' in df_completed_transactions.columns:
            st.markdown("#### Monthly Realized P/L from Sales (USD)")
            realized_pl_monthly = df_completed_transactions.copy()
            realized_pl_monthly['Month'] = realized_pl_monthly['Date'].dt.to_period('M').astype(str)
            realized_pl_summary = realized_pl_monthly.groupby('Month')['Realized P/L (USD)'].sum().reset_index()
            pl_chart = alt.Chart(realized_pl_summary).mark_bar().encode(
                x=alt.X('Month:O', title='Month', sort=None),
                y=alt.Y('Realized P/L (USD):Q', title='Realized P/L (USD)'),
                color=alt.condition(alt.datum['Realized P/L (USD)'] > 0, alt.value('green'), alt.value('red')),
                tooltip=[alt.Tooltip('Month:O'), alt.Tooltip('Realized P/L (USD):Q', format='$,.2f')]
            ).properties(title='Monthly Realized P/L (USD)')
            st.altair_chart(pl_chart, use_container_width=True)

        CASH_TOP_UP_TYPES = ['CASH TOP-UP', 'DEPOSIT', 'WIRE IN']
        deposits_df = df_analysis[df_analysis['Type'].str.upper().isin(CASH_TOP_UP_TYPES)].copy()
        if not deposits_df.empty:
            st.markdown("#### Monthly Cash Deposits (USD)")
            deposits_df['Month'] = deposits_df['Date'].dt.to_period('M').astype(str)
            deposits_over_time = deposits_df.groupby('Month')['Amount'].sum().reset_index()
            chart_deposits = alt.Chart(deposits_over_time).mark_bar().encode(
                x=alt.X('Month:O', title='Month', sort=None),
                y=alt.Y('Amount:Q', title='Total Deposits (USD)'),
                tooltip=[alt.Tooltip('Month:O'), alt.Tooltip('Amount:Q', title='Deposits (USD)', format='$,.2f')]
            ).properties(title='Monthly Cash Deposits (USD)')
            st.altair_chart(chart_deposits, use_container_width=True)

        BUY_SELL_TYPES = ['BUY - MARKET', 'BUY', 'SELL - MARKET', 'SELL']
        buys_sells_df = df_analysis[df_analysis['Type'].isin(BUY_SELL_TYPES)].copy()
        if not buys_sells_df.empty:
            st.markdown("#### Monthly Buy vs. Sell Activity (Value in USD)")
            buys_sells_df['Month'] = buys_sells_df['Date'].dt.to_period('M').astype(str)
            buys_sells_df['Tx_Category'] = buys_sells_df['Type'].apply(lambda x: 'Buy' if 'BUY' in x else ('Sell' if 'SELL' in x else 'Other'))
            buys_sells_over_time = buys_sells_df.groupby(['Month', 'Tx_Category'])['Amount'].sum().unstack(fill_value=0).reset_index()
            if 'Buy' not in buys_sells_over_time.columns: buys_sells_over_time['Buy'] = 0
            if 'Sell' not in buys_sells_over_time.columns: buys_sells_over_time['Sell'] = 0
            buys_sells_over_time_melted = buys_sells_over_time.melt(id_vars=['Month'], value_vars=['Buy', 'Sell'], var_name='Transaction Category', value_name='Total Amount (USD)')
            chart_buys_sells = alt.Chart(buys_sells_over_time_melted).mark_line(point=True).encode(
                x=alt.X('Month:O', title='Month', sort=None),
                y=alt.Y('Total Amount (USD):Q', title='Total Transaction Amount (USD)'),
                color='Transaction Category:N',
                tooltip=[alt.Tooltip('Month:O'), 'Transaction Category', alt.Tooltip('Total Amount (USD):Q', format='$,.2f')]
            ).properties(title='Monthly Buy vs. Sell Activity (Value in USD)')
            st.altair_chart(chart_buys_sells, use_container_width=True)

        DIVIDEND_TYPES = ['DIVIDEND', 'DIVIDEND INCOME']
        dividends_df = df_analysis[df_analysis['Type'].isin(DIVIDEND_TYPES)].copy()
        if not dividends_df.empty:
            st.markdown("#### Monthly Dividend Income (USD)")
            dividends_df['Month'] = dividends_df['Date'].dt.to_period('M').astype(str)
            dividends_over_time = dividends_df.groupby('Month')['Amount'].sum().reset_index()
            chart_dividends = alt.Chart(dividends_over_time).mark_bar().encode(
                x=alt.X('Month:O', title='Month', sort=None),
                y=alt.Y('Amount:Q', title='Total Dividends (USD)'),
                tooltip=[alt.Tooltip('Month:O'), alt.Tooltip('Amount:Q', title='Dividends (USD)', format='$,.2f')]
            ).properties(title='Monthly Dividend Income (USD)')
            st.altair_chart(chart_dividends, use_container_width=True)

elif not uploaded_file and not st.session_state.manual_entries:
    st.info("👈 Upload your brokerage transaction file or add transactions manually to begin analysis.")
elif st.session_state.transactions_df is not None and st.session_state.transactions_df.empty:
     st.warning("No processable transaction data found. Please upload a file with valid transactions or add them manually.")
elif st.session_state.transactions_df is None:
        st.error("Failed to load or process transaction data. Check error messages above or ensure file format and essential columns are correct.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Disclaimers & Notes:**
* **Average Cost Basis:** Used for P&L on sales and current holdings.
* **Data Accuracy:** Assumes chronological 'Date' and correct transaction details.
* **Live Prices:** From Yahoo Finance (delayed). 'Q' tickers valued at $0. Fallbacks used if live price fails.
* **Currency:** Core calculations in USD. Display currency conversion uses live EUR/USD rate. 'Total Invested in EUR' (etc.) refers to original buy amounts in that currency.
* **Not Financial Advice:** For informational purposes only.
""")
st.markdown("---")
st.caption(f"Brokerage Account Analyzer Pro | Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")