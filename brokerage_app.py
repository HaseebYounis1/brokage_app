import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date as date_type

st.set_page_config(layout="wide", page_title="Brokerage Account Analyzer Pro")

# ─── Session State ───
for key, default in [
    ('transactions_df', pd.DataFrame()),
    ('manual_entries', []),
    ('eur_usd_rate', None),
    ('last_fx_fetch_time', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Transaction Type Constants ───
BUY_TYPES          = ['BUY - MARKET', 'BUY']
SELL_TYPES         = ['SELL - MARKET', 'SELL', 'SELL - LIMIT', 'SELL - STOP']
POSITION_CLOSURE   = ['POSITION CLOSURE']
FEE_TYPES          = ['CUSTODY FEE', 'FEE', 'SERVICE FEE', 'COMMISSION']
DIVIDEND_TYPES     = ['DIVIDEND', 'DIVIDEND INCOME']
DIVIDEND_TAX_TYPES = ['DIVIDEND TAX (CORRECTION)']
CASH_TOP_UP_TYPES  = ['CASH TOP-UP', 'DEPOSIT', 'WIRE IN']
MERGER_TYPES       = ['MERGER - STOCK']
STOCK_SPLIT_TYPES  = ['STOCK SPLIT']
ALL_SELL_LIKE      = SELL_TYPES + POSITION_CLOSURE

# Irish CGT constants (Revenue.ie)
IRISH_CGT_RATE      = 0.33      # 33%
IRISH_CGT_EXEMPTION = 1270.0    # €1,270 annual personal exemption


# ─── Helpers ───
def clean_monetary_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        for sym in ['$', '€', 'â\x82¬', 'USD', 'EUR', ' ', ',']:
            value = value.replace(sym, '')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan


def fetch_eur_usd_rate():
    now = datetime.now()
    if (st.session_state.eur_usd_rate and st.session_state.last_fx_fetch_time and
            (now - st.session_state.last_fx_fetch_time < timedelta(hours=1))):
        return st.session_state.eur_usd_rate
    try:
        hist = yf.Ticker("EURUSD=X").history(period="2d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            st.session_state.eur_usd_rate = rate
            st.session_state.last_fx_fetch_time = now
            return rate
    except Exception:
        pass
    st.warning("Could not fetch live EUR/USD rate. EUR conversions may be approximate.", icon="⚠️")
    return None


def usd_to_display(amount_usd, currency, eur_usd_rate):
    if currency == "EUR" and eur_usd_rate:
        return amount_usd / eur_usd_rate
    return amount_usd


def fmt(amount, currency, eur_usd_rate):
    val = usd_to_display(amount, currency, eur_usd_rate)
    sym = "€" if currency == "EUR" else "$"
    return f"{sym}{val:,.2f}"


def parse_dates_robust(series):
    """Parse date strings, supporting both pandas 1.x and 2.x."""
    try:
        return pd.to_datetime(series, format='mixed', errors='coerce')
    except TypeError:
        return pd.to_datetime(series, errors='coerce')


def normalize_to_usd(df):
    df = df.copy()
    if not all(c in df.columns for c in ['Total Amount', 'Currency']):
        st.error("Critical: 'Total Amount' or 'Currency' column missing.")
        return None

    df['Original_Amount'] = df['Total Amount'].apply(clean_monetary_value)
    df['Original_Currency'] = df['Currency'].astype(str).str.upper().str.strip()

    df['Price_per_share_Cleaned'] = np.nan
    if 'Price per share' in df.columns:
        df['Price_per_share_Cleaned'] = df['Price per share'].apply(clean_monetary_value)

    df['Amount_USD'] = df['Original_Amount'].copy()
    df['Price_per_share_USD'] = df['Price_per_share_Cleaned'].copy()

    if 'FX Rate' in df.columns:
        df['FX Rate'] = pd.to_numeric(df['FX Rate'], errors='coerce')
        mask = (df['Original_Currency'] != 'USD') & df['FX Rate'].notna() & (df['FX Rate'] != 0)
        df.loc[mask, 'Amount_USD'] = df.loc[mask, 'Original_Amount'] * df.loc[mask, 'FX Rate']
        df.loc[mask, 'Price_per_share_USD'] = df.loc[mask, 'Price_per_share_Cleaned'] * df.loc[mask, 'FX Rate']
        df['FX Rate'] = df['FX Rate'].fillna(1.0)
    else:
        st.warning("'FX Rate' column not found. Only USD amounts processed correctly.")
        df['FX Rate'] = 1.0

    # EUR equivalent: USD / FX_rate (EUR/USD) gives the EUR value at time of transaction
    safe_fx = df['FX Rate'].replace(0, np.nan)
    df['EUR_Equivalent'] = np.where(
        df['Original_Currency'] == 'EUR',
        df['Original_Amount'],
        df['Amount_USD'] / safe_fx
    )
    df.rename(columns={'Amount_USD': 'Amount'}, inplace=True)
    return df


def load_data(uploaded_file):
    try:
        name = uploaded_file.name
        if name.endswith('.csv'):
            peek = uploaded_file.read(1024).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            delim = ';' if ';' in peek.splitlines()[0] else ','
            df = pd.read_csv(uploaded_file, delimiter=delim)
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file type. Use .csv, .xlsx, or .xls")
            return None

        if 'Date' not in df.columns:
            st.error("Critical: 'Date' column missing.")
            return None

        orig_dates = df['Date'].astype(str).copy()
        df['Date'] = parse_dates_robust(orig_dates)
        if df['Date'].isnull().any():
            n = df['Date'].isnull().sum()
            examples = orig_dates[df['Date'].isnull()].unique()[:3]
            st.error(f"Could not parse {n} date(s). Examples: {', '.join(examples)}")
            return None

        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        df.rename(columns={"Total Amo": "Total Amount", "Price per s": "Price per share"}, inplace=True)

        if 'Type' not in df.columns:
            st.error("Critical: 'Type' column missing.")
            return None
        df['Type'] = df['Type'].astype(str).str.strip().str.upper()

        df = normalize_to_usd(df)
        if df is None:
            return None
        df.dropna(subset=['Amount', 'Type'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def get_live_price(ticker_symbol, fallback_price_usd):
    if pd.isna(ticker_symbol) or not isinstance(ticker_symbol, str):
        return 0.0, "Invalid"
    if ticker_symbol.endswith('Q') or '.OLD' in ticker_symbol:
        return 0.0, "Delisted ($0)"
    try:
        hist = yf.Ticker(ticker_symbol).history(period="5d")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            age = (datetime.now() - hist.index[-1].to_pydatetime().replace(tzinfo=None)).days
            return price, "Live" if age <= 7 else "Stale"
        return fallback_price_usd, "No Data"
    except Exception:
        return fallback_price_usd, "API Error"


def calculate_portfolio_metrics(df):
    empty = dict(
        overall_pl=0, portfolio_value=0, holdings=pd.DataFrame(),
        completed=pd.DataFrame(), live_count=0, q_investment=0,
        total_buys=0, total_sells=0, total_dividends=0, total_fees=0,
        total_realized_pl=0, total_div_tax=0, cash_balance=0,
        total_deposited_usd=0, total_deposited_eur=0, buy_turnover_eur=0,
        unrealized_pl=0
    )
    if df is None or df.empty:
        return empty

    df = df.copy()
    df['Quantity'] = pd.to_numeric(df.get('Quantity', 0), errors='coerce').fillna(0)

    total_buys      = df[df['Type'].isin(BUY_TYPES)]['Amount'].sum()
    total_sells     = df[df['Type'].isin(ALL_SELL_LIKE)]['Amount'].sum()
    total_dividends = df[df['Type'].isin(DIVIDEND_TYPES)]['Amount'].sum()
    total_fees      = df[df['Type'].isin(FEE_TYPES)]['Amount'].sum()
    total_div_tax   = df[df['Type'].isin(DIVIDEND_TAX_TYPES)]['Amount'].sum()

    total_deposited_usd = df[df['Type'].isin(CASH_TOP_UP_TYPES)]['Amount'].sum()
    total_deposited_eur = df[df['Type'].isin(CASH_TOP_UP_TYPES)]['EUR_Equivalent'].sum() \
        if 'EUR_Equivalent' in df.columns else 0

    # Buy turnover in EUR: total of all BUY transactions' EUR equivalent
    # This INCLUDES reinvestments — it is NOT "money from pocket"
    buy_turnover_eur = df[df['Type'].isin(BUY_TYPES)]['EUR_Equivalent'].sum() \
        if 'EUR_Equivalent' in df.columns else 0

    q_investment = df[
        df['Type'].isin(BUY_TYPES) & df['Ticker'].astype(str).str.endswith('Q')
    ]['Amount'].sum()

    last_prices = {}
    if 'Price_per_share_USD' in df.columns and 'Ticker' in df.columns:
        last_prices = (
            df.dropna(subset=['Price_per_share_USD', 'Ticker'])
            .groupby('Ticker')['Price_per_share_USD'].last().to_dict()
        )

    holdings = {}  # ticker -> {quantity, cost_basis, avg_price, first_buy_date}
    completed_rows = []

    for _, row in df.iterrows():
        ticker  = str(row.get('Ticker', '')) if not pd.isna(row.get('Ticker')) else None
        tx_type = row.get('Type', '')
        qty     = float(row.get('Quantity', 0))
        amount  = float(row.get('Amount', 0))
        price_ps = row.get('Price_per_share_USD', np.nan)
        date    = row.get('Date')
        fx_rate = float(row.get('FX Rate', 1.0))

        # ── BUY ──
        if tx_type in BUY_TYPES:
            if not ticker:
                continue
            h = holdings.setdefault(ticker, {'q': 0, 'cb': 0.0, 'avg': 0.0, 'first_buy': date})
            h['q'] += qty
            h['cb'] += amount
            h['avg'] = h['cb'] / h['q'] if h['q'] > 1e-9 else 0.0

        # ── SELL / POSITION CLOSURE ──
        elif tx_type in ALL_SELL_LIKE:
            if not ticker or ticker not in holdings or holdings[ticker]['q'] < 1e-9:
                continue
            h = holdings[ticker]
            sold = min(qty if qty > 0 else h['q'], h['q'])
            cogs = sold * h['avg']
            proceeds = amount
            if (pd.isna(price_ps) or price_ps == 0) and sold > 0:
                price_ps = proceeds / sold
            realized_pl = proceeds - cogs
            eur_pl = realized_pl / fx_rate if fx_rate else realized_pl
            days_held = (date - h['first_buy']).days if (
                h.get('first_buy') and not pd.isna(date) and not pd.isna(h['first_buy'])
            ) else 0
            completed_rows.append({
                'Date': date, 'Ticker': ticker, 'Type': tx_type,
                'Quantity Sold': sold,
                'Avg. Buy Price (USD)': h['avg'],
                'Sell Price p.s. (USD)': price_ps,
                'Cost Basis Sold (USD)': cogs,
                'Total Proceeds (USD)': proceeds,
                'Realized P/L (USD)': realized_pl,
                'Realized P/L (EUR)': eur_pl,
                'Days Held': days_held,
                'FX Rate at Sale': fx_rate,
            })
            h['q'] -= sold
            h['cb'] -= cogs
            if h['q'] <= 1e-5:
                del holdings[ticker]
            else:
                h['avg'] = h['cb'] / h['q']

        # ── STOCK SPLIT: adjust qty, keep cost basis ──
        elif tx_type in STOCK_SPLIT_TYPES:
            if not ticker or ticker not in holdings:
                continue
            holdings[ticker]['q'] += qty
            if holdings[ticker]['q'] > 1e-9:
                holdings[ticker]['avg'] = holdings[ticker]['cb'] / holdings[ticker]['q']
            else:
                del holdings[ticker]

        # ── MERGER: positive = receive new shares at $0; negative = surrender old at $0 ──
        elif tx_type in MERGER_TYPES:
            if not ticker:
                continue
            if qty > 0:
                h = holdings.setdefault(ticker, {'q': 0, 'cb': 0.0, 'avg': 0.0, 'first_buy': date})
                h['q'] += qty
                h['avg'] = h['cb'] / h['q'] if h['q'] > 1e-9 else 0.0
            elif qty < 0 and ticker in holdings:
                h = holdings[ticker]
                surrendered = min(abs(qty), h['q'])
                cogs = surrendered * h['avg']
                days_held = (date - h['first_buy']).days if (
                    h.get('first_buy') and not pd.isna(date)
                ) else 0
                completed_rows.append({
                    'Date': date, 'Ticker': ticker, 'Type': 'MERGER (surrendered)',
                    'Quantity Sold': surrendered,
                    'Avg. Buy Price (USD)': h['avg'],
                    'Sell Price p.s. (USD)': 0.0,
                    'Cost Basis Sold (USD)': cogs,
                    'Total Proceeds (USD)': 0.0,
                    'Realized P/L (USD)': -cogs,
                    'Realized P/L (EUR)': -cogs / fx_rate if fx_rate else -cogs,
                    'Days Held': days_held,
                    'FX Rate at Sale': fx_rate,
                })
                h['q'] -= surrendered
                h['cb'] -= cogs
                if h['q'] <= 1e-5:
                    del holdings[ticker]
                else:
                    h['avg'] = h['cb'] / h['q']

    # ── Build live holdings table ──
    portfolio_value = 0.0
    live_count = 0
    holdings_rows = []

    for ticker, h in holdings.items():
        if h['q'] < 1e-5:
            continue
        fallback = last_prices.get(ticker, h['avg'])
        live_price, source = get_live_price(ticker, fallback)
        if source == "Live":
            live_count += 1
        mkt_val   = h['q'] * live_price
        portfolio_value += mkt_val
        unreal_pl = mkt_val - h['cb']
        pct       = (unreal_pl / h['cb'] * 100) if h['cb'] != 0 else 0.0
        holdings_rows.append({
            'Ticker': ticker,
            'Quantity': h['q'],
            'Avg. Buy Price (USD)': h['avg'],
            'Cost Basis (USD)': h['cb'],
            'Current Price (USD)': live_price,
            'Price Source': source,
            'Market Value (USD)': mkt_val,
            'Unrealized P/L (USD)': unreal_pl,
            '% Unrealized P/L': pct,
        })

    holdings_df  = pd.DataFrame(holdings_rows)
    completed_df = pd.DataFrame(completed_rows)
    if not completed_df.empty:
        completed_df = completed_df.sort_values('Date', ascending=False).reset_index(drop=True)

    total_realized_pl = completed_df['Realized P/L (USD)'].sum() if not completed_df.empty else 0.0
    unrealized_pl     = holdings_df['Unrealized P/L (USD)'].sum() if not holdings_df.empty else 0.0

    # Fees stored as negative values → add them directly (they already reduce the total)
    overall_pl   = (portfolio_value + total_sells + total_dividends + total_div_tax) - total_buys + total_fees
    cash_balance = total_deposited_usd - total_buys + total_sells + total_dividends + total_div_tax + total_fees

    return dict(
        overall_pl=overall_pl, portfolio_value=portfolio_value,
        holdings=holdings_df, completed=completed_df,
        live_count=live_count, q_investment=q_investment,
        total_buys=total_buys, total_sells=total_sells,
        total_dividends=total_dividends, total_fees=total_fees,
        total_realized_pl=total_realized_pl, total_div_tax=total_div_tax,
        cash_balance=cash_balance,
        total_deposited_usd=total_deposited_usd, total_deposited_eur=total_deposited_eur,
        buy_turnover_eur=buy_turnover_eur, unrealized_pl=unrealized_pl,
    )


def calculate_irish_cgt(completed_df, holdings_df, live_eur_usd_rate):
    """
    Computes Irish CGT per calendar year using realized P/L (EUR).
    Applies €1,270 annual exemption and carries forward unused losses.
    Returns a list of dicts (one per year) and total CGT owed.
    """
    if completed_df.empty or 'Realized P/L (EUR)' not in completed_df.columns:
        return [], 0.0

    df = completed_df.copy()
    df = df[df['Type'].isin(SELL_TYPES + POSITION_CLOSURE + ['MERGER (surrendered)'])]
    if df.empty:
        return [], 0.0

    df['Year'] = pd.to_datetime(df['Date']).dt.year
    years = sorted(df['Year'].dropna().unique().astype(int).tolist())

    rows = []
    carry_forward = 0.0

    for year in years:
        g = df[df['Year'] == year]
        gains  = g[g['Realized P/L (EUR)'] > 0]['Realized P/L (EUR)'].sum()
        losses = g[g['Realized P/L (EUR)'] < 0]['Realized P/L (EUR)'].sum()
        net    = g['Realized P/L (EUR)'].sum()

        adj = net - carry_forward  # deduct any carried-forward losses
        if adj > 0:
            taxable   = max(0.0, adj - IRISH_CGT_EXEMPTION)
            cgt       = taxable * IRISH_CGT_RATE
            carry_out = 0.0
        else:
            taxable   = 0.0
            cgt       = 0.0
            carry_out = abs(adj)

        # Filing / payment status
        current_year = datetime.now().year
        current_month = datetime.now().month
        if year < current_year - 1:
            status = "✅ Past — should be filed"
        elif year == current_year - 1:
            if current_month >= 11:
                status = "✅ Filed (deadline passed)"
            else:
                status = "⚠️ Due — file by 31 Oct / 15 Nov (ROS)"
        elif year == current_year:
            status = "📅 Current year (pay Dec 15 for Jan–Nov gains)"
        else:
            status = "🔮 Future"

        rows.append({
            'Year': year,
            'Gains (€)': gains,
            'Losses (€)': losses,
            'Net P/L (€)': net,
            'Loss Relief Used (€)': min(carry_forward, max(net, 0)),
            'Adj. Net (€)': adj,
            'Annual Exemption (€)': IRISH_CGT_EXEMPTION if adj > IRISH_CGT_EXEMPTION else max(0, adj),
            'Taxable Gain (€)': taxable,
            'CGT @ 33% (€)': cgt,
            'Losses Carried Fwd (€)': carry_out,
            'Status': status,
        })
        carry_forward = carry_out

    total_cgt = sum(r['CGT @ 33% (€)'] for r in rows)

    # Unrealized CGT: if all holdings sold today
    unrealized_eur_pl = 0.0
    if not holdings_df.empty and 'Unrealized P/L (USD)' in holdings_df.columns and live_eur_usd_rate:
        unrealized_eur_pl = holdings_df[holdings_df['Unrealized P/L (USD)'] > 0]['Unrealized P/L (USD)'].sum() / live_eur_usd_rate
    potential_tax = max(0.0, unrealized_eur_pl - IRISH_CGT_EXEMPTION) * IRISH_CGT_RATE

    return rows, total_cgt, carry_forward, unrealized_eur_pl, potential_tax


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
st.title("📈 Brokerage Account Analyzer Pro")
st.sidebar.header("⚙️ Settings & Data Input")

display_currency = st.sidebar.selectbox("Display currency:", ["USD", "EUR"], key="display_currency_select")
eur_rate = fetch_eur_usd_rate()
if display_currency == "EUR" and eur_rate:
    st.sidebar.caption(f"EUR/USD rate: {eur_rate:.4f} (live)")

uploaded_file = st.sidebar.file_uploader("Upload Transactions (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])

# Manual entry form
st.sidebar.subheader("✍️ Manual Transaction Entry")
with st.sidebar.expander("Add a Transaction"):
    with st.form("manual_tx_form", clear_on_submit=True):
        mt_date     = st.date_input("Date", value=datetime.now())
        mt_ticker   = st.text_input("Ticker")
        mt_type     = st.selectbox("Type", ["BUY - MARKET", "SELL - MARKET", "SELL - LIMIT",
                                            "SELL - STOP", "DIVIDEND", "CASH TOP-UP", "FEE",
                                            "CUSTODY FEE", "SERVICE FEE"])
        mt_qty      = st.number_input("Quantity", min_value=0.0, step=0.0001, format="%.4f")
        mt_price    = st.number_input("Price per Share", min_value=0.0, format="%.4f")
        mt_amount   = st.number_input("Total Amount (overrides Price×Qty)", min_value=0.0, format="%.2f")
        mt_currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD"])
        mt_fx       = st.number_input("FX Rate (to USD)", value=1.0, min_value=0.0, format="%.6f")
        if st.form_submit_button("Add Transaction"):
            if not mt_ticker and mt_type not in ["CASH TOP-UP", "FEE", "CUSTODY FEE", "SERVICE FEE"]:
                st.error("Ticker required for Buy/Sell/Dividend.")
            else:
                final_amount = mt_amount if mt_amount > 0 else mt_qty * mt_price
                st.session_state.manual_entries.append({
                    'Date': pd.to_datetime(mt_date), 'Ticker': mt_ticker.upper() if mt_ticker else None,
                    'Type': mt_type.upper(), 'Quantity': mt_qty, 'Price per share': mt_price,
                    'Total Amount': final_amount, 'Currency': mt_currency, 'FX Rate': mt_fx
                })
                st.success(f"Added {mt_type} for {mt_ticker or 'N/A'}.")

if st.session_state.manual_entries:
    if st.sidebar.button("Clear Manual Transactions"):
        st.session_state.manual_entries = []
        st.rerun()

# ─── Data Loading ───
if uploaded_file:
    df_loaded = load_data(uploaded_file)
    if df_loaded is not None:
        st.session_state.transactions_df = df_loaded
        st.session_state.manual_entries = []
        st.success("File uploaded and processed successfully!")
elif st.session_state.manual_entries and st.session_state.transactions_df.empty:
    tmp = normalize_to_usd(pd.DataFrame(st.session_state.manual_entries))
    if tmp is not None:
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        st.session_state.transactions_df = tmp.sort_values('Date').reset_index(drop=True)

if not st.session_state.transactions_df.empty and st.session_state.manual_entries:
    manual_df = normalize_to_usd(pd.DataFrame(st.session_state.manual_entries))
    if manual_df is not None:
        manual_df['Date'] = pd.to_datetime(manual_df['Date'])
        st.session_state.transactions_df = pd.concat(
            [st.session_state.transactions_df, manual_df], ignore_index=True
        ).sort_values('Date').reset_index(drop=True)
        st.session_state.manual_entries = []
        st.info("Manual transactions merged with uploaded data.")


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if (st.session_state.transactions_df is not None and
        not st.session_state.transactions_df.empty):

    df_analysis = st.session_state.transactions_df.copy()
    m = calculate_portfolio_metrics(df_analysis)

    overall_pl        = m['overall_pl']
    portfolio_value   = m['portfolio_value']
    df_holdings       = m['holdings']
    df_completed      = m['completed']
    live_count        = m['live_count']
    q_invest          = m['q_investment']
    total_buys        = m['total_buys']
    total_sells       = m['total_sells']
    total_dividends   = m['total_dividends']
    total_fees        = m['total_fees']
    total_realized_pl = m['total_realized_pl']
    total_div_tax     = m['total_div_tax']
    cash_balance      = m['cash_balance']
    dep_usd           = m['total_deposited_usd']
    dep_eur           = m['total_deposited_eur']
    buy_turnover_eur  = m['buy_turnover_eur']
    unrealized_pl     = m['unrealized_pl']

    total_account_value = portfolio_value + max(cash_balance, 0)
    roi_pct = ((total_account_value - dep_usd) / dep_usd * 100) if dep_usd > 0 else 0.0

    # Account age
    if 'Date' in df_analysis.columns:
        first_date = df_analysis['Date'].min()
        last_date  = df_analysis['Date'].max()
        days_active = (last_date - first_date).days
        years_active = days_active / 365.25
        annualized_roi = ((total_account_value / dep_usd) ** (1 / years_active) - 1) * 100 \
            if dep_usd > 0 and years_active > 0 else 0.0
    else:
        days_active = annualized_roi = 0

    # Sell trade stats
    if not df_completed.empty:
        sells_only = df_completed[df_completed['Type'].isin(SELL_TYPES + POSITION_CLOSURE)]
        wins   = (sells_only['Realized P/L (USD)'] > 0).sum()
        losses = (sells_only['Realized P/L (USD)'] < 0).sum()
        total_trades = len(sells_only)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_win  = sells_only[sells_only['Realized P/L (USD)'] > 0]['Realized P/L (USD)'].mean() or 0.0
        avg_loss = sells_only[sells_only['Realized P/L (USD)'] < 0]['Realized P/L (USD)'].mean() or 0.0
        profit_factor = abs(
            sells_only[sells_only['Realized P/L (USD)'] > 0]['Realized P/L (USD)'].sum() /
            sells_only[sells_only['Realized P/L (USD)'] < 0]['Realized P/L (USD)'].sum()
        ) if losses > 0 else float('inf')
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        avg_hold_days = int(sells_only[sells_only['Days Held'] > 0]['Days Held'].mean() or 0)
    else:
        wins = losses = total_trades = avg_hold_days = 0
        win_rate = avg_win = avg_loss = profit_factor = expectancy = 0.0

    # Irish CGT
    cgt_result = calculate_irish_cgt(df_completed, df_holdings, eur_rate)
    cgt_rows, total_cgt, cgt_carry_fwd, unreal_eur_gains, potential_cgt = cgt_result

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "💼 Holdings", "🤝 Completed Trades",
        "📋 Transactions", "📈 Charts", "🧾 Tax (Ireland)"
    ])

    # ─── TAB 1: OVERVIEW ───
    with tab1:
        st.subheader("Key Financial Metrics")
        st.caption(f"Monetary values in {display_currency}. Live prices from Yahoo Finance.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Market Value", fmt(portfolio_value, display_currency, eur_rate))
        c2.metric("Cash Balance (est.)", fmt(max(cash_balance, 0), display_currency, eur_rate),
                  help="Deposits − Buys + Sells + Dividends − Fees")
        c3.metric("Total Account Value", fmt(total_account_value, display_currency, eur_rate))
        c4.metric("Overall ROI", f"{roi_pct:+.2f}%",
                  delta=f"{roi_pct:+.2f}%",
                  delta_color="normal" if roi_pct >= 0 else "inverse")

        st.markdown("---")
        c5, c6, c7 = st.columns(3)
        c5.metric("Overall Net P/L", fmt(overall_pl, display_currency, eur_rate),
                  delta=fmt(overall_pl, display_currency, eur_rate),
                  delta_color="normal" if overall_pl >= 0 else "inverse")
        c6.metric("Realized P/L (Sales)", fmt(total_realized_pl, display_currency, eur_rate),
                  delta=fmt(total_realized_pl, display_currency, eur_rate),
                  delta_color="normal" if total_realized_pl >= 0 else "inverse")
        c7.metric("Unrealized P/L (Holdings)", fmt(unrealized_pl, display_currency, eur_rate),
                  delta=fmt(unrealized_pl, display_currency, eur_rate),
                  delta_color="normal" if unrealized_pl >= 0 else "inverse")

        st.markdown("---")
        st.subheader("💶 Money In & Out")
        c8, c9, c10 = st.columns(3)
        c8.metric("Cash Deposited (USD recorded)", f"${dep_usd:,.2f}",
                  help="Sum of all CASH TOP-UP amounts as recorded by Revolut in USD.")
        c9.metric("Cash Deposited (EUR equiv.)", f"€{dep_eur:,.2f}",
                  help="EUR value of each deposit = USD amount ÷ EUR/USD rate at time of deposit. "
                       "This is the actual EUR you transferred from your bank.")
        c10.metric("Annualized Return", f"{annualized_roi:+.1f}%",
                   help=f"Based on {days_active} days of account history.")

        st.info(
            f"ℹ️ **Why does 'Total Buy Turnover' show €{buy_turnover_eur:,.0f}?**\n\n"
            f"Your actual out-of-pocket deposits were **€{dep_eur:,.2f}**. "
            f"The €{buy_turnover_eur:,.0f} figure is the EUR equivalent of ALL buy transactions, "
            f"including stocks purchased with proceeds from previous sells. "
            f"Because you've traded frequently — selling one stock then reinvesting those proceeds "
            f"into another — the buy turnover compounds much higher than the original capital."
        )

        st.markdown("---")
        st.subheader("🔢 Trade Statistics")
        c11, c12, c13, c14 = st.columns(4)
        c11.metric("Total Closed Trades", total_trades)
        c12.metric("Win Rate", f"{win_rate:.1f}%", help=f"{wins} wins / {losses} losses")
        c13.metric("Profit Factor", f"{profit_factor:.2f}",
                   help="Total gains ÷ total losses. >1 = profitable system.")
        c14.metric("Avg Holding Period", f"{avg_hold_days} days")

        c15, c16, c17, c18 = st.columns(4)
        c15.metric("Avg Win", fmt(avg_win, display_currency, eur_rate))
        c16.metric("Avg Loss", fmt(avg_loss, display_currency, eur_rate))
        c17.metric("Trade Expectancy", fmt(expectancy, display_currency, eur_rate),
                   help="(Win rate × avg win) + (loss rate × avg loss). Positive = edge.")
        c18.metric("Dividends Received", fmt(total_dividends, display_currency, eur_rate))

        st.markdown("---")
        st.subheader(f"P&L Breakdown ({display_currency})")
        cost_basis_held = portfolio_value - unrealized_pl
        cost_basis_sold = total_sells - total_realized_pl
        st.markdown(f"""
**Income side:**
- Portfolio market value: {fmt(portfolio_value, display_currency, eur_rate)}
  - Cost basis of holdings: {fmt(cost_basis_held, display_currency, eur_rate)} | Unrealized P/L: **{fmt(unrealized_pl, display_currency, eur_rate)}**
- Total sell proceeds: {fmt(total_sells, display_currency, eur_rate)}
  - Cost of sold shares: {fmt(cost_basis_sold, display_currency, eur_rate)} | Realized P/L: **{fmt(total_realized_pl, display_currency, eur_rate)}**
- Dividends: {fmt(total_dividends, display_currency, eur_rate)} | Dividend tax adjustments: {fmt(total_div_tax, display_currency, eur_rate)}

**Cost side:**
- Total cost of all buys: {fmt(total_buys, display_currency, eur_rate)}
- Total fees paid: {fmt(abs(total_fees), display_currency, eur_rate)}

---
**Overall Net P/L: {fmt(overall_pl, display_currency, eur_rate)}**
*(= Unrealized {fmt(unrealized_pl, display_currency, eur_rate)} + Realized {fmt(total_realized_pl, display_currency, eur_rate)} + Dividends {fmt(total_dividends, display_currency, eur_rate)} − Fees {fmt(abs(total_fees), display_currency, eur_rate)})*
        """)

        if q_invest > 0:
            st.warning(f"⚠️ Cost of delisted ('Q') stocks: {fmt(q_invest, display_currency, eur_rate)} — valued at $0.")
        if live_count > 0:
            st.info(f"💡 Live prices fetched for {live_count} ticker(s).")
        else:
            st.warning("⚠️ No live prices. Using last transaction prices.")

    # ─── TAB 2: HOLDINGS ───
    with tab2:
        st.subheader("Current Holdings")
        st.caption(f"Values in {display_currency}.")
        if not df_holdings.empty:
            disp = df_holdings.copy()
            for col in ['Avg. Buy Price (USD)', 'Cost Basis (USD)', 'Current Price (USD)',
                        'Market Value (USD)', 'Unrealized P/L (USD)']:
                new = col.replace('(USD)', f'({display_currency})')
                disp[new] = disp[col].apply(lambda x: usd_to_display(x, display_currency, eur_rate))

            show = ['Ticker', 'Quantity',
                    f'Avg. Buy Price ({display_currency})', f'Cost Basis ({display_currency})',
                    f'Current Price ({display_currency})', 'Price Source',
                    f'Market Value ({display_currency})', f'Unrealized P/L ({display_currency})',
                    '% Unrealized P/L']

            disp = disp.sort_values(f'Market Value ({display_currency})', ascending=False).reset_index(drop=True)
            pl_col = f'Unrealized P/L ({display_currency})'
            pct_col = '% Unrealized P/L'

            styled = disp[show].style.format({
                'Quantity': '{:.4f}',
                f'Avg. Buy Price ({display_currency})': '{:,.2f}',
                f'Cost Basis ({display_currency})': '{:,.2f}',
                f'Current Price ({display_currency})': '{:,.2f}',
                f'Market Value ({display_currency})': '{:,.2f}',
                pl_col: '{:,.2f}',
                pct_col: '{:.2f}%',
            }).map(lambda v: 'color: red' if isinstance(v, (int, float)) and v < 0
                  else ('color: green' if isinstance(v, (int, float)) and v > 0 else ''),
                  subset=[pl_col, pct_col])
            st.dataframe(styled, use_container_width=True)

            total_mv  = disp[f'Market Value ({display_currency})'].sum()
            total_cb  = disp[f'Cost Basis ({display_currency})'].sum()
            total_upl = disp[f'Unrealized P/L ({display_currency})'].sum()
            st.markdown(f"**Cost basis:** {display_currency} {total_cb:,.2f} &nbsp;|&nbsp; "
                        f"**Market value:** {display_currency} {total_mv:,.2f} &nbsp;|&nbsp; "
                        f"**Unrealized P/L:** {display_currency} {total_upl:+,.2f}")

            # Tax-loss harvesting section (also in Tax tab, summarised here)
            losers = disp[disp[pl_col] < 0].copy()
            if not losers.empty:
                st.markdown("---")
                st.markdown("#### 📉 Tax-Loss Harvesting Opportunities")
                st.caption("Positions currently at a loss. Selling these realises the loss "
                           "and offsets gains for Irish CGT purposes.")
                total_harvestable_eur = abs(losers[pl_col].sum()) if display_currency == 'EUR' else \
                    abs(losers[f'Unrealized P/L ({display_currency})'].sum()) / (eur_rate or 1)
                potential_saving = total_harvestable_eur * IRISH_CGT_RATE
                st.info(f"Total harvestable loss: **€{total_harvestable_eur:,.2f}** → "
                        f"potential CGT saving of **€{potential_saving:,.2f}** (at 33% rate)")
                st.dataframe(losers[['Ticker', 'Quantity', f'Cost Basis ({display_currency})',
                                     f'Market Value ({display_currency})', pl_col, pct_col]]
                             .style.format({pl_col: '{:,.2f}', pct_col: '{:.2f}%',
                                            f'Cost Basis ({display_currency})': '{:,.2f}',
                                            f'Market Value ({display_currency})': '{:,.2f}',
                                            'Quantity': '{:.4f}'}),
                             use_container_width=True)
        else:
            st.info("No active holdings.")

    # ─── TAB 3: COMPLETED TRADES ───
    with tab3:
        st.subheader("Completed Transactions")
        st.caption(f"Values in {display_currency}. EUR P/L uses FX rate at time of sale.")
        if not df_completed.empty:
            disp = df_completed.copy()
            for col in ['Avg. Buy Price (USD)', 'Sell Price p.s. (USD)',
                        'Cost Basis Sold (USD)', 'Total Proceeds (USD)', 'Realized P/L (USD)']:
                new = col.replace('(USD)', f'({display_currency})')
                disp[new] = disp[col].apply(lambda x: usd_to_display(x, display_currency, eur_rate))

            pl_col = f'Realized P/L ({display_currency})'
            # Only add the raw EUR column when the display currency is NOT EUR —
            # otherwise pl_col and 'Realized P/L (EUR)' are the same string,
            # which creates duplicate columns and breaks pandas Styler.
            show_eur_col = display_currency != 'EUR' and 'Realized P/L (EUR)' in disp.columns
            show = ['Date', 'Ticker', 'Type', 'Quantity Sold',
                    f'Avg. Buy Price ({display_currency})',
                    f'Sell Price p.s. ({display_currency})',
                    f'Cost Basis Sold ({display_currency})',
                    f'Total Proceeds ({display_currency})',
                    pl_col,
                    *(['Realized P/L (EUR)'] if show_eur_col else []),
                    'Days Held']

            fmt_dict = {
                'Quantity Sold': '{:.4f}', 'Days Held': '{:.0f}',
                f'Avg. Buy Price ({display_currency})': '{:,.2f}',
                f'Sell Price p.s. ({display_currency})': '{:,.2f}',
                f'Cost Basis Sold ({display_currency})': '{:,.2f}',
                f'Total Proceeds ({display_currency})': '{:,.2f}',
                pl_col: '{:,.2f}',
                'Date': '{:%Y-%m-%d}',
            }
            if show_eur_col:
                fmt_dict['Realized P/L (EUR)'] = '€{:,.2f}'

            color_subset = [pl_col] + (['Realized P/L (EUR)'] if show_eur_col else [])
            styled = disp[show].style.format(fmt_dict).map(
                lambda v: 'color: red' if isinstance(v, (int, float)) and v < 0
                else ('color: green' if isinstance(v, (int, float)) and v > 0 else ''),
                subset=color_subset
            )
            st.dataframe(styled, use_container_width=True)
            st.markdown(f"**Total realized P/L: {display_currency} {disp[pl_col].sum():+,.2f}** "
                        f"({wins} wins / {losses} losses, win rate {win_rate:.1f}%)")
        else:
            st.info("No completed trades yet.")

    # ─── TAB 4: ALL TRANSACTIONS ───
    with tab4:
        st.subheader("Processed Transaction Data")
        st.caption("'Amount' is USD-normalized. 'EUR_Equivalent' uses the historical FX rate per row.")
        show = ['Date', 'Ticker', 'Type', 'Quantity', 'Price per share',
                'Original_Currency', 'Original_Amount', 'FX Rate',
                'Amount', 'Price_per_share_USD', 'EUR_Equivalent']
        show = [c for c in show if c in df_analysis.columns]
        disp = df_analysis[show].sort_values('Date', ascending=False).copy()
        for col in ['Original_Amount', 'Amount', 'Price per share', 'Price_per_share_USD',
                    'FX Rate', 'Quantity', 'EUR_Equivalent']:
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors='coerce')
        disp['Date'] = disp['Date'].dt.strftime('%Y-%m-%d')
        fmt_map = {k: v for k, v in {
            'Original_Amount': '{:,.2f}', 'Amount': '${:,.2f}', 'EUR_Equivalent': '€{:,.2f}',
            'Price per share': '{:,.4f}', 'Price_per_share_USD': '${:,.4f}',
            'FX Rate': '{:.4f}', 'Quantity': '{:.4f}',
        }.items() if k in disp.columns}
        st.dataframe(disp.style.format(fmt_map), use_container_width=True)

    # ─── TAB 5: CHARTS ───
    with tab5:
        st.subheader("Visualizations")

        # Asset allocation pie
        if not df_holdings.empty and 'Market Value (USD)' in df_holdings.columns:
            st.markdown("#### Asset Allocation by Market Value")
            chart_df = df_holdings[df_holdings['Market Value (USD)'] > 0].copy()
            if not chart_df.empty:
                chart_df['MV_display'] = chart_df['Market Value (USD)'].apply(
                    lambda x: usd_to_display(x, display_currency, eur_rate))
                st.altair_chart(
                    alt.Chart(chart_df).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta('MV_display:Q', stack=True),
                        color=alt.Color('Ticker:N'),
                        tooltip=['Ticker', alt.Tooltip('MV_display:Q', format=',.2f',
                                                        title=f'Value ({display_currency})')]
                    ).properties(title=f'Holdings by Market Value ({display_currency})'),
                    use_container_width=True
                )

        # Annual P/L bar chart
        if not df_completed.empty:
            st.markdown("#### Annual Realized P/L (EUR)")
            ann = df_completed.copy()
            ann['Year'] = pd.to_datetime(ann['Date']).dt.year.astype(str)
            ann_sum = ann.groupby('Year')['Realized P/L (EUR)'].sum().reset_index()
            ann_sum['color'] = ann_sum['Realized P/L (EUR)'].apply(lambda x: 'Profit' if x >= 0 else 'Loss')
            st.altair_chart(
                alt.Chart(ann_sum).mark_bar().encode(
                    x=alt.X('Year:O'),
                    y=alt.Y('Realized P/L (EUR):Q', title='Realized P/L (€)'),
                    color=alt.Color('color:N', scale=alt.Scale(domain=['Profit', 'Loss'],
                                                                range=['#2ecc71', '#e74c3c']),
                                    legend=None),
                    tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('Realized P/L (EUR):Q', format=',.2f')]
                ).properties(title='Annual Realized P/L (EUR)'),
                use_container_width=True
            )

        # Monthly Realized P/L
        if not df_completed.empty:
            st.markdown("#### Monthly Realized P/L")
            tmp = df_completed.copy()
            tmp['Month'] = pd.to_datetime(tmp['Date']).dt.tz_localize(None).dt.to_period('M').astype(str)
            monthly = tmp.groupby('Month')['Realized P/L (USD)'].sum().reset_index()
            monthly['PL_disp'] = monthly['Realized P/L (USD)'].apply(
                lambda x: usd_to_display(x, display_currency, eur_rate))
            st.altair_chart(
                alt.Chart(monthly).mark_bar().encode(
                    x=alt.X('Month:O', sort=None),
                    y=alt.Y('PL_disp:Q', title=f'Realized P/L ({display_currency})'),
                    color=alt.condition(alt.datum.PL_disp > 0, alt.value('#2ecc71'), alt.value('#e74c3c')),
                    tooltip=[alt.Tooltip('Month:O'), alt.Tooltip('PL_disp:Q', format=',.2f')]
                ).properties(title=f'Monthly Realized P/L ({display_currency})'),
                use_container_width=True
            )

        # Top winners / losers
        if not df_completed.empty:
            st.markdown("#### Top Winners & Losers by Ticker")
            per_ticker = (
                df_completed[df_completed['Type'].isin(SELL_TYPES + POSITION_CLOSURE)]
                .groupby('Ticker')['Realized P/L (USD)'].sum().reset_index()
            )
            per_ticker['PL_disp'] = per_ticker['Realized P/L (USD)'].apply(
                lambda x: usd_to_display(x, display_currency, eur_rate))
            per_ticker = per_ticker.sort_values('PL_disp', ascending=False)
            top = pd.concat([per_ticker.head(5), per_ticker.tail(5)]).drop_duplicates()
            st.altair_chart(
                alt.Chart(top).mark_bar().encode(
                    x=alt.X('Ticker:N', sort=alt.SortField('PL_disp', 'descending')),
                    y=alt.Y('PL_disp:Q', title=f'Realized P/L ({display_currency})'),
                    color=alt.condition(alt.datum.PL_disp > 0, alt.value('#2ecc71'), alt.value('#e74c3c')),
                    tooltip=['Ticker', alt.Tooltip('PL_disp:Q', format=',.2f')]
                ).properties(title='Top 5 Winners & Losers'),
                use_container_width=True
            )

        # Cumulative realized P/L
        if not df_completed.empty:
            st.markdown("#### Cumulative Realized P/L Over Time")
            cum = df_completed[df_completed['Type'].isin(SELL_TYPES + POSITION_CLOSURE)].copy()
            cum = cum.sort_values('Date')
            cum['Cumulative'] = cum['Realized P/L (USD)'].cumsum().apply(
                lambda x: usd_to_display(x, display_currency, eur_rate))
            line = alt.Chart(cum).mark_line(point=True).encode(
                x=alt.X('Date:T'),
                y=alt.Y('Cumulative:Q', title=f'Cumulative P/L ({display_currency})'),
                tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Ticker',
                         alt.Tooltip('Cumulative:Q', format=',.2f')]
            ).properties(title=f'Cumulative Realized P/L ({display_currency})')
            zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray',
                             strokeDash=[4, 4]).encode(y='y:Q')
            st.altair_chart(line + zero, use_container_width=True)

        # Holding period distribution
        if not df_completed.empty:
            st.markdown("#### Holding Period Distribution (Closed Trades)")
            hp = df_completed[df_completed['Days Held'] > 0][['Days Held', 'Ticker', 'Realized P/L (USD)']].copy()
            hp['Profit/Loss'] = hp['Realized P/L (USD)'].apply(lambda x: 'Win' if x >= 0 else 'Loss')
            if not hp.empty:
                st.altair_chart(
                    alt.Chart(hp).mark_bar(opacity=0.7, binSpacing=0).encode(
                        x=alt.X('Days Held:Q', bin=alt.Bin(maxbins=30), title='Days Held'),
                        y=alt.Y('count():Q', title='Number of Trades'),
                        color=alt.Color('Profit/Loss:N',
                                        scale=alt.Scale(domain=['Win', 'Loss'],
                                                        range=['#2ecc71', '#e74c3c'])),
                        tooltip=[alt.Tooltip('Days Held:Q', bin=alt.Bin(maxbins=30)), 'count()']
                    ).properties(title='Holding Period Distribution'),
                    use_container_width=True
                )

        # Monthly deposits + cumulative
        deps_df = df_analysis[df_analysis['Type'].isin(CASH_TOP_UP_TYPES)].copy()
        if not deps_df.empty:
            st.markdown("#### Cash Deposits Over Time")
            deps_df['Month'] = deps_df['Date'].dt.tz_localize(None).dt.to_period('M').astype(str)
            deps_month = deps_df.groupby('Month')['Amount'].sum().reset_index()
            deps_month['Dep_disp'] = deps_month['Amount'].apply(
                lambda x: usd_to_display(x, display_currency, eur_rate))
            bar_dep = alt.Chart(deps_month).mark_bar(color='#3498db').encode(
                x=alt.X('Month:O', sort=None),
                y=alt.Y('Dep_disp:Q', title=f'Deposits ({display_currency})'),
                tooltip=[alt.Tooltip('Month:O'), alt.Tooltip('Dep_disp:Q', format=',.2f')]
            ).properties(title=f'Monthly Deposits ({display_currency})')
            st.altair_chart(bar_dep, use_container_width=True)

            deps_df2 = deps_df.sort_values('Date').copy()
            deps_df2['Cumulative'] = deps_df2['Amount'].cumsum().apply(
                lambda x: usd_to_display(x, display_currency, eur_rate))
            area_dep = alt.Chart(deps_df2).mark_area(opacity=0.5, color='#3498db').encode(
                x=alt.X('Date:T'),
                y=alt.Y('Cumulative:Q', title=f'Cumulative Deposits ({display_currency})'),
                tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'),
                         alt.Tooltip('Cumulative:Q', format=',.2f')]
            ).properties(title=f'Cumulative Cash Deposited ({display_currency})')
            st.altair_chart(area_dep, use_container_width=True)

        # Monthly buy vs sell
        buysell_df = df_analysis[df_analysis['Type'].isin(BUY_TYPES + ALL_SELL_LIKE)].copy()
        if not buysell_df.empty:
            st.markdown("#### Monthly Buy vs. Sell Activity")
            buysell_df['Month'] = buysell_df['Date'].dt.tz_localize(None).dt.to_period('M').astype(str)
            buysell_df['Cat'] = buysell_df['Type'].apply(lambda t: 'Buy' if t in BUY_TYPES else 'Sell')
            pivot = buysell_df.groupby(['Month', 'Cat'])['Amount'].sum().unstack(fill_value=0).reset_index()
            for col in ['Buy', 'Sell']:
                if col not in pivot.columns:
                    pivot[col] = 0
                pivot[f'{col}_disp'] = pivot[col].apply(lambda x: usd_to_display(x, display_currency, eur_rate))
            melted = pivot.melt(id_vars='Month', value_vars=['Buy_disp', 'Sell_disp'],
                                var_name='Category', value_name='Amount')
            melted['Category'] = melted['Category'].str.replace('_disp', '', regex=False)
            st.altair_chart(
                alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X('Month:O', sort=None),
                    y=alt.Y('Amount:Q', title=f'Amount ({display_currency})'),
                    color=alt.Color('Category:N',
                                    scale=alt.Scale(domain=['Buy', 'Sell'],
                                                    range=['#e74c3c', '#2ecc71'])),
                    tooltip=[alt.Tooltip('Month:O'), 'Category', alt.Tooltip('Amount:Q', format=',.2f')]
                ).properties(title=f'Monthly Buy vs. Sell ({display_currency})'),
                use_container_width=True
            )

        # Dividends
        div_df = df_analysis[df_analysis['Type'].isin(DIVIDEND_TYPES)].copy()
        if not div_df.empty:
            st.markdown("#### Monthly Dividend Income")
            div_df['Month'] = div_df['Date'].dt.tz_localize(None).dt.to_period('M').astype(str)
            divs = div_df.groupby('Month')['Amount'].sum().reset_index()
            divs['Div_disp'] = divs['Amount'].apply(lambda x: usd_to_display(x, display_currency, eur_rate))
            st.altair_chart(
                alt.Chart(divs).mark_bar(color='#f1c40f').encode(
                    x=alt.X('Month:O', sort=None),
                    y=alt.Y('Div_disp:Q', title=f'Dividends ({display_currency})'),
                    tooltip=[alt.Tooltip('Month:O'), alt.Tooltip('Div_disp:Q', format=',.2f')]
                ).properties(title=f'Monthly Dividend Income ({display_currency})'),
                use_container_width=True
            )

    # ─── TAB 6: IRISH TAX ───
    with tab6:
        st.subheader("🧾 Irish CGT Calculator")
        st.markdown("""
> **Disclaimer:** This is an estimate for planning purposes only. Consult a qualified Irish tax adviser
> (tax agent/accountant) for your official CGT return. Exchange rates and exact lot-level cost bases
> may differ from Revenue's required calculation.
        """)

        col_a, col_b = st.columns(2)
        col_a.markdown(f"""
**Irish CGT Rules (Revenue.ie):**
- **Rate:** 33% on net capital gains
- **Annual exemption:** €1,270 per person (use-it-or-lose-it — does NOT carry forward)
- **Losses:** Offset same-year gains first; any remaining loss carries forward indefinitely
- **Currency:** Gains/losses calculated in EUR using the exchange rate on each transaction date
- **No indexation relief** for assets acquired after January 2003
        """)
        col_b.markdown(f"""
**Payment deadlines:**
- **Jan 1 – Nov 30 gains** → pay CGT by **December 15** of the *same* year
- **Dec 1 – Dec 31 gains** → pay CGT by **January 31** of the *following* year
- **CGT return (Form CG1)** → file by **31 October** (or **15 November** via ROS) of the *following* year

**Dividend income note:**
- Foreign dividends (US stocks) are taxed as *income*, not CGT
- Marginal income tax (20%/40%) + PRSI 4% + USC applies
- 15% US withholding tax is creditable under the Ireland–USA tax treaty
        """)

        st.markdown("---")
        st.subheader("Year-by-Year CGT Breakdown (EUR)")

        if cgt_rows:
            cgt_df = pd.DataFrame(cgt_rows)
            st.dataframe(cgt_df.style.format({
                'Gains (€)': '€{:,.2f}', 'Losses (€)': '€{:,.2f}',
                'Net P/L (€)': '€{:,.2f}', 'Loss Relief Used (€)': '€{:,.2f}',
                'Adj. Net (€)': '€{:,.2f}', 'Annual Exemption (€)': '€{:,.2f}',
                'Taxable Gain (€)': '€{:,.2f}', 'CGT @ 33% (€)': '€{:,.2f}',
                'Losses Carried Fwd (€)': '€{:,.2f}',
            }).map(lambda v: 'color: red; font-weight: bold'
                  if isinstance(v, (int, float)) and v > 0 and 'CGT' in str(v) else '',
                  subset=['CGT @ 33% (€)']),
            use_container_width=True)
        else:
            st.info("No completed trades yet — no CGT to calculate.")

        st.markdown("---")
        c_t1, c_t2, c_t3 = st.columns(3)
        c_t1.metric("Total CGT Owed (all years, est.)", f"€{total_cgt:,.2f}",
                    help="Based on EUR P/L calculated using FX rate at time of each sale.")
        c_t2.metric("Losses Carried Forward", f"€{cgt_carry_fwd:,.2f}",
                    help="Unused losses from any year, available to offset future gains.")
        c_t3.metric("Unrealized EUR Gains (holdings)", f"€{unreal_eur_gains:,.2f}",
                    help="Sum of positive unrealized gains converted to EUR at today's rate.")

        if potential_cgt > 0:
            st.warning(
                f"⚠️ **Potential CGT if all profitable holdings sold today:** €{potential_cgt:,.2f}  \n"
                f"(This uses today's EUR/USD rate of {eur_rate:.4f} and assumes €1,270 exemption not yet used.)"
            )

        st.markdown("---")
        st.subheader("Year-by-Year Realized P/L (EUR) — Chart")
        if cgt_rows:
            ann_chart_df = pd.DataFrame(cgt_rows)[['Year', 'Net P/L (€)', 'CGT @ 33% (€)']].copy()
            ann_chart_df['Year'] = ann_chart_df['Year'].astype(str)
            melted_ann = ann_chart_df.melt(id_vars='Year', var_name='Metric', value_name='Amount (€)')
            st.altair_chart(
                alt.Chart(melted_ann).mark_bar().encode(
                    x=alt.X('Year:O'),
                    y=alt.Y('Amount (€):Q'),
                    color=alt.Color('Metric:N',
                                    scale=alt.Scale(domain=['Net P/L (€)', 'CGT @ 33% (€)'],
                                                    range=['#3498db', '#e74c3c'])),
                    xOffset='Metric:N',
                    tooltip=[alt.Tooltip('Year:O'), 'Metric', alt.Tooltip('Amount (€):Q', format=',.2f')]
                ).properties(title='Annual Net P/L vs CGT Owed (EUR)'),
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📉 Tax-Loss Harvesting")
        if not df_holdings.empty:
            losers_tax = df_holdings[df_holdings['Unrealized P/L (USD)'] < 0].copy()
            if not losers_tax.empty and eur_rate:
                losers_tax['Unrealized Loss (€)'] = losers_tax['Unrealized P/L (USD)'] / eur_rate
                losers_tax['CGT Saving if Sold (€)'] = abs(losers_tax['Unrealized Loss (€)']) * IRISH_CGT_RATE
                total_saving = losers_tax['CGT Saving if Sold (€)'].sum()
                st.info(
                    f"Selling all positions below cost would realise **€{abs(losers_tax['Unrealized Loss (€)'].sum()):,.2f}** "
                    f"in losses and save up to **€{total_saving:,.2f}** in CGT by offsetting future gains."
                )
                st.dataframe(losers_tax[['Ticker', 'Quantity', 'Cost Basis (USD)',
                                         'Market Value (USD)', 'Unrealized Loss (€)',
                                         'CGT Saving if Sold (€)']].style.format({
                    'Quantity': '{:.4f}', 'Cost Basis (USD)': '${:,.2f}',
                    'Market Value (USD)': '${:,.2f}',
                    'Unrealized Loss (€)': '€{:,.2f}', 'CGT Saving if Sold (€)': '€{:,.2f}',
                }), use_container_width=True)
            else:
                st.success("All current holdings are at a profit — no tax-loss harvesting available right now.")
        else:
            st.info("No current holdings to analyse.")

elif not uploaded_file and not st.session_state.manual_entries:
    st.info("👈 Upload your Revolut brokerage statement (.xlsx) or add transactions manually.")
elif st.session_state.transactions_df is not None and st.session_state.transactions_df.empty:
    st.warning("No processable transaction data found. Check your file.")
else:
    st.error("Failed to load transaction data. Check error messages above.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Notes:**
- All sell types (market, limit, stop, position closure) counted in P&L.
- Stock splits & mergers handled: quantity adjusted, cost basis preserved.
- EUR Equivalent uses each row's own FX rate — NOT today's rate.
- CGT is estimated; consult a tax adviser for your official return.
- Live prices via Yahoo Finance (may be delayed). Delisted ('Q') = $0.
- Not financial advice.
""")
st.caption(f"Brokerage Account Analyzer Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
