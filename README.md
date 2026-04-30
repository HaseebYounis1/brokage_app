# Brokerage Account Analyzer Pro

A Streamlit web app for analyzing your Revolut brokerage account statement. Upload your transaction history and get a full breakdown of your portfolio, P&L, deposits, trade statistics, and Irish CGT tax estimate — displayed in USD or EUR.

## Getting Started

### 1. Download your statement from Revolut

1. Open the Revolut app
2. Go to **Investing** → your portfolio
3. Tap the menu (top right) → **Account statement**
4. Select the full date range and export as **Excel (.xlsx)**

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run brokerage_app.py
```

Then upload your `.xlsx` file using the sidebar.

---

## Tabs & Features

### 📊 Overview
- **Portfolio market value** — live prices from Yahoo Finance; fallback to last transaction price
- **Estimated cash balance** — deposits − buys + sells + dividends − fees
- **Total account value** — portfolio + cash
- **Overall ROI %** and **annualized return %** based on account history
- **Overall / realized / unrealized P&L** in one view
- **Cash deposited (USD)** — as recorded by Revolut
- **Cash deposited (EUR equivalent)** — each deposit's USD amount divided by the EUR/USD FX rate at the time of deposit; this is the actual EUR transferred from your bank
- **Buy turnover vs. deposits explained** — the app clarifies why total buy volume is much higher than your actual deposits (reinvestments compound the turnover figure)
- **Trade statistics** — win rate, profit factor, trade expectancy, average win/loss, average holding period
- Full P&L breakdown showing every income and cost component

### 💼 Current Holdings
- All unsold positions: live price, cost basis, unrealized P&L, % gain/loss
- Color-coded green/red
- Sorted by market value
- **Tax-loss harvesting** section: highlights positions currently at a loss with the potential CGT saving if sold

### 🤝 Completed Trades
- Every sell (market, limit, stop-loss), position closure, and merger surrender
- Shows avg buy price, sell price, cost basis, proceeds, realized P&L (display currency + raw EUR using historical FX rate), and days held
- Summary totals with win/loss count and win rate

### 📋 All Transactions
- Full processed transaction log
- Original currency, FX rate, USD-normalized amount, and EUR equivalent per row

### 📈 Charts
| Chart | Description |
|-------|-------------|
| Asset allocation | Pie chart of current holdings by market value |
| Annual realized P&L | Bar chart by year in EUR |
| Monthly realized P&L | Bar chart, green = profit / red = loss |
| Top winners & losers | Realized P&L by ticker (top 5 each) |
| Cumulative realized P&L | Line chart over time with zero reference line |
| Holding period distribution | Histogram of closed trades, wins vs losses colored |
| Monthly cash deposits | Bar chart |
| Cumulative deposits | Area chart |
| Monthly buy vs. sell activity | Line chart |
| Monthly dividend income | Bar chart |

### 🧾 Tax (Ireland) — Irish CGT Calculator
- **Year-by-year CGT table** — gains, losses, net P&L, €1,270 annual exemption applied, CGT @ 33%, and losses carried forward
- **Loss carry-forward** — unused losses automatically rolled into the next year
- **Filing & payment deadlines** — Dec 15 for Jan–Nov gains; Jan 31 for December gains; return by Oct 31 / Nov 15 (ROS)
- **Dividend income note** — US dividends taxed as income (not CGT); 15% US withholding tax is creditable
- **Potential CGT if all profitable holdings sold today**
- **Tax-loss harvesting table** — which current positions at a loss could offset future gains, with the estimated CGT saving per position
- Annual net P&L vs CGT owed bar chart

> CGT figures use the EUR/USD FX rate at the date of each sale to convert USD gains to EUR. This is an estimate — consult a qualified Irish tax adviser for your official Revenue return.

---

## Supported Transaction Types

| Type | Handled as |
|------|------------|
| BUY - MARKET | Buy — adds to position, updates average cost basis |
| SELL - MARKET | Sell — realizes P&L |
| SELL - LIMIT | Sell — realizes P&L |
| SELL - STOP | Sell — realizes P&L |
| POSITION CLOSURE | Sell at broker-assigned cash value |
| CASH TOP-UP | Cash deposit |
| DIVIDEND | Income — added to P&L |
| DIVIDEND TAX (CORRECTION) | Tax adjustment on dividends |
| CUSTODY FEE | Fee — reduces P&L |
| STOCK SPLIT | Share quantity adjusted; total cost basis unchanged |
| MERGER - STOCK | New shares received at €0 cost; surrendered shares recorded as realized loss |

---

## Irish CGT Rules (summary)

| Item | Detail |
|------|--------|
| Rate | 33% |
| Annual exemption | €1,270 per person (use-it-or-lose-it each year) |
| Loss relief | Losses offset same-year gains; remainder carried forward indefinitely |
| Payment — Jan–Nov gains | Pay by **15 December** of the same year |
| Payment — December gains | Pay by **31 January** of the following year |
| Return filing | By **31 October** (or **15 November** via ROS) of the following year |
| Currency | Gains/losses calculated in EUR using exchange rate at each transaction date |

---

## Display Currency

Switch between **USD** and **EUR** in the sidebar at any time. EUR conversion for display uses the live EUR/USD rate fetched from Yahoo Finance. The `EUR_Equivalent` columns always use the historical FX rate from each transaction — not today's rate.

---

## Notes

- **P&L method:** Average cost basis per ticker (FIFO-avg).
- **Live prices:** Yahoo Finance (may be delayed). Falls back to last transaction price.
- **Delisted stocks:** Tickers ending in `Q` or containing `.OLD` are valued at $0.
- **Not financial advice.** For personal tracking and planning purposes only.
