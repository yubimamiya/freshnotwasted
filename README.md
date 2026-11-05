# FreshCast: Fresh, Not Wasted

Walmart Global Tech Student AI Hackathon · Group 3 challenge: **Fresh, Not Wasted**

> Managing fresh produce, bakery, and meat requires a delicate balance: overstocking creates landfill waste, while understocking leaves shelves empty and shoppers frustrated. Transform how a store manages fresh inventory to maximize product abundance while minimizing spoilage.
>
> **The twist:** A severe winter storm warning is announced for tomorrow. Customer foot traffic surges immediately, but supplier truck deliveries are delayed by 48 hours. How should your solution adapt in real time?

FreshCast predicts how much of each fresh item each store will sell, turns that prediction into an order, and tells staff what to mark down or donate before it spoils. It also estimates how many staff hours that work takes. When a storm hits and trucks are delayed, the plan adjusts so the stock already in the store lasts until the next delivery.

This repo contains a **working prototype**: a forecasting and ordering engine, a back-test against a simple ordering rule, a local website that walks through the solution step by step with live data, and a chat assistant that answers questions about the plan using Claude.

---

## Results (simulated data)

Tested on days the model never trained on, against a simple rule: *"order what sold on the same weekday last week, plus 10%."* All 3 stores and 24 items combined.

| | Current rule | FreshCast |
|---|---|---|
| **Normal 4 weeks** (Oct 8 – Nov 4, 2026) | | |
| Food thrown away | $3,059 | **$361** (88% less) |
| Customer demand met (in stock) | 83.1% | **94.8%** |
| Sales lost to empty shelves (units) | 12,095 | **4,514** |
| Food donated | 0 kg | **454 kg** |
| Gross profit | $99,132 | **$106,296** |
| **Snowstorm week** (Jan 19 – 28, 2026) | | |
| Customer demand met | 73.4% | **95.0%** |
| Sales lost to empty shelves (units) | 6,033 | **1,447** |
| Food thrown away | $789 | **$325** |

**Caveats:** all data is simulated, and the comparison rule is a simple stand-in, not Walmart's actual ordering system. The results show the approach works. They are not a prediction of exact savings in a real store.

---

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/yubimamiya/freshnotwasted
cd freshnotwasted
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py
# macOS / Linux:
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
```

Then open **http://localhost:8000**. The plans for both scenarios are computed at startup, which takes about 2 seconds. Everything runs locally.

Use the arrow keys or the dots on the right to move between sections. The bar at the top switches stores and switches between **Normal day** and **Winter storm + 48h truck delay**. Every live section updates.

### Chat assistant

The assistant in Step 7 tries these in order:

1. **Claude Code CLI**: if `claude` is installed and signed in on the machine, questions go through it. No API key needed. Answers take about 20 seconds.
2. **Claude API**: if `ANTHROPIC_API_KEY` is set, it calls `claude-opus-5-5` through the `anthropic` Python SDK.
3. **Built-in rules**: if neither is available, it answers common questions (a product, the storm, donations, staff hours, trends) from the plan data.

In every case the assistant is given only the current plan's data and is told not to invent numbers.

### Run the engine without the website

```bash
python freshcast.py          # prints back-test results and Store 101's top actions
python freshcast.py --storm  # same, in the storm scenario
```

---

## How it works

The system runs seven steps every night, for every item in every store.

```
 1 Data  →  2 Features  →  3 Forecast  →  4 Order  →  5 Surplus  →  6 Staff time  →  7 Manager plan + assistant
```

### 1. Data

In a real deployment, each input comes from a source most stores already have:

| Input | Real-world source | Why it matters |
|---|---|---|
| Sales per item, store, day | Point-of-sale | The core signal |
| When a shelf went empty | Shelf cameras | Sales on a sold-out day understate demand; this lets the model learn true demand instead of repeating shortages |
| Stock by expiry date | Date-coded barcodes / RFID | Needed to know what's usable and what to mark down or donate |
| Holidays, promotions | Calendar, pricing system | Known in advance |
| Weather, storm alerts | Weather service alerts | Storms cause a buying rush before and a drop during |
| Truck schedule and delays | Truck GPS / logistics | Sets how many days current stock must last |
| Social media trends | Trend monitoring | e.g. "Dubai chocolate" posts → strawberry demand a few days later |

**In this prototype, all of it is simulated** (`World` class in `freshcast.py`): 3 stores (Supercenter, Neighborhood Market, Supercenter), 24 fresh items across Produce, Bakery, Meat, Dairy and Deli, and daily data from Jan 2024 to Nov 2026. Demand includes:

- day-of-week and seasonal patterns, temperature effects, and slow growth
- holiday build-up, peak and dip, with a different strength per product (e.g. turkey ×25 at Thanksgiving, pumpkin pie ×10, avocados ×3 for the Super Bowl)
- promotions with category price sensitivity
- 4 past winter storms (pre-storm rush, drop on storm day, recovery), each cancelling 2 days of deliveries
- social trend bursts linked to specific products, affecting demand with a 3-day lag
- random day-to-day noise (gamma-Poisson)

### 2. Features

Each training row is **one product × one store × one day**. Features (`World.X`):

| Group | Features |
|---|---|
| Day of week | 6 indicator columns |
| Time of year | sine/cosine of day of year |
| Weather | temperature vs. 50°F |
| Promotion | on sale that day |
| **Holiday markers** | for each of 8 holidays (Thanksgiving, Christmas, New Year's Eve, Easter, July 4th, Super Bowl, Valentine's Day, Cinco de Mayo), 5 windows: 6–10 days before, 3–5 days before, 1–2 days before (peak), the day itself, 1–2 days after (dip) |
| Storm phase | 2 days before, 1 day before, storm day, 1 and 2 days after |
| Social trend | trend score (0–100) of the linked topic, 3 days earlier |
| Growth | time trend |

### 3. Forecast (one model per product)

- Every product in every store gets **its own model**, because products react very differently to the same day.
- Model: ridge regression on `log(1 + sales)`, so effects multiply (e.g. "Saturday ×1.25" × "Thanksgiving week ×4").
- **Margin of error:** the 10th and 90th percentiles of the model's past errors give an 80% "likely range" for every forecast.
- **Explanations:** each feature's contribution is grouped (Day of week, Time of year, Holiday, Storm warning, Social media trend, …) and shown as "+X% vs. a typical day."
- Forecast horizon: 14 days.

A production version would keep the same features and use gradient-boosted trees (e.g. LightGBM with quantile loss), plus a category-level model for new or very slow-selling items.

### 4. Ordering

```
target in-stock level = cost of running out ÷ (cost of running out + cost of waste)    (clipped to 75–97%)
stock target          = expected sales + z(target) × uncertainty
order                 = stock target − usable stock on hand  → rounded to full cases
```

- **Cost of running out** = price − cost, plus a penalty of 50% of price for staples (bread, milk, eggs, meat).
- **Cost of waste** = 60% of cost, since markdowns and donations recover some value.
- **Usable stock** = units that won't expire before they can sell. Oldest stock sells first.
- **Coverage window:** normally until tomorrow's truck. In a storm, until trucks return (3 days).
- **Cap:** never order more than can sell before it spoils.

### 5. Surplus: markdowns, then donations

For every batch expiring today or tomorrow:

1. **Predict leftovers:** batch size − (expected sales before expiry − older stock ahead of it).
2. **Mark down** if at least 15% is likely left over: 30% off, or 50% if more than half.
3. **Donate** what's still expected to be left: the nearest partner that can take that food type (meat and dairy need cold storage) is notified today for a 9 AM pickup.

In the storm, markdowns stop on essentials because they will sell at full price.

### 6. Staff time

Each action becomes minutes of work using these standards (assumptions; a pilot would use the store's own time studies):

| Task | Minutes |
|---|---|
| Receive and stock one case | 4.0 |
| Rotate and check dates (per case on hand) | 0.4 |
| Markdown (per batch + per unit) | 2.0 + 0.15 |
| Donation (per unit + per pickup) | 0.3 + 10 |
| Storm prep (per essential item) | 12 |

### 7. Manager plan + assistant

All actions are merged into one prioritized to-do list (Storm → Order → Markdown → Donate) with summary numbers. The manager can ask questions in plain English (see *Chat assistant* above).

### The storm twist

Switching to the storm scenario changes only the inputs. The same pipeline then:

- **Forecast:** storm features fire, giving a rush on essentials today and a drop on the storm day.
- **Ordering:** stock must last today through Saturday. For essentials short of that, it requests a same-day distribution-center top-up and sets a 2-per-customer limit.
- **Surplus:** markdowns stop on essentials. Items whose sales fall in a storm (e.g. sushi, rotisserie chicken) are marked down or donated sooner.
- **Staff:** adds tasks: release backroom stock in 3 rounds (now, 6 PM, 7 AM), bake extra bread in-store (the only supply that doesn't depend on trucks), put up limit signs.

### Back-test

`simulate()` replays each test window day by day with batch-level inventory (each delivery is a batch with an expiry date):

- Models are trained **only on data before the window**.
- Both policies see the same customers, shelf lives and truck cancellations.
- **Current rule:** order 1.1 × sales on the same weekday last week, minus usable stock. No markdowns, no donations, and it doesn't know about storms or truck delays. It learns from sales, not demand, so shortages repeat.
- **FreshCast:** forecast-based ordering with multi-day coverage when trucks are cancelled, markdowns on expiring batches (extra bargain demand at 30% off), and donation of 85% of the remaining expired units.

---

## Project structure

```
freshcast.py     Data simulator, features, per-product models, ordering / markdown / donation / staffing logic, back-test
agent.py         Chat assistant: Claude Code CLI → Claude API → built-in rules
server.py        Local web server: serves the site, /api/plan?storm=0|1, POST /api/ask
site/index.html  Step-by-step walkthrough website with live charts and tables (no external libraries)
requirements.txt numpy (required), anthropic (optional)
```

### API

| Endpoint | Returns |
|---|---|
| `GET /api/plan?storm=0` or `1` | Full plan JSON: per-store products, forecasts, actions, markdowns, donations, labor, plus trends, events and back-test |
| `POST /api/ask` with body `{"q": "...", "store": "Store 101 · Supercenter", "storm": false}` | `{"text": "...", "source": "..."}` |

---

## Limits and next steps

**Limits**
- All data is simulated.
- The "current rule" is a simple stand-in, not Walmart's ordering system.
- Shelf lives, labor minutes and donation rates are assumptions.
- The forecasting model is deliberately simple (ridge regression per product).

**Next steps**
- Pilot in a few stores with real sales and shelf-camera data.
- Switch to gradient-boosted trees, and update forecasts during the day as sales come in.
- Move stock between nearby stores when one has extra.
- Connect donation notices to food banks' scheduling systems.
