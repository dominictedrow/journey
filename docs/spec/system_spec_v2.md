# ML Quant Pipeline — System Spec v2

> Source of truth, copied verbatim from Dom's spec (Nov 25, 2026 · @Dom). Only the formatting was changed (tables rebuilt as Markdown, formulas in LaTeX blocks).
> Changes found later are **not** edited in here. They are logged in `DECISIONS.md`, and the component docs in `docs/components/` override this spec where they disagree.

## Summary
This spec locks the pipeline's flow, data contracts, quant feature layer and portfolio rules, so the only thing left for Claude Code prompts is each model's internals. It merges the written notes with the Lucid "ML Quant Pipeline" chart and adds four things neither had:
- a quant feature layer between collection and the models
- a defined prediction target
- a numeric portfolio rules engine
- risk circuit breakers

Decisions settled on 2026-11-25:
- **Capital split:** 15% cash; invested capital 70% equity / 30% crypto, which is 59.5% / 25.5% of NAV.
- **Horizon:** one week (5 trading days for stocks, 7 days for crypto).
- **Long-hold positions are protected:** the rules engine never lets a model signal sell Anchors, Supports or crypto core.
- **Primaries removed:** its 20% is split evenly, +4 points each, across Anchors, Supports, Leaders, Standards and Satellites.
- **Position counts:** 4 Leaders, 5 Standards, 3 Satellites, plus one extra Asymmetric slot reserved for enormous-upside candidates.
- **Defaults kept:** Reddit and X dropped for v1; model sizes are config values and start small.

Model design choices are in Model contracts and LLM design. Still open: the KV-cache experiment and the LLM's final size.

## Gaps and conflicts
Twelve issues would break or bias the system if built as drawn; each has a resolution that the rest of this spec follows.

| # | Issue | Why it matters | Resolution in this spec |
|---|---|---|---|
| 1 | No prediction target defined | "Bullish/bearish" has no label, so nothing can be trained or scored | Per-asset forward 5-day excess return, triple-barrier labels (Labels section) |
| 2 | "Vote" is market-level, but the meta learner picks stocks | A single bull/bear vote can't rank 60 tickers | Every base model outputs per-asset P(up) plus one market-level score |
| 3 | Chart wires the LLM into the MLP and both CNNs; notes say MLP and CNN #1 are numeric-only | Shared inputs make the ensemble's errors correlated | Decided: all four read LLM-adjusted numbers; only the Transformer and CNN #2 also get keywords. Ablation checks the cost |
| 4 | No quant feature layer, database or portfolio state on the chart | Models would train on raw levels; rules engine can't see holdings | Added as explicit stages (Flow section) |
| 5 | Data list has S&P price but no per-stock OHLCV or fundamentals | Can't score stocks or apply tier rules (cash flow, market cap) | Added to collection |
| 6 | LLM "modifies price data" | Overwriting raw numbers destroys the training record and leaks LLM errors into every model | LLM writes an adjusted copy (raw + bounded, learned Δ); raw tables stay append-only |
| 7 | No historical news archive | Text-dependent models can only train on news collected from day one | Start collecting now; numeric models train on 10+ years immediately, text models after 6–12 months |
| 8 | Meta learner trained on in-sample base predictions | Classic stacking leak: meta learner trusts overfit models | Train meta learner only on out-of-fold, walk-forward predictions |
| 9 | "Order book from each stock" | Alpaca serves Level 2 books for crypto only; stocks get top-of-book quotes (free IEX feed, full SIP is paid) | Executor uses quotes + bars for stocks, L2 for crypto |
| 10 | Primaries = IPO / private / rights issue | Not purchasable via Alpaca API, and contradicts "low volatility" core | Primaries tier removed; its budget split across the other five equity tiers |
| 11 | Bonds in Supports | Alpaca trades bond ETFs, not individual bonds, for this use | Supports use ETFs (SHY, IEF, TLT, GLD, DBC) |
| 12 | "Trade 1–2 times a day" vs a 20+ position book | Literal reading caps the system at 2 orders | One decision cycle per day, max 8 orders per cycle |

## Model sizing check
The specified dimensions are about 1,000x beyond what the data or your RTX 5070 (12 GB) supports. Ten years of daily data is about 2,520 trading days; even as a 100-asset panel that's ~250k rows.

| Model as specified | Approx. size | Memory (bf16 weights only) |
|---|---|---|
| Transformer: 30 layers, d=20,000 | 12 × 30 × 20,000² ≈ 144B params | ~288 GB |
| MLP: 5 layers, width 20,000 | ≈ 2B params | ~4 GB, ~800 params per training row per asset |
| CNN #1: 10k × 10k input | 100M values per sample from ~150 real features | 400 MB per sample (fp32) |

- Make every dimension a config value. Suggested start: Transformer d=256, 6 layers, 8 heads (~5M params); MLP widths 512→32; CNN #1 input as a features × 64-day image.
- Paged attention and custom KV-cache logic only pay off in autoregressive decoding. These models do one forward pass per sample, so there is no cache to page. The homemade LLM is the exception: it generates tokens, so the KV-cache research belongs there.

## Flow and schedule
One decision cycle runs after each close, and orders execute the next morning; the rules engine is the only stage allowed to create an order.

All four base models read the LLM's news-adjusted numbers; only the Transformer and CNN #2 also get keywords. Sharing the adjustment makes their errors more correlated, so the ablation test in Validation checks that it pays for itself. The meta learner also reads the feature matrices and regime probabilities directly, and the rules engine reads live positions so it can enforce limits on the whole book.

| Time (ET) | Stage | Output |
|---|---|---|
| Sunday 18:00 | Weekly jobs: refresh screen, fundamentals and tier eligibility; rebuild n-gram table; retrain SVM | universe_weekly, ngram_vocab |
| First Sunday of month | Retrain base models, meta learner, HMM and price transformer (walk-forward) | New model versions |
| 16:30 | Collect prices, macro, crypto snapshot | Raw tables |
| 17:00 | News fetch, dedup, quota, SVM | Filtered articles |
| 17:30 | LLM pass | llm_outputs |
| 18:00 | Feature layer, base models, meta learner | predictions, meta_scores |
| 18:30 | Rules engine | proposed_orders |
| 09:00 next day | Pre-open check on overnight news and 08:30 releases; re-run rules if triggered | Updated proposals |
| 09:45 | Executor places orders | orders |
| 15:30 | Cancel unfilled entries, push unfilled exits | Final fills, portfolio_daily |

## Data collection
Everything comes from free APIs; FRED alone covers most macro series, and Alpaca covers prices for anything the system can trade. Items marked new are additions needed by the feature layer or the tier rules.

| Data | Source / identifier | Cadence |
|---|---|---|
| **new** Daily OHLCV for every screened stock, ETF and crypto | Alpaca Market Data (stocks: free IEX feed; crypto bars) | Daily |
| S&P 500, Nasdaq futures, gold, DXY, VIX, **new** VIX3M | yfinance: SPY, NQ=F, GC=F, DX-Y.NYB, ^VIX, ^VIX3M | Daily |
| Real estate ETF | VNQ via Alpaca | Daily |
| Treasury yields, **new** 3-month and 2-year | FRED: DGS3MO, DGS2, DGS10 | Daily |
| **new** Real yield and breakeven inflation | FRED: DFII10, T10YIE | Daily |
| Fed funds rate | FRED: DFF | Daily |
| **new** High-yield credit spread | FRED: BAMLH0A0HYM2 | Daily |
| CPI, core CPI | FRED: CPIAUCSL, CPILFESL (BLS) | Monthly |
| PCE, core PCE | FRED: PCEPI, PCEPILFE (BEA) | Monthly |
| Unemployment | FRED: UNRATE (BLS) | Monthly |
| GDP | FRED: GDPC1 (BEA) | Quarterly |
| Global liquidity (built, not fetched) | FRED: WALCL, WTREGEN, RRPONTSYD, ECBASSETSW, JPNASSETS, DEXUSEU, DEXJPUS | Weekly |
| FOMC calendar, statements, minutes, speeches | federalreserve.gov calendar page + RSS feeds | Daily check |
| Fiscal policy | Congress.gov API: bills with status changes in finance, tax and appropriations committees | Daily |
| **new** Fundamentals (market cap, FCF, margins, debt, growth) | SEC EDGAR companyfacts API (XBRL) or yfinance | Weekly with screen |
| **new** Earnings calendar | yfinance or Nasdaq calendar | Weekly |
| **new** Crypto structure: BTC dominance, stablecoin supply, funding, open interest, ETF flows | CoinGecko global, DefiLlama, Coinglass, Farside | Daily |
| News, 5 sources (2 crypto, 3 economics) | Each outlet's markets RSS feed + full-text extraction (trafilatura) | Each run |
| Reddit / X | Dropped for v1: paid or rate-limited APIs, low signal per byte | — |

Verify US access for any crypto derivatives endpoint before relying on it; several exchange APIs geo-block US users.

### Point-in-time rules
- Every raw row stores period (what it measures), released_at (when it became public) and ingested_at. A feature for decision time T may only use rows with released_at ≤ T. This one rule prevents most look-ahead bias.
- Macro data gets revised. Training uses ALFRED vintages (the value as first published); live runs use whatever is current.
- Store each weekly screen as a snapshot. Backtests use the universe as it was that week, never today's list (survivorship bias). Before your first stored screen, use historical S&P 500 membership and note the bias.
- All timestamps in UTC; trading-calendar logic in America/New_York.
- Raw tables are append-only. Nothing downstream (including the LLM) overwrites them.

### News intake and filtering
The order is: fetch → extract → dedup → quota → SVM → LLM. Dedup runs before the SVM so repeated wire copies never reach the classifier or the n-gram table.
1. Fetch every article published since the last run from each source's markets section.
2. Extract full text. Reject bodies under 150 words (paywall stubs, video pages). Tag tickers by matching against an aliases table (ticker, company name, common short names).
3. Exact dedup: SHA-256 of the canonical URL (query string stripped) and of the normalized title, checked against article_seen.
4. Near dedup: 64-bit SimHash over 5-word shingles; Hamming distance ≤ 3 to anything seen in the last 30 days = duplicate. This catches the same Reuters/AP story republished by several outlets.
5. Quota to a daily pool of about 150 articles:
   - no source above 30% of the pool
   - at least 60% mention a screened ticker or its sector
   - at most 40% general macro or unrelated

   Over quota, sample at random within each bucket with a fixed seed. Views, shares and ranking position are never collected, so popularity cannot steer selection.
6. SVM filter: keep articles with P(relevant) ≥ 0.5, but always keep the top 20 by score so the filter can never empty the pool.

The SVM gets its labels automatically. An article counts as "impactful" if an asset it mentions had a next-day excess return beyond 2× its daily volatility. The n-gram table is rebuilt weekly: TF-IDF on 1–3-grams, chi-square selection of the top 5,000 against that impact label.

## Quant feature layer
The feature layer turns raw rows into two daily matrices that every model reads:
- a market vector (about 120 macro, cross-asset and regime features)
- an asset matrix (about 40 features per screened stock, ETF and coin)

Models never see raw price levels.

### Per-asset price features
Computed for every tradable asset. Crypto annualizes with √365, everything else with √252.

| Feature | Definition | Windows (days) |
|---|---|---|
| Log return | ln(Pₜ / Pₜ₋ₕ) | 1, 5, 21, 63, 126, 252 |
| Momentum 12-1 | ln(Pₜ₋₂₁ / Pₜ₋₂₅₂), skips the last month's reversal | 252 |
| Realized vol | std of daily log returns × √252 | 21, 63 |
| Garman-Klass vol | OHLC-based vol estimator, less noisy than close-to-close | 21 |
| Downside deviation | std of negative daily returns only | 63 |
| Max drawdown | largest peak-to-trough fall in window | 63, 252 |
| Trend distance | P / SMAₙ − 1, and SMA50 / SMA200 − 1 | 50, 200 |
| RSI, MACD, Bollinger %B | standard; MACD histogram divided by price | 14; 12/26/9; 20 |
| ATR % | ATR(14) / P (also used for stops) | 14 |
| Volume z-score | (V − mean) / std of volume | 21 |
| Amihud illiquidity | mean of \|r\| / dollar volume | 21 |
| Beta and residual vol | OLS on SPY (coins: on BTC); residual std = idiosyncratic risk | 63, 252 |
| Correlations | to SPY, DXY returns and 10-year yield changes | 63 |
| Mean-reversion z | (P − SMA63) / std63 | 63 |
| OU half-life | regress Δp on pₜ₋₁ (log price): half-life = −ln 2 / ln(1 + b) | 126 |
| Hurst exponent | variance-ratio estimate; above 0.5 trends, below 0.5 mean-reverts | 126 |
| Earnings proximity | trading days to next report; last surprise % | — |

### Cross-sectional factors (stocks)
Each day every stock feature becomes a percentile rank (0–1) across the screened universe, and again within its GICS sector. Five composite factor scores are the average of their component ranks; they feed the models and drive tier classification.
- Value: FCF yield, earnings yield, EBITDA / EV
- Quality: ROIC, gross-margin stability (inverse std over 8 quarters), inverse net debt / EBITDA
- Momentum: 12-1 momentum, 63-day return
- Low volatility: inverse 252-day realized vol, inverse beta
- Growth: revenue growth YoY, EPS growth YoY

### Macro and cross-asset features
| Feature | Definition | Reads as |
|---|---|---|
| Curve slope | 10y − 2y and 10y − 3m, plus 21-day change in bps | Below zero = inversion |
| Real yield | DFII10 level and 21-day change | Rising = pressure on long-duration assets |
| Policy path proxy | 2y yield − fed funds | Negative = market pricing cuts |
| Inflation momentum | core CPI 3-month annualized − 12-month YoY (same for core PCE) | Positive = accelerating |
| Sahm indicator | 3-mo avg UNRATE − min of 3-mo avg over prior 12 months | ≥ 0.5 = recession signal |
| Real GDP growth | quarter-on-quarter, annualized | Growth leg of regime |
| Credit stress | HY OAS level, 252-day z-score, 21-day change | z > 2 = stress |
| Vol term structure | VIX / VIX3M | > 1 = backwardation, acute stress |
| VIX z-score | 252-day | Fear vs its own history |
| Dollar | DXY 21-day return and 252-day z-score | Strong dollar = headwind for crypto, gold, EM |
| Stock-bond correlation | 63-day corr(SPY, TLT) | Positive = inflation-driven regime |
| Gold residual | gold return minus fit on real-yield change (252-day OLS), z-scored | Gold rich or cheap vs rates |
| Real estate relative | VNQ / SPY, 63-day return | Rate-sensitivity gauge |
| Event clock | days to next FOMC, CPI and earnings-heavy weeks | Lets models learn pre-event behavior |
| Crypto structure | BTC dominance and 21-day change, stablecoin supply 30-day % change, 7-day avg funding (annualized), OI / market cap, 5-day ETF net flows | Leverage and liquidity in crypto |

Global liquidity index (built, not fetched). No free API publishes one, so compute it weekly in USD billions:

$$\text{GLI} = (\text{WALCL} - \text{WTREGEN} - \text{RRPONTSYD}) + \text{ECB assets} \times \text{EURUSD} + \text{BoJ assets} / \text{USDJPY}$$

The FRED series use different units (millions, billions, 100-million yen), so convert each before adding. Features: level, 13-week % change, 252-day z-score.

### Regime models
Three regime signals sit in the market vector and also steer the portfolio rules.
1. **Growth/inflation quadrant** (rules, matches your framework's Tier 1).
   - Growth up = Sahm indicator falling and real GDP growth above its 4-quarter average.
   - Inflation up = inflation momentum above zero.
   - Quadrants: Goldilocks (growth up, inflation down), Reflation (up, up), Stagflation (down, up), Deflation (down, down). Encoded one-hot.
2. **Hidden Markov model, 3 states.**
   - Gaussian HMM on weekly SPY 21-day return, SPY 21-day realized vol, 21-day change in HY OAS, and VIX / VIX3M.
   - States sorted by vol into calm, normal, stress. Output: the three state probabilities.
   - Use filtered probabilities only; smoothed probabilities use future data and leak.
   - Refit monthly on an expanding window.
3. **Stress score.** Mean of z(VIX), z(HY OAS), z(−curve slope) and a VIX / VIX3M > 1 flag. One number the circuit breakers read.

### Normalization rules
- Time-series features: rolling 252-day z-score using only past data, winsorized at ±4.
- Stock features: daily cross-sectional ranks (above), so scores compare stocks, not market levels.
- Macro gaps: forward-fill only until the next release, never back-fill. Add a days_since_release feature per macro series so models know how stale a value is.
- Feature registry table: name, formula version, window, source series. Each feature function gets a unit test against a hand calculation.

## Labels, model contracts and validation
Every model predicts the same thing: where each asset's excess return goes over the next 5 trading days. A shared target is what lets the meta learner combine them.

### Labels
- Horizon: 5 trading days for stocks and ETFs; 7 calendar days for crypto (it trades every day).
- Benchmark: stocks and ETFs vs SPY; BTC, ETH and SOL vs cash; altcoins vs BTC.
- Triple-barrier label. Let σ be the asset's 63-day EWMA daily vol and h the horizon. Track cumulative excess return R over the next h days:

$$\text{label} = \begin{cases} +1 & R \text{ hits } +1.5\,\sigma\sqrt{h} \text{ first} \\ -1 & R \text{ hits } -1.5\,\sigma\sqrt{h} \text{ first} \\ 0 & \text{neither by day } h \end{cases}$$

- Meta learner target: realized h-day excess return divided by σ√h, so scores compare across a quiet utility and a volatile altcoin.
- Executor target: next session's low (for buys) or high (for sells) relative to the prior close, in ATR units.

### Model contracts
Internals are yours to specify later; these are the fixed inputs and outputs each model must honor.

| Model | Inputs | Output | Refresh |
|---|---|---|---|
| SVM | Article TF-IDF over the 1–3-gram vocabulary; one model for all sources, with source type as a feature (easy to split into crypto and economics models later) | P(relevant) per article, binary | Retrain weekly |
| LLM (homemade) | Filtered articles + the full feature snapshot (see LLM design) | Per article: tickers, top 1–3-gram keywords, sentiment −1 to 1, event type. Per feature: a bounded adjustment Δ, giving the adjusted feature set | Every run; retrain monthly |
| Transformer | 64 days of adjusted features. One token per data point: each feature's 64-day series is projected into one token. Plus keyword tokens from the closed n-gram vocabulary, with a type embedding marking feature vs keyword | Per asset: P(up), P(flat), P(down) | Retrain monthly |
| MLP | Last 5 trading days of adjusted features, flattened, plus an asset-class flag. One model for stocks and crypto | Per asset: P(up), P(flat), P(down) | Retrain monthly |
| CNN #1 | Image of adjusted features × 64 days, rows ordered by feature family. 2D convolutions first; fall back to 1D over time if 2D underperforms | Per asset: P(up), P(flat), P(down) | Retrain monthly |
| CNN #2 | One day's tensor: assets (rows, sorted by sector so neighbors are related) × top-K 1–3-gram keywords × 5 embedding channels. Fully convolutional, so it returns one output per row | Per asset: P(up), P(flat), P(down) | Retrain monthly |
| Meta learner | Full data: 4 base-model outputs + asset matrix + market vector + regime probabilities + current weight. Scores each asset separately with shared weights | Per asset: score s, conviction c = calibrated P(up). No market score: the rules engine owns market risk | Retrain monthly, out-of-fold only. Experiment: retrain on base outputs + regime only and compare |
| Price transformer | Intraday minute bars, latest quote (L2 book for crypto), spread, ATR, side, order size / ADV | Two heads: execution slot (09:45, 11:00, 13:00 or 15:00 ET) and limit-price quantiles 10/30/50/70/90 | Timing head trains now on historical intraday bars; limit head after ~300 paper fills, using the q70 rule until then |

Two details that change results:
- Calibrate every probability (isotonic regression on a held-out fold). The portfolio rules use thresholds like c ≥ 0.65, which only mean something if 0.65 really wins about 65% of the time.
- CNN #1 needs meaningful row order. Convolutions assume neighbors are related, so order feature rows by family (returns, vol, trend, macro) or by correlation clustering, not alphabetically.

### LLM design
The LLM is a homemade decoder model trained on your machine. It reads every article plus all collected data, and its main job is to nudge the numbers for news before the four base models see them.
- **Size:** 50–125M parameters, which trains on the 12 GB RTX 5070. Bigger only if validation says the extraction is too weak.
- **Training happens in two stages** (this is what question 7 was asking: a homemade model has no instruction-following ability, so every task has to be trained in):
  1. Pretrain on general English plus a finance-text corpus so it can read at all.
  2. Fine-tune task heads on labels the pipeline makes automatically:
     - tickers from the aliases table
     - sentiment from the sign of the next-day excess return
     - event type from keyword rules at first
     - Δ from the realized next-day change in each feature's z-score
- **Numbers go in as numbers.** The feature snapshot enters through a learned projection layer, one token per feature, not as text. That's cheaper than writing 6,000 numbers out as words and avoids tokenization noise.
- **How the adjustment works:**
  - Adjusted feature = raw z-score + Δ.
  - Δ is clipped to ±1.0 and applied only for assets whose news_shock is in the top 20% that day ("if it applies"). Otherwise Δ = 0.
  - Raw values stay untouched in the database; the adjusted set is stored as its own feature version.
- **History before news collection starts:** Δ = 0 with a news_available = 0 flag, so the base models can train on 10+ years of numbers now and learn the news-adjusted regime as it accumulates.
- **Injection and manipulation defenses.** Article text is data, never instructions.
  - Strip HTML and hidden text.
  - Keep the source whitelist and per-source caps.
  - Clip Δ.
  - Raise a risk event if one article drives more than 25% of a day's total |Δ|.
  - The keyword tokens fed to the Transformer and CNN #2 come from a closed n-gram vocabulary, so article text can never smuggle in arbitrary tokens.
- **KV cache:** build with a standard KV cache first. The LLM is the only model that generates tokens, so your cache-manipulation experiments slot in here later.

### Validation
1. Walk-forward: train on an expanding window, test the next 3 months, roll forward.
2. Purge and embargo: drop training rows whose 5-day label window overlaps the test period, and skip 5 days after each test block.
3. Out-of-fold stacking: the meta learner's training set is built only from base-model predictions made on data those models never trained on.
4. Metrics:
   - log loss and Brier score per model
   - daily rank IC (Spearman of score vs forward return); a mean IC above 0.03 with a t-stat above 2 is worth keeping
   - backtest Sharpe, Sortino, max drawdown and turnover after costs
5. Costs in every backtest: assume 10 bps per side for equities and 30 bps per side for crypto until you measure real fills.
6. Ablation: drop each base model in turn; any model that doesn't lower the meta learner's out-of-sample loss gets cut.
7. Go-live gate: at least 3 months on Alpaca paper trading, with live drawdown under 1.5× the backtest's worst and realized IC within one standard error of backtest IC.

## Portfolio rules engine
The rules engine is deterministic code, not a model. It turns the meta learner's per-asset score s and conviction c into target weights, checks them against hard limits, and emits at most 8 orders a day with a reason code on each. All percentages below are config values.

### Top-level allocation and cash modes
Default split of NAV: 15% cash, 59.5% equity sleeve, 25.5% crypto sleeve (invested capital split 70/30). Tier budgets the system can't fill with qualifying candidates stay in cash; they are never redistributed to other tiers.

| Mode | Trigger | Cash target | What changes |
|---|---|---|---|
| Normal | Default | 15% | Nothing |
| Opportunity | ≥ 2 candidates with c ≥ 0.75 and s in top 10%, stress score < 1, HMM P(stress) < 0.3 | 10% floor | Extra 5% goes only to those candidates. Must restore 15% within 10 trading days by trimming active tiers |
| Risk-off | HMM P(stress) > 0.6 or stress score > 2 | 25% | Active equity tiers cut 50%, crypto risk sleeve to 0. Core untouched. Exit only after P(stress) < 0.4 for 5 straight days |

### Tier map
| Sleeve / tier | % of sleeve | % of NAV | Slots | Weight per slot (% NAV) |
|---|---|---|---|---|
| Equity core: Anchors | 19% | 11.3% | 2 | 5.65 |
| Equity core: Supports | 29% | 17.3% | 3 | 5.75 |
| Equity active: Leaders | 24% | 14.3% | 4 | 3.57 (sized by conviction) |
| Equity active: Standards | 19% | 11.3% | 5 | 2.26 (sized by conviction) |
| Equity active: Satellites | 9% | 5.4% | 3 | 1.79 (sized by conviction) |
| Asymmetric slot (stock or coin) | — | up to 1.5% | 1 | up to 1.5, funded from cash |
| Crypto core: BTC / ETH / SOL | 40 / 20 / 10% | 10.2 / 5.1 / 2.55% | 3 | fixed |
| Crypto risk | 30% | 7.65% | up to 5 | up to 2.55 |

### Tier eligibility (rebuilt weekly with the screen)
A stock that qualifies for several tiers takes the highest one (Anchor > Leader > Standard > Satellite), and each ticker is held in one tier only.
- **Anchors:**
  - market cap ≥ $200B
  - 252-day vol ≤ 25%
  - beta 0.6–1.1
  - positive FCF in each of the last 4 quarters and FCF margin ≥ 15%
  - net debt / EBITDA ≤ 2
  - quality rank ≥ 0.7

  Pick the top 2 by quality + low-vol composite. Replace an incumbent only after it fails a criterion 2 weeks in a row.
- **Supports**, 3 fixed slots chosen by regime:
  - Bond: SHY by default; IEF in Goldilocks; TLT in Deflation or HMM stress.
  - Commodity: GLD by default; DBC in Reflation. Buy or add only when price ≤ SMA200 or mean-reversion z ≤ −1 (your "bought low" rule); until then the slot's budget waits in cash.
  - Broad ETF: VTI.
  - Switch a support only after a quadrant change holds 2 weeks.
- **Leaders:** c ≥ 0.70, s in top 10% of the universe, momentum rank ≥ 0.6, quality rank ≥ 0.5.
- **Standards:** c ≥ 0.62, s in top 25%.
- **Satellites:** value rank ≥ 0.8 (deep value), c ≥ 0.60, s in top 30%.
- **Crypto risk:** c ≥ 0.65; tradable on Alpaca (check the Assets API each run); 30-day average daily volume ≥ $50M. Temporary BTC, ETH or SOL overweights live here, not in core.
- **Asymmetric slot** (1 extra position, stock or coin), all of these must hold:
  - c ≥ 0.75 and s in the top 2% of the whole universe.
  - Reward-to-risk ≥ 3, where reward = P(up) × upper barrier and risk = P(down) × lower barrier, from the averaged base-model probabilities.
  - Big enough to matter: the asset's 1-week expected move (σ√h) is at least 2× the universe median.
  - A fresh catalyst: an LLM event (earnings, guidance, M&A, regulation, protocol upgrade) in the last 5 days with news_shock in the top 5%.
  - Same liquidity filters as its asset class.
  - Funded from cash: it may take cash below 15%, never below the 10% floor. Risk-off mode closes it.

**Protected holdings.** Anchors, Supports and crypto core are long holds, so no model output can sell or shrink them: conviction drops, vol-target scaling and risk-off cuts don't apply. They change only through the eligibility, regime-switch and band-rebalance rules. Leaders are semi-protected: a model exit needs a full week of weak conviction (below).

### Position sizing
Core tiers are equal-weight per slot. Active tiers and the crypto risk sleeve weight each name by edge divided by volatility:

$$w_i = B_{\text{tier}} \cdot \frac{e_i / \sigma_i}{\sum_j e_j / \sigma_j}, \qquad e_i = \max(0,\ 2c_i - 1)$$

B is the tier budget, c the calibrated conviction and σ the 63-day annualized vol. Cap each name at 1.5× the tier's equal-slot weight; the excess goes to cash.

**Portfolio vol target.** Estimate ex-ante annualized portfolio vol from a 63-day covariance matrix with Ledoit-Wolf shrinkage. If it exceeds 16%, multiply active tiers and crypto risk by k = 14% / σₚ (never below 0.5). Core is never scaled.

### Hard constraints
Every proposed order passes all of these or is clipped or rejected, with the failing rule logged.

| Rule | Limit |
|---|---|
| Single stock | ≤ 8% of NAV |
| Single altcoin | ≤ 2.55% of NAV |
| Sector concentration | ≤ 30% of the equity sleeve per GICS sector (broad ETFs exempt) |
| Redundancy | No new entry with 63-day correlation > 0.85 to a holding in the same tier; keep the higher-c name |
| Liquidity | Order ≤ 1% of 20-day ADV (crypto: ≤ 0.5% of daily volume) |
| Earnings blackout | No new Standards or Satellites within 3 trading days before earnings |
| Minimum trade | Skip any change under 0.25% of NAV |
| Switching hurdle | Replace a holding in a full tier only if c_new − c_worst ≥ 0.08 |
| Hysteresis | Active names enter at their tier threshold, exit only when c < 0.50 on 2 consecutive runs |
| Order cap | ≤ 8 orders per cycle, priority: circuit-breaker exits > stops > band rebalances > new entries by c |
| Round trips | Never sell a stock bought the same session (pattern-day-trader rules apply to margin accounts under $25k) |

### Exits
| Tier | Stop | Time limit | Signal exit |
|---|---|---|---|
| Anchors (protected) | None | None | None from models; leaves only if it fails eligibility 2 weeks running |
| Supports (protected) | None | None | None from models; switches only on a regime change held 2 weeks |
| Leaders | Trailing −25% from peak | None | c < 0.40 on 5 consecutive runs (one full week) |
| Standards | Entry − 2.5×ATR, trailing 2.5×ATR after +1 ATR gain | 63 trading days (min hold 10) | c < 0.50 twice |
| Satellites | Entry − 2×ATR; sell half at +3×ATR, trail the rest at 1.5×ATR | 21 trading days | c < 0.50 twice |
| Asymmetric slot | Entry − 2×ATR; sell half at +3×ATR, trail the rest at 1.5×ATR | 7 days (one horizon) | c < 0.60 once |
| Crypto risk | Entry − 3×ATR | 14 days | c < 0.50 twice |
| Crypto core (protected) | None | None | None from models; band rebalance only |

### Rebalancing bands
- Core positions trade only when weight leaves target ± max(20% of target, 0.5pp of NAV). BTC at a 10.2% target rebalances outside 8.2–12.2%. Trades go back to target, on the Monday cycle only.
- Sleeves rebalance whenever equity drifts more than ±5pp of NAV or crypto more than ±4pp from target, any day.

### Order generation, each cycle
1. Load holdings, NAV, cash, meta outputs, regime probabilities and stress score. Reconcile positions with Alpaca first.
2. Set the cash mode, which fixes the cash target and the active-tier scale.
3. Map candidates to tiers (weekly eligibility + today's c and s).
4. Build target weights: core by rule, active by the sizing formula, crypto core fixed.
5. Apply the vol-target scale.
6. Apply exits; triggered names get target 0.
7. Apply hard constraints; clipped weight goes to cash.
8. Diff targets vs current weights; drop sub-minimum trades; apply the switching hurdle and hysteresis.
9. Rank by priority, keep the top 8, put sells before buys.
10. Simulate the post-trade book; confirm cash ≥ the mode's floor and every constraint holds; write proposed_orders with reason codes.

### Circuit breakers
- NAV down 3% in a day → exits only next cycle.
- Drawdown from peak NAV of 10% → active tiers and crypto risk at 50%. At 20% → stops only, manual reset required.
- Any critical feed (prices, yields, VIX) older than one trading day → skip the cycle.
- Std of the four base models' P(up) above 0.2 on a candidate → halve that entry.
- Any single order above 10% of NAV, or a limit price more than 5% from last trade → reject.

This isn't financial advice; the thresholds are starting points to tune in backtests and paper trading.

## Execution layer
Orders are decided after the close and placed the next session in the time slot the price transformer picks (09:45 ET by default). The earliest slot still avoids the opening auction and comes after a fresh look at overnight news.
1. **Pre-open check, 09:00 ET.** Re-pull overnight news and any 08:30 ET data releases (CPI, jobs, GDP). If the stress score moves by more than 1, or a held name gets a large news_shock, re-run the rules engine before executing.
2. **Limit price from the price transformer.** For a buy, the limit is the 70th-percentile prediction of the session low, which targets roughly a 70% fill rate; sells mirror this with the session high.
3. **Urgency by reason code.** Stop, circuit-breaker and risk-off exits go as marketable limits (ask + 0.1%). New entries go at the model limit as day orders.
4. **15:30 ET sweep.** Unfilled entries are canceled, not chased. Unfilled exits convert to marketable limits.
5. **Crypto runs on the same 09:45 ET clock.** Alpaca crypto supports market, limit and stop-limit orders with gtc and ioc time-in-force (Alpaca crypto docs), so entries go as gtc limits canceled after 24 hours. The executor can read a Level 2 book for crypto (orderbook endpoint); for stocks it uses the latest quote and bars.
6. **Stops live at the broker** as stop or trailing-stop orders (stop-limit for crypto), so they protect positions between daily cycles.
7. **Fill accounting.** Log every fill; measure slippage against both the decision price (prior close) and the arrival price at submission. A weekly report replaces the backtest's assumed costs with measured ones.
8. **Paper first.** All of the above runs against Alpaca's paper endpoint until the go-live gate in the Validation section is met.

## Database schema
One dedicated PostgreSQL database (TimescaleDB extension optional for the time-series tables) holds everything; each of your original "pages" maps to a table or view below. Raw tables are append-only; every run is tied to a run_id so any decision can be replayed.

| Table | Key columns | Written by | Your original page |
|---|---|---|---|
| assets | asset_id, ticker, asset_class, gics_sector, ipo_date, alpaca_tradable | Weekly screen | — |
| universe_weekly | week, asset_id, tier_eligible, factor ranks | Weekly screen | Screened list |
| raw_prices | asset_id, ts, open, high, low, close, volume, source, ingested_at | Collector | Per-data-point history |
| raw_macro | series_id, period, value, released_at, vintage, ingested_at | Collector | Per-data-point history |
| raw_fundamentals | asset_id, fiscal_period, metric, value, filed_at | Weekly collector | Per-data-point history |
| articles | article_id, url_hash, simhash, source, published_at, title, body, tickers[], svm_prob, kept | News pipeline | — |
| article_seen | url_hash, simhash, first_seen | News pipeline | Headlines already seen |
| llm_outputs | run_id, article_id, keywords[], sentiment, event_type, confidence | LLM | Marked section |
| ngram_vocab | version, ngram, chi2, built_at | Weekly job | n-gram training table |
| feature_registry | feature_name, version, formula, window, source_series | Developer | — |
| features | asof, asset_id (null = market), feature_name, value | Feature layer | — |
| model_input_latest (view) | latest market vector + asset matrix | Feature layer | Current viewing page / data pool |
| predictions | run_id, model_name, model_version, asset_id, p_up, p_flat, p_down | Base models | — |
| meta_scores | run_id, asset_id, score, conviction, market_risk | Meta learner | — |
| regime | asof, quadrant, hmm_calm, hmm_normal, hmm_stress, stress_score | Feature layer | — |
| positions | asset_id, qty, avg_cost, tier, opened_at, peak_price, stop_price | Executor (reconciled with Alpaca) | Portfolio |
| portfolio_daily | date, nav, cash, sleeve weights, mode, drawdown | Executor | Portfolio |
| proposed_orders | run_id, asset_id, side, target_w, delta_notional, tier, reason_code, rule_checks (JSON), status | Rules engine | — |
| orders | order_id, alpaca_id, proposed_id, qty, limit_price, submitted_at, status, filled_qty, avg_fill, slippage_bps | Executor | Order history |
| runs | run_id, started_at, finished_at, status, feed_freshness (JSON), model versions | Orchestrator | — |
| risk_events | ts, type, details | Rules engine, executor | — |

## Build order
Build the plumbing and a dumb baseline first, so every model after it has to beat something measurable. Each step is one Claude Code project prompt.
1. Database and collectors with point-in-time storage. Start news collection on day one: text models can only learn from history collected from that day on.
2. Feature layer with the registry and unit tests.
3. Labels, walk-forward harness and cost model.
4. Rules engine driven by a baseline scorer (s = momentum factor rank, c = 0.5 + 0.3 × rank). Backtest it. This tests the engine before any ML exists and sets the IC bar models must clear.
5. Executor on Alpaca paper using the baseline scorer; fills start accumulating for the price transformer.
6. Numeric models (MLP, CNN #1) and the meta learner; keep each only if it beats baseline out-of-sample IC.
7. SVM and LLM news pipeline.
8. Text-dependent models (Transformer, CNN #2) once 6+ months of filtered news exist.
9. Price transformer for limit prices, trained on paper fills, replacing the simple q70 rule.
10. Go-live gate: 3+ months of paper trading meeting the Validation criteria.
