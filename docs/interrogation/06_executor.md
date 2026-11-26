# Questioning: Phase 6 · Executor (Alpaca paper)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/06_executor.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Execution layer

1. **D-003:** whole shares only, so broker-side GTC stops work, or fractional with daily re-armed DAY stops?
2. **Equity stops:** fixed stops as `stop` orders and Leaders as `trailing_stop` (GTC; auto-cancelled after 90 days, so the executor renews them). OK?
3. **Crypto stops (D-004):** use stop_limit, with the limit how far below the stop? Trailing done by the executor each cycle?
4. **Pre-open check:** what counts as a "large news_shock" on a held name (top 5%?) before the price transformer exists?
5. **The machine has to be awake at 09:00, 09:45 and 15:30 ET.** Is the Mac on during market hours until the Windows move?
6. **Reconciliation:** if Alpaca positions differ from the DB, halt the cycle and alert, or trust Alpaca?
7. Should the weekly slippage report be Markdown in `reports/`?
