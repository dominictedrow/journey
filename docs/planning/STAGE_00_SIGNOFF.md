# Stage 00 research-design sign-off — APPROVED

## Ready for approval as *design direction*
- US brokerage jurisdiction; private GitHub repo name `trading`; create only on explicit account authorization.
- Claude Code in VS Code on Mac, later x86-64 mini PC Linux runtime; buy nothing yet.
- First pilot studies: MES futures and EUR/USD spot FX; long/short, intraday, possible risk-controlled overnight holds.
- TradingView Essential charts; independent broker demo adapter for automated paper orders; do not rely on native TradingView Paper Trading as automated order gateway.
- OANDA practice as first FX adapter feasibility test; local simulator for MES until futures API/data costs and access are validated. Tradovate conditional only.
- Independent protective stop required with every position; fail-closed on unresolved order, feed or account state; local SVM baseline vs deterministic indicators; RNN a later experimental option.
- Full publicly available courses for Joovier and Swaggy C to be catalogued and analyzed with video-level coverage and timestamp provenance; do not claim watched content before access/review.

## Not yet authorized / outstanding independent gates
- **GATE A: APPROVED 2026-10-18 PDT.** Owner authorized Stage 01 research only. No broker login or orders permitted.
- **GATE B (broker access):** Confirm provider/regulatory eligibility, data license, account age/identity rules, and approve any account creation, deposits, paid plan, API subscription or credentials access. No automatic authorization by Stage 00 approval.
- **GATE C (quantitative risk):** Owner approves numbers in `docs/planning/RISK_POLICY_PROPOSAL.md` after paper-equity and instrument-specific min-size/cost checks; may revise limits.
- **GATE D (GitHub):** Owner explicitly authorizes creation of `trading` as a **private** repo and first branch push. Verify visibility before pushing; never expose credentials or recorded course material.
- **GATE E (Stage 06):** Separate consent before activating automated *demo* orders and verify independent protective stops and kill switch.
- **GATE F (Stage 07):** Separate written live trading authorization after robust out-of-sample evidence, risk review, account/permissions checks and live-specific failure testing.

## Completion criteria for Stage 00
Design scope and instrument candidates documented; unresolved funding, risk and integration decisions are bounded by clear gates; `START_HERE.md` state consistent with on-disk repository; owner approves Stage 01 transition. Stage 00 **research design is signed off** by the owner in the present conversation; unresolved independent gates remain open.

**Owner authorization recorded:** “I approve the Stage 00 research design: MES/EURUSD pilot research, OANDA practice feasibility evaluation, local MES simulator, and transition to Stage 01 creator course research. I do not authorize account creation, GitHub operations, spending, or activation of paper/live orders.”
