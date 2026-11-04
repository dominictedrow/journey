# Stage 00 — scope, constraints and project contract
**Status:** ACTIVE. **Current task:** 00-04.

- [x] 00-01 Obtain first-round owner's answers to `docs/planning/QUESTIONS.md`; record unanswered points and avoid material assumptions.
- [ ] 00-02 Confirm asset class, initial symbols, trading venue, legal/jurisdictional requirements, brokerage connectivity and data rights.
- [x] 00-03 Choose broker demo API for autonomous paper orders, TradingView Essential for charting; confirm protective-stop requirement; document unresolved numeric controls.
- [ ] 00-03b Set detailed paper mode, desired signal cadence, allowed actions, risk hard-stops and hardware/budget envelope.
- [ ] 00-04 Define quantitative evaluation measures, baselines, held-out data policy, no-go criteria and success criteria.
- [ ] 00-05 Confirm Claude Code filesystem access, GitHub remote, branch policy and approvals.
- [ ] 00-06 Produce requirements spec and acceptance tests; present for explicit sign-off.

**Done when:** approved written scope and risk contract; pending decisions closed or clearly fenced. Next: `01_research.md`. Stage transition requires human approval.


### v3 additional exit gate
- [x] Resolve TradingView-only native paper versus broker-demo API feasibility via `docs/evaluations/TRADINGVIEW_FEASIBILITY.md`.
- [ ] Record pilot futures/forex instruments, operator jurisdiction, feasible broker/data entitlements, and independent hard risk control policy.
- [ ] Capture full-course accessibility details. Do not claim any videos watched.

### V4 next checkpoint work
- [ ] `00-04a` Compare provider-specific OANDA practice vs futures demo API costs, eligibility, real-time/historical market data rights, stop handling, reconnect behavior.
- [ ] `00-04b` Formally propose small pilot universe and evidence-based order/data budget; owner must approve symbols/providers.
- [ ] `00-04c` Draft quantitative risk-default proposal with portfolio-wide breakers and protective stop acknowledgment/partial-fill rules; request approval before paper trading.
- [ ] `00-05a` Initialize *new private* repo only after GitHub authorization; verify remote privacy and record SHA, branch and push.
- [ ] `00-06a` Present research/evaluation protocol and request Stage 00 signoff.

### v5 checkpoint
- [x] Capture owner choices: US jurisdiction, research futures costs, GitHub private repo `trading`, request recommended risk limits.
- [x] Draft researched provider + pilot comparison in `docs/evaluations/PROVIDER_EVALUATION.md`.
- [x] Draft numeric limits as **unapproved** defaults in `docs/planning/RISK_POLICY_PROPOSAL.md`.
- [x] Prepare Stage 00 dossier in `docs/planning/STAGE_00_SIGNOFF.md`.
- [ ] Owner approves research-only Stage 01 transition (do not automatically infer).
- [ ] Explicit permission for any GitHub creation/push, costs, account access or demo orders remains separate.

### Owner sign-off update (2026-10-18 PDT)
Stage 00 research design **APPROVED**; transition to Stage 01 **AUTHORIZED**. Execution credentials, spending, GitHub, numerical risk parameters and all paper/live orders are **not authorized**. See `docs/planning/STAGE_00_SIGNOFF.md`.
