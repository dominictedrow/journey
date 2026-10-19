# DECISIONS.md — approvals and standing decisions

| ID | Date | Decision | Approved by | Notes |
|----|------|----------|-------------|-------|
| D-001 | 2026-10-18 | Build mode only; optimize for prize money, not learning | Dom | Learning happens in a separate dashboard |
| D-002 | 2026-10-18 | Libraries, external data and pretrained models allowed | Dom | Within competition rules |
| D-003 | 2026-10-18 | Physics/chemistry tools allowed as features (RDKit, formula decomposition, neutral losses, mass defect) | Dom | |
| D-004 | 2026-10-18 | Try fragmentation trees | Dom | Implemented as P7 reranker |
| D-005 | 2026-10-18 | Features cached for training; computed on the fly at inference | Dom | Parity test required |
| D-006 | 2026-10-18 | Train on both the local GPU and Kaggle GPUs | Dom | 5060 for side jobs |
| D-007 | 2026-10-18 | Data streamed / memmapped, never fully loaded | Dom | 16 GB RAM |
| D-008 | 2026-10-18 | No limit on GPU hours | Dom | >4 h single runs still gated |
| D-009 | 2026-10-18 | Session ends at ~30% context | Dom | |
| D-010 | 2026-10-18 | Leaderboard in its own file, referenced from START-HERE | Dom | |
| D-011 | 2026-10-18 | Submit whenever a new model exists | Dom | Revisit once submission limit is known |
| D-012 | 2026-10-18 | Git commits per session, tags per phase | Dom | |
| D-013 | 2026-10-18 | Experiment tracking with W&B + runs.csv | Dom | |
| D-014 | 2026-10-18 | No model-freeze date; train as long as useful | Dom | Hard deadline Dec 14 still applies |
| D-015 | 2026-10-18 | Transformer encoder core; InChIKey14-grouped val; 25-candidate output | Dom | From brainstorm |
| D-016 | 2026-10-18 | "Big change" = any deviation from PLAN.md (plan bugs, inefficiencies, wrong assumptions, better routes); agent must ask before rerouting | Dom | CLAUDE.md rule zero; specific gates kept as always-ask cases |
| D-017 | 2026-10-18 | Install WSL Ubuntu-24.04, move repo to ~/casmi (ext4); Windows copy kept until P0 exit | Dom | P0.1 gate |
| D-018 | 2026-10-18 | P1.4 option B: keep formula decomposition + subformula scoring, add M+1/M isotope-ratio score from in-MS2 isotope peaks (P2.2, P2.4, P7.2) | Dom | MS2 has M+1 peak in 21% train / 50% test |
