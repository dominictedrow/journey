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
| D-019 | 2026-10-18 | P2.3 validation design approved as written (timsTOF queries; V-lib/V-pool/V-novel + scaffold slice; ik14-grouped; equal weights). Sizes 1000/2000/1000/1000, seed 2026 | Dom | Sizes agent-chosen; only 2,509 enveda mols have other-lib spectra |
| D-020 | 2026-10-18 | Drop [2M…] adducts from training (13.4% of kept rows); ablate in P5 | Dom | No test spectra use them |
| D-021 | 2026-10-19 | Delete stale Windows repo copy (only commit baf9d88, already on main; data/ identical in WSL) | Dom | Empty folder left until the Claude Code session that lived there closes |
| D-022 | 2026-10-19 | Final submissions selectable = 2 (Dom confirmed) | Dom | Choosing them stays gate 10 |
| D-023 | 2026-10-19 | Gate 7: upload private dataset casmi26-bundle; gate 6: Submission #1 (library baseline) | Dom | Public LB 0.150 |
| D-024 | 2026-10-19 | P4.1 pool sources: train + COCONUT 2.0 + LOTUS + PubChem filtered (150–1,200 Da, CHNOPS+F/Cl/Br/I, neutral, single component, no isotopes). P4.2 reroute: packed Morgan fp stored only for train/COCONUT/LOTUS; PubChem stores SMILES/mass/ik14/formula/prior, fps computed per ppm window at inference | Dom | Full PubChem fps ~25–35 GB is too big for Kaggle RAM |
| D-025 | 2026-10-19 | P4.2 reroute: PubChem keyed by plain InChIKey14 (tautomer ik14 kept for train/COCONUT/LOTUS and the 467 PubChem chunks already done); inference re-keys top candidates by tautomer ik14 to dedup | Dom | Tautomer ik14 ran ~1,150/s -> ~26 h+ for PubChem; plain key ~0.2 ms. ≤3.6% PubChem dup keys |
| D-026 | 2026-10-20 | V-novel kept as defined (option a); report V-novel MRR with and without external pool sources; revisit a kind-3 V-novel at P8 | Dom | 97.6% of V-novel answers are in COCONUT/LOTUS/PubChem, so with pool_v1 it measures kind 2 |
| D-027 | 2026-10-20 | Skip P4.4 upload and P4 submission; tag p4-pool, start P5; pool ships with the first P5 submission | Dom | Pool ranking without fingerprints ~0 on kind 2 (honest val weighted 0.292 vs ~0.150 LB expectation unchanged) |
| D-028 | 2026-10-21 | Gates 7+6 approved for Submission #3: create dataset casmi26-pool (6.1 GB), version casmi26-bundle with p5_ft model.pt, push kernel and submit | Dom | P5 encoder honest val weighted 0.311 (V-pool 0.0286); expected LB ~0.17 |
| D-029 | 2026-10-22 | Gate 1 approved for P6.1/P6.2: molecule encoder MLP (2048 fp + 10 element counts -> 1024 -> 256, L2) + 512->256 projection on the P5 pooled vector; InfoNCE (learnable temperature) over in-batch + 32 mined hard negatives, focal fp loss kept as auxiliary; init p5_ft; 30k steps all libs + 10k enveda (~2 GPU-h) | Dom | Hard negatives mined on CPU (only the 5070 is visible in WSL); stereoisomer decoys with identical fp dropped |
| D-030 | 2026-10-22 | Rule zero (D-029 negative spec): P6 round 2 uses model-mined negatives: per train molecule the top 64 wrong pool_v1 candidates in its +-5 ppm window by p6_ft blend score (cos + 4 x soft Tanimoto), truth keys and identical-fp decoys dropped; fine-tune p6_ft on enveda libs 10k steps lr 5e-5 (mirrors p6_ft) | Dom | P6.4: truth in window 97%, top-100 67%, top-1 0.5%; training NCE ~0.005 says the 32 random same-formula negatives are too easy |
| D-031 | 2026-10-22 | Repo moved ~/casmi -> ~/envenda (done by Dom; recorded after the fact); scripts and plan paths updated | Dom | Gate 8, logged at S15 catch-up |
| D-032 | 2026-10-22 | Rule zero P6.5: skip Submission #4, tag p6-contrastive, start P7; P6 ckpt ships with the P7 submission. Push bd19688 approved | Dom | Val +0.002 weighted (0.311 -> 0.313) is below LB noise; R@100 0.67 vs R@1 0.005 favours a reranker |
| D-033 | 2026-10-22 | P7.1 gate: in-house RDKit fragmenter (1-2 bond cuts, +-2 H); MetFrag dropped. Push tags p2-data..p5-encoder approved | Dom | 2.5 ms/cand (~15 s for top-50 x 400 on 4 cores); truth beats random window decoys 93%; no JRE needed |
