# Report labels read by Claude Code agents (instead of the API)

_Decided 2026-09-27 (Dom): Dom has a Claude subscription but no API credits, so our report reader runs as Claude Code
agents instead of Message Batches. Everything except the reading itself is code: `src/knee/labels/agent_reader.py`._

This file is the method record for the write-up (winner obligations: a reproducible method description) and the
runbook for redoing a run with another Claude Code session.

## 1. What is the same as the API reader, and what differs

| | API reader (`reader.py`) | agent reader (`agent_reader.py`) |
|---|---|---|
| Instructions | `prompt.SYSTEM_PROMPT` (v1) | the same text, verbatim, in `AGENT_INSTRUCTIONS.md` |
| Output | JSON schema enforced by the API | compact line per report (`AGENT_FORMAT`), expanded + validated against the same schema at ingest |
| Unit of work | 1 request = 1 report, isolated context | 1 agent = 1–4 shards of 25 reports, one context |
| Model | chosen per run (`--model`) | the model the Claude Code session runs; recorded per study (`--model`, default `claude-opus-5-5`) |
| Sampling settings | effort/thinking set per request | not controllable; whatever Claude Code uses |
| Raw records | `raw/<custom_id>.json`, `mode="batch"` | the same layout, `mode="agent"`, `usage=None`, plus `shard` and `agent_format` |
| Downstream | `reader parse` → `build_labels --claude-run <run>` | identical |

Known differences that matter for the write-up:
- **One context reads 25–100 reports.** An agent may drift or carry context from one report to the next. The rules
  forbid copying answers between reports; the pilot's gold agreement is the check that it didn't hurt.
- **Not bit-reproducible.** Neither route is (the API also samples). A re-run should reproduce the *numbers* (gold
  macro AUC within the bootstrap CI), not the exact labels. Keep the raw files of the run that trained the submitted
  model: they are the labels of record.
- **No token counts.** A subscription doesn't report per-call usage, so `cost` shows 0 for agent runs.

## 2. Safeguards

- **Blinding.** A shard holds only `{"id": "r"+sha1(uid)[:24], "report": text}`, shuffled (seeded), so a gold study
  can't be spotted by its UID or position. The agent is told to open no file but its instructions and shards
  (`train.csv` carries the gold labels). The instructions don't mention that some studies are gold.
- **No code labeling.** The agent must read each report; writing regex/keyword labelers is forbidden by the rules in
  the instructions (the method claim is "an LLM reads each report").
- **Validation at ingest.** Every line is parsed, expanded, and checked (state enum, severity 0–3, flag types, known
  finding keys, id belongs to the shard). Rejected or missing reports are logged in `ingest_log.json` and go into
  new shards on the next `export`. Nothing is hand-edited.

## 3. Runbook (any machine with the repo, `configs/paths.yaml` and `train.csv`; no GPU, no API key)

```
export PYTHONPATH=src
python -m knee.labels.agent_reader export --split pilot         # 100 reports (58 gold + 42 random) → 4 shards
python -m knee.labels.agent_reader prompt s0000                 # the exact message for one agent
#  → start one Claude Code agent per shard (general-purpose subagent, or a fresh `claude` session)
#    and give it exactly that message; nothing else.
python -m knee.labels.agent_reader ingest                        # validate out/ → raw/
python -m knee.labels.agent_reader status
python -m knee.labels.reader parse --run-name agent_v1
python -m knee.labels.build_labels --version pilot_agent --claude-run agent_v1   # gold AUC in docs/labels.md

python -m knee.labels.agent_reader export --split full           # the remaining ~4,300 reports → ~173 shards
#  → dispatch agents with `prompt sNNNN sNNNN ...` (4 shards = 100 reports per agent worked well), ≤ ~10 at a time
python -m knee.labels.agent_reader ingest && python -m knee.labels.agent_reader export --split full  # re-shard failures
python -m knee.labels.reader parse --run-name agent_v1
python -m knee.labels.build_labels --version v1 --claude-run agent_v1
```

`export` is resumable: it skips studies that already have a valid raw record and studies sitting in a shard that has
no output yet. To redo a run from scratch, use a new `--run-name` (e.g. `agent_v2`) in every command.

**Dispatching from a Claude Code session.** The orchestrating session only runs the commands above and sends each
agent the output of `prompt`. It must not read the shards or the agents' output itself (it has seen `train.csv`
and the gold labels in its own context, so it can't be a blinded reader).

## 4. Runs

| run | date | model | reports | agents × shards | gold macro AUC (reader alone) | notes |
|---|---|---|---|---|---|---|
| agent_v1 pilot | 2026-09-27 | claude-opus-5-5 | 100 | 4 × 1 | 0.853 [0.813, 0.888] prior map; 0.797 LOO-calibrated | 100/100 valid, 0 rejects; ~78k agent tokens and ~2.5 min per 25-report shard. Blend public + agent (prior map) 0.889 = public blend 0.889: no measurable gain on gold. LOO calibration lowers the score (see §5) |
| agent_v2 pilot | 2026-09-27 | claude-opus-5-5 | 100 (same shards) | 4 × 1 | 0.864 [0.827, 0.898] prior map | 100/100 valid. Paired bootstrap v2 − v1 = +0.012 [+0.001, +0.024]. Blend public + agent_v2 0.890 vs public 0.889: still no measurable gain. Ties pilkwang (0.864), below steven_v1 (0.878) |
| scan_v2 | 2026-09-27 | claude-opus-5-5 | 100 non-gold, non-pilot (`--split scan`) | 4 × 1 | n/a (no gold) | prompt v2; 100/100 valid. Purpose: agents' notes on rule gaps → prompt v3 (§7) |
| agent_v3 pilot | 2026-09-27 | claude-opus-5-5 | 100 (same shards) | 4 × 1 | 0.864 [0.827, 0.897] prior map | 100/100 valid. Paired v3 − v2 = −0.000 [−0.009, +0.007]. Blend public + agent_v3 0.891 vs public 0.889 (paired +0.002 [−0.005, +0.010]): still no measurable gain |

## 5. Pilot findings (2026-09-27)

- **Reading quality is on par with the public readers.** Prior-map agent reader 0.853 vs pilkwang (Opus 5 via API)
  0.864, steven_v1 0.878, laymond 0.865, lixin 0.835; all CIs overlap. Per finding (prior map): ACL 0.956, MCL 0.941,
  MM 0.950, LM 0.847, MOA 0.880, LOA 0.853, PFOA 0.884, Effusion 0.727, Synovitis 0.716, Baker's 0.937,
  Contusion 0.776, Fracture 0.765.
- **Agents were consistent with each other.** The effusion severity mix is similar in all 4 shards, and the ambiguities
  they reported were resolved the same way. Effusion is weak because gold splits 14 negative / 19 positive on
  "small effusion" (present_1): that's the gold readers' policy, not reading noise.
- **LOO calibration hurts on coarse bins** (0.853 → 0.797). Every study in a bin gets the same value except for the
  leave-one-out shift, and that shift lowers a bin's value exactly for its positives. So ties get broken against
  the truth. This is a flaw in `calibrate.py`'s gold scoring, not in the reading. **Fixed:** 5-fold cross-fitting
  (seeded; `oof="loo"` still available) scores 0.808–0.824 over 5 seeds, and 2-fold 0.820–0.831. That's still below
  the prior map because calibration is monotone per finding: it can only merge bins or add fold-to-fold noise, never
  improve a finding's ranking. **So compare readers on the prior map;** the calibrated score only measures noise.
  Whether the training targets should use the calibrated values (absolute scale, e.g. small effusion ≈ 0.4) is a
  separate question for Dom.
- **Prompt v2 candidates** (from agent notes): no rule for severity when the size/grade isn't stated (one agent used 2
  for effusion/synovitis/OA and 1 for Baker's); "small partial radial tear" fits meniscus severity 1 and 3;
  contradictory findings vs impression (follow the impression?); osteochondral impaction injury vs fracture.

## 6. Prompt v2 (2026-09-27)

Changes, all taken from the v1 agents' own notes (not from looking at gold answers, so the re-pilot stays an honest
comparison):
1. Asserted or suspected finding with no size, grade or extent → severity 2 for every finding.
2. Meniscus: tear type beats size words ("small partial radial tear" = 3).
3. Fracture: "osteochondral impaction injury" = present, severity 1, even beside "no acute fracture". A chronic
   osteochondral lesion / OCD without trauma language is not a fracture.
4. If findings and impression disagree, follow the impression. Two reports for the same knee in one text: a finding
   asserted in either one is present.

Run `agent_v2` = the same 100 pilot studies, same shards, prompt v2.

### Re-pilot result

Per finding (prior map, v1 → v2): Effusion 0.727 → 0.806, Medial OA 0.880 → 0.911, PF OA 0.884 → 0.899, Contusion
0.776 → 0.805; the rest are within ±0.01. The gain is real on these 100 (paired CI excludes 0) but slightly
optimistic: the v2 rules came from the agents' notes on these same reports, though never from gold answers.
Side effect: an ungraded ACL "rupture" now gets severity 2 (the scale's "partial"); a v3 would say that an unqualified
"rupture/tear" of a ligament is 3.

**Against the go/no-go rule Dom set ("go full only if it beats the public readers"): not met.** Our reader ties
the best comparable public reader and doesn't lift the gold blend. The full run (~4,300 reports, ~13M agent tokens)
is on hold until Dom decides.

## 7. Prompt v3 (2026-09-27)

**Where the rules came from.** v2's rules came from notes on the pilot reports themselves, so its gain was slightly
optimistic. For v3, 4 agents read 100 **non-gold reports outside the pilot** (`export --split scan`, run `scan_v2`,
prompt v2). Each was asked for up to 10 notes on gaps in the rules (`prompt --scan`). v3 fixes the points where the
agents *disagreed with each other*, because those add ranking noise, plus the ungraded-rupture rule from §6.
No v3 rule came from a gold answer or a pilot report.

Changes (all in `prompt.SYSTEM_PROMPT`):
1. Severity conventions for every finding: a type word beats a size word (e.g. "small ruptured Baker's" = 3); an
   explicit grade word beats descriptive detail; a range takes the higher grade; a size in cm/mm only → the finding's
   cut-offs (Baker's: < 2 / 2–5 / > 5 cm), else 2. (The scan agents split lower/higher on ranges.)
2. ACL/MCL: an unqualified "tear"/"rupture" = 3; partial only when stated. ACL mucoid degeneration of an intact ligament
   = absent, degenerative (like the meniscus rule; v1/v2 had it as severity 1). MCL periligamentous fluid attributed
   to extrusion/OA/bursa with an intact ligament = absent, degenerative; Pellegrini-Stieda = present 1.
3. Meniscus: "tear" or grade-3 signal (reaching the surface, incl. "grade III degeneration") = present; a "tear"
   graded 2a/2b = present 1; tear extending into the root = root tear (3).
4. Effusion: "considerable/increased" = 2; suprapatellar bursa fluid = effusion.
5. Synovitis: suprapatellar fat pad oedema counts; debris inside a Baker's cyst doesn't.
6. Contusion: the size word sets severity; with no size word, two or more bones = 3; oedema only around a fracture belongs
   to the fracture; oedema with no stated cause = uncertain.
7. A generic impression ("otherwise normal") doesn't cancel specific findings; bilateral texts combine both knees
   (higher severity, knee_side unknown).

Domain choices in 2 and 5 are **conventions, not verified against the gold readers' policy** (UNVERIFIED).

### Re-pilot result

Macro AUC is unchanged (0.864 vs 0.864; paired −0.000 [−0.009, +0.007]). Per finding (prior map, v2 → v3): Synovitis
0.709 → 0.734, MCL 0.941 → 0.946, LM 0.847 → 0.852, Medial OA 0.911 → 0.894, Contusion 0.805 → 0.791, MM 0.950 → 0.946;
the rest are unchanged. With 58 gold studies, per-finding moves of ±0.02 are within noise.

**Reading:** v3 is at least as good as v2, and its number is the more honest one (its rules never saw the pilot). **v3 stays
the current prompt.** The go/no-go rule for the full run ("beats the public readers") is still not met: the blend gain is
+0.002 [−0.005, +0.010]. The reader's quality now looks capped by the gold readers' own policy (e.g. small effusion,
synovitis) rather than by the prompt. Further prompt rounds are unlikely to show up on 58 studies.
