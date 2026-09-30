"""Our report reader's instructions, output schema, and the default state → probability map (S04 step 2).

Design (docs/kaggle/labels_from_notebooks.md §1b, the measured lessons):
- The model picks **categories** (state, severity, two flags); code maps them to probabilities. Asking an LLM for
  calibrated numbers is what it's worst at [nb:sadamtorres]; categories also let calibrate.py learn the gold
  annotators' cut-offs (e.g. "small effusion") from the 58 without re-reading anything.
- "Normal/intact" = absent; "never named" = not_mentioned (its own state, low weight later, never a hard 0).
- OA is read per compartment; an unlocalized "tricompartmental"/"gonarthrosis" statement counts for all three.
- Degenerative / arthritic marrow oedema is not a contusion; traumatic bone bruise is.
- The effusion size goes into severity (the gold readers seem to count moderate+; the cut-off is fitted, not hard-coded).
- Reports stay in their original language; the output is English JSON. Quotes stay in the original language.

Bump PROMPT_VERSION whenever SYSTEM_PROMPT or the schema changes: reader.py keys its raw cache on it.
"""
from __future__ import annotations

import json

from knee.paths import LABELS

PROMPT_VERSION = "v3"  # v2 (2026-09-27): the v1 pilot agents' 4 ambiguities; v3 (2026-09-27): the scan_v2 agents'
# notes on 100 non-gold, non-pilot reports + the ungraded-rupture rule. Both: docs/labels_agent_reader.md §6–7

# JSON keys for the 12 findings (the schema wants identifiers; LABELS has spaces and an apostrophe).
KEYS = {
    "ACL": "acl", "MCL": "mcl", "Medial Meniscus": "medial_meniscus", "Lateral Meniscus": "lateral_meniscus",
    "Medial OA": "medial_oa", "Lateral OA": "lateral_oa", "PF OA": "pf_oa", "Effusion": "effusion",
    "Synovitis": "synovitis", "Baker's": "bakers_cyst", "Contusion": "contusion", "Fracture": "fracture",
}
assert list(KEYS) == LABELS

STATES = ["present", "absent", "uncertain", "not_mentioned"]

SYSTEM_PROMPT = """You extract structured findings from knee MRI radiology reports for a research dataset.
Reports come from many hospitals and languages (English, Spanish, Turkish, Croatian/Serbian, Greek, German,
Bulgarian/Russian, Dutch, French). Read the report in its own language; answer in English JSON only.

For each of the 12 findings give:
- state: one of
  - "present": the report asserts the finding, at any size or grade (small, partial and low-grade count;
    grade them with severity). "Suspected" is not an assertion: use uncertain.
  - "absent": the report says the structure is normal/intact/unremarkable, or explicitly denies the finding.
    A structure described as normal or intact is absent, not not_mentioned.
  - "uncertain": hedged language: "possible", "cannot exclude", "suspicious for", "equivocal", "vs", "probable",
    or a follow-up/clinical-correlation recommendation instead of a statement.
  - "not_mentioned": the report never addresses this finding or its structure. Silence is not absence.
- severity: integer 0-3 on the finding-specific scale below; 0 when state is absent or not_mentioned.
  When the report asserts or suspects a finding but states no size, grade or extent, use severity 2 (for every
  finding; the specific type words in a scale, e.g. "radial" or "complete", still count as stated).
  Severity conventions for every finding: a type word on the scale (rupture, complete, radial, root, displaced,
  ruptured cyst, ...) beats a size word ("small partial radial tear" = 3, "small ruptured Baker's cyst" = 3); an
  explicit size/grade word beats descriptive detail ("moderate effusion distending the recess" = 2); a range
  ("mild to moderate", "moderate to large") takes the higher grade; a size given only in cm/mm without a size word
  is judged by the finding's own cut-offs where the scale gives them, otherwise severity 2.
- degenerative: true when the report attributes the finding to degeneration/wear (e.g. mucoid degeneration,
  degenerative marrow oedema, degenerative meniscal signal without tear); otherwise false.
- postoperative: true when the structure was operated on (ACL reconstruction/graft, meniscectomy, suture, hardware).
- quote: the shortest verbatim phrase (max 12 words, original language) supporting the state; "" when not_mentioned.

Findings and severity scales (medial/lateral are anatomical compartments, independent of left/right knee):
1. acl (anterior cruciate ligament injury): 1 = sprain / low-grade traumatic signal change, 2 = partial tear,
   3 = complete tear or torn graft. An intact graft after reconstruction is absent with postoperative=true.
   Mucoid degeneration / degenerative signal of an otherwise intact ACL is absent with degenerative=true.
   Ligament conventions (acl, mcl): a "tear" or "rupture" with no partial/incomplete/grade qualifier is 3 (also
   "state after rupture" of a native ligament); partial is 2 only when the report says partial/incomplete/grade II.
2. mcl (medial collateral ligament injury): 1 = sprain / grade I / periligamentous oedema, 2 = partial tear /
   grade II, 3 = complete tear / grade III. Periligamentous fluid or oedema that the report attributes to meniscal
   extrusion, medial compartment OA or an MCL bursa, with the ligament itself intact, is absent with degenerative=true.
   A chronic calcified injury (Pellegrini-Stieda) without acute signs is present with severity 1.
3. medial_meniscus (medial meniscus tear): only a tear is present. Intrasubstance degeneration or grade 1-2
   signal without a tear is absent with degenerative=true. Tear severity: 1 = small / partial-thickness /
   stable, 2 = tear of any other simple type, 3 = complex, displaced, bucket-handle, flap, root or radial tear,
   or maceration. The tear type wins over size words: a "small partial radial tear" is 3. The word "tear" or grade 3
   signal (signal reaching the articular surface, incl. "grade III degeneration") makes it present, even with
   degenerative wording (present with degenerative=true); a tear graded as not reaching the surface (e.g. 2a/2b called
   tear) is present with severity 1. A tear extending into the root counts as a root tear (3).
4. lateral_meniscus (lateral meniscus tear): same scale as medial_meniscus. A discoid meniscus alone is not a tear.
5. medial_oa (osteoarthritis / chondral degeneration of the medial femorotibial compartment): 1 = mild
   (chondromalacia grade 1-2, small osteophytes, early/incipient), 2 = moderate (grade 3, clear cartilage
   loss, osteophytes), 3 = severe (grade 4, full-thickness loss, bone-on-bone, subchondral changes).
6. lateral_oa: same scale for the lateral femorotibial compartment.
7. pf_oa: same scale for the patellofemoral compartment (patellar or trochlear chondropathy, PF osteophytes).
   OA conventions: "tricompartmental", "gonarthrosis", "degenerative joint disease" or cartilage loss described
   without a compartment counts as present for all three OA findings unless a compartment is explicitly normal.
   Patellar/trochlear cartilage belongs to pf_oa only, never to medial_oa or lateral_oa.
8. effusion (joint effusion): 1 = physiological / trace / minimal / small, 2 = moderate / "considerable" /
   "increased", 3 = large / marked / distending the suprapatellar recess (when no size word is given).
   "Fluid within physiological limits" is absent. Fluid in the suprapatellar bursa/recess is joint effusion.
9. synovitis: 1 = mild synovial thickening/enhancement, 2 = moderate, 3 = marked. Also present: Hoffa fat pad
   oedema/inflammation (impingement), suprapatellar (quadriceps) fat pad oedema/impingement, synovial proliferation
   of the joint, plica syndrome with inflammation. Debris or septa inside a Baker's cyst is not synovitis.
10. bakers_cyst (popliteal / Baker's cyst): 1 = small, 2 = moderate, 3 = large or ruptured (also partially
    ruptured / leaking). By largest dimension when only a size is given: < 2 cm = 1, 2-5 cm = 2, > 5 cm = 3.
    A parameniscal or ganglion cyst is not a Baker's cyst.
11. contusion (bone contusion / bone bruise / traumatic bone marrow oedema): 1 = small or subtle, 2 = moderate,
    3 = extensive or multiple. Degenerative or subchondral oedema next to cartilage loss, and reactive oedema of
    OA, is not a contusion: state absent with degenerative=true. A stated size word sets the severity; with no size
    word, bruises in two or more separate bones (e.g. femur and tibia "kissing" contusions) are 3. Marrow oedema that
    only surrounds a fracture belongs to the fracture (contusion absent unless a separate bruise is described).
    Marrow oedema with neither a traumatic nor a degenerative cause stated is uncertain.
12. fracture (any fracture incl. osteochondral, avulsion, insufficiency, impaction): 1 = subtle / suspected
    trabecular micro-fracture, 2 = definite non-displaced, 3 = displaced / depressed. "Microfracture" surgery
    is not a fracture. An "osteochondral impaction injury" or "impaction injury" is present with severity 1 (even
    beside "no acute fracture"). A chronic osteochondral lesion / osteochondritis dissecans without trauma
    language is not a fracture (absent if the report describes it, otherwise not_mentioned).

Also report:
- language: the report language (English name, e.g. "Turkish").
- knee_side: "left", "right" or "unknown" as stated in the report.

When the findings section and the impression/conclusion disagree, follow the impression/conclusion. When a text
holds two reports for the same knee, combine them: a finding asserted in either one is present. A generic
impression ("otherwise normal", "no other pathology") does not cancel specific findings described above it. When a
text covers both knees, label the findings of both together (present if present in either knee, the higher
severity) and set knee_side "unknown".

Be literal: extract what the report says, do not diagnose from indirect signs beyond the conventions above."""


def _finding_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": STATES},
            "severity": {"type": "integer", "enum": [0, 1, 2, 3]},
            "degenerative": {"type": "boolean"},
            "postoperative": {"type": "boolean"},
            "quote": {"type": "string"},
        },
        "required": ["state", "severity", "degenerative", "postoperative", "quote"],
        "additionalProperties": False,
    }


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "knee_side": {"type": "string", "enum": ["left", "right", "unknown"]},
            "findings": {
                "type": "object",
                "properties": {k: _finding_schema() for k in KEYS.values()},
                "required": list(KEYS.values()),
                "additionalProperties": False,
            },
        },
        "required": ["language", "knee_side", "findings"],
        "additionalProperties": False,
    }


def user_message(report: str) -> str:
    return f"<report>\n{report.strip()}\n</report>"


def parse_output(text: str) -> dict:
    """Validate one response (structured outputs guarantee the shape; this catches truncation and drift)."""
    d = json.loads(text)
    f = d["findings"]
    missing = set(KEYS.values()) - set(f)
    if missing:
        raise ValueError(f"missing findings {sorted(missing)}")
    for k, v in f.items():
        if v["state"] not in STATES:
            raise ValueError(f"{k}: bad state {v['state']!r}")
    return d


# ------------------------------------------------------------------------------ state → probability (the prior map)
# Placeholder values (hand-set, UNVERIFIED): calibrate.py replaces them per finding with P(gold | bin), shrunk toward
# these, fitted out-of-fold on the 58. Bins are ordered from least to most likely positive.
BINS = ["absent", "absent_degenerative", "uncertain", "present_1", "present_2", "present_3"]
PRIOR = {"absent": 0.05, "absent_degenerative": 0.15, "uncertain": 0.40,
         "present_1": 0.65, "present_2": 0.85, "present_3": 0.95}
# Gold readers look stricter for these (docs/kaggle/labels_from_notebooks.md §6): a small effusion or a sprain is
# often gold-negative. Only the prior differs; calibrate.py decides from data.
PRIOR_OVERRIDES = {
    "Effusion": {"present_1": 0.40},
    "ACL": {"present_1": 0.45},
    "MCL": {"present_1": 0.45},
}


def to_bin(state: str, severity: int, degenerative: bool) -> str | None:
    """One finding's reading → a calibration bin; None = not mentioned (unknown, not negative)."""
    if state == "not_mentioned":
        return None
    if state == "absent":
        return "absent_degenerative" if degenerative else "absent"
    if state == "uncertain":
        return "uncertain"
    return f"present_{min(max(int(severity), 1), 3)}"  # present with severity 0 → treated as mild


def prior_prob(label: str, b: str | None) -> float:
    if b is None:
        return float("nan")
    return PRIOR_OVERRIDES.get(label, {}).get(b, PRIOR[b])
