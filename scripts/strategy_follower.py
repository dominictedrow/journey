"""Submitted agent's strategy follower: combined livestock + melon build.

Plain top-level agent(obs) function -- no factory/closure -- so it reads as
a flat, directly-editable action layout and stays picklable for
harness.run_matches()'s ProcessPoolExecutor.

v2: generalized the crew-assignment layer (was hardcoded to exactly 2
livestock hands + 1 melon hand) so ANIMAL_MIX and hand counts can scale --
needed to chase a higher money target than the original 6-animal/19-melon-
tile build could reach. Sizing rationale (see RULES.md's Price Function
table): each glut-prone product's price floors once cumulative season sales
pass roughly its `T` throughput constant. At this build's yield rates,
wool's T=105 and milk's T=122 support far more animals before crashing than
melon's T=300 supports tiles (melon's higher yield/tile/day burns through
its bigger T just as fast) -- so the scale-up lever is more livestock, not
more melon. NE land ($1k) is bought to give melon room now that a bigger
herd claims more of NW.

AB-tested via test_gen.py (see ab_rounds/round4-15*.json) across two passes:

Pass 1 (round4-7): scaling the herd past ~14 animals backfired at first
(round5's 18/21-animal configs crashed to $16k-69k -- ongoing daily wheat
OPEX outpaces revenue before the bigger roster pays off), and a single melon
hand became the bottleneck once the herd grew (round6: 2 melon hands beats
1). round7 (n=30) confirmed winner: 82,793 combined mean, beating the
original 6-animal/1-melon-hand build's 60,970 by +36%. Also fixed a real bug
found along the way: wheat purchases were gated on `money > 0`, so a big
enough one-time spend (animals + land) could zero the cash out and then
*permanently* block wheat buying, starving the whole herd to death (2 unfed
days = animal escapes, unrecoverable) with no way to recover -- wheat is now
funded first, before animals/land get whatever's left.

Pass 2 (round8-15, chasing a 95k target): added nearest-need routing (was
walking to the first needy tile in index order, not the closest one) and an
urgency tier (an unfed animal/unwatered-young-melon is one missed day from
irrecoverable loss, so it now always outranks routine harvest/care) -- a real
correctness fix, but its measured effect was within noise once confirmed at
scale (n=30), not a big lever on its own. Land-purchase timing/priority and
animal-purchase-pacing knobs were added and swept; none beat the original
"greedy, animals first" defaults. Re-sweeping herd size found a real (if
narrow) sweet spot at 17 animals (11 COW + 6 SHEEP, 6 hands) -- 18+ reliably
regressed again, this time because a bigger roster's upfront cost crowds out
land purchase entirely (confirmed: `unlocked_quadrants` stayed `['NW']` the
whole game in the 18-animal case). The one genuine additional win was sell
price gating (`SELL_MIN_PRICE_FRAC`): CONSOLIDATED_FINDINGS flagged this as
"worth testing" for glut-fragile goods and phase 1 never tried it at this
production volume -- holding melon/milk/wool instead of dumping them while
the price is beaten down below 40% of base beat dump-immediate by a
consistent, confirmed margin. n=50 confirm: 86,253 combined mean.

Pass 3 (round16-21, crop diversification, still chasing 95k): generalized
melon's tile-action into `_crop_action(crop, max_yield_day, ...)`, shared by
MELON/WHEAT/CARROT, and generalized the crew-assignment into `_CREW_PLAN` (a
flat hand-slot -> (job, tile-group) list) so any number of job types can
share the hire pool. Also found and fixed a real crash along the way:
standing on a still-locked tile (a plain "LOCKED" string, not a dict or None)
called `.get()` on it and raised AttributeError -- latent in melon's tile
pool too (it also reaches into NE) but never triggered until wheat's spare-
tile pool reached deeper into still-locked territory. Also found that HIRE
competes with every BUY_* for the same 10-orders/turn cap, and with hand
counts approaching double digits HIRE alone could fill it, silently dropping
every buy (and every sell) that turn -- fixed by giving each purchase
category its own hour instead of bundling them all into hour 0 (only HIRE
actually needs hour 0, since hands are hired once for the whole day).

The crop economics looked compelling on paper -- wheat barely reacts to glut
(T=400, above_target=0.2) so growing our own feed instead of buying it
looked like near-free money, and carrot's price pool is entirely separate
from melon's -- but none of it held up under confirmation. Small-n screens
(n=15) showed wheat self-farming beating baseline by several thousand
(round16/17), but every n=30+ confirm (round18/19/20) put wheat-alone,
carrot-alone, and the no-crops baseline within noise of each other
(81k-86k), and wheat+carrot combined together was outright worse and
unstable (round16: bimodal, stdev ~43k, sometimes $0). The marginal hand's
hire cost plus the crop-tending overhead apparently roughly cancels out
whatever wheat/carrot revenue they add at this scale -- this game's
per-episode variance (stdev commonly $7k-45k depending on config) is large
enough that a lever has to be worth several thousand dollars before n=30-50
can tell it apart from noise, and neither crop cleared that bar. Final
confirmed number, unchanged from pass 2: 86,810 combined mean (n=40,
consistent with pass 2's 86,253 at n=50) -- didn't reach the 95k target; see
the chat writeup for what a genuinely structural change (SW/SE land, a
different crew architecture) would need to try next.

Pass 4 (AB-confirmed regression, reverted): three changes drawn from
report.md's replay analysis of the real top-10 leaderboard (992+ episodes
across 3 distinct strategy families), not from phase-1 sweeps against
"random"/"starter" -- see CLAUDE.md's caveat on why that distinction
matters.

(1) BUY_LAND now targets 3 quadrants (NW+NE+SW, generalized from the old
NE-only BUY_NE_LAND flag) -- every top-10 family buys a 3rd quadrant by
day 9-11 using the same "retry every day until affordable" idiom this file
already used for NE, so this is that same proven idiom extended, not new
design. SE ($4k) is skipped for now -- no crew/tile plan built for it yet.

(2) Melon stops being replanted after MELON_EXIT_DAY (day 10) -- every
top-10 family abandons melon by day ~20 rather than fighting its permanent
glut (zero town demand, above_target=3.6), and since melon matures in
exactly 10 days, day-10 is the latest planting that still finishes near
that window. Already-growing melon is left to mature normally (no point
discarding a sunk seed cost); once a tile empties post-exit it gets WHEAT
instead, using the same hand/tile assignment with no crew restructuring --
wheat's price climbs all season on staple demand (RULES.md Price Function
table) rather than gluting, so this doubles as the "wheat as exit vehicle"
late-game pattern report.md found across the whole leaderboard, just
triggered by a day threshold instead of literally being end-of-season.

(3) The herd grows from 17 to 23 animals (11->15 COW, 6->8 SHEEP) with
livestock hands 6->7 (N_HIRES 8->9), chasing report.md's observed top-10
hand counts -- capped well short of that 12-14 range by a real constraint
report.md doesn't surface: HIRE cost is `fib(hires_today)` and hands drop
at end of day, so it's not a one-time cost, it's paid in full *every
single day* (RULES.md "Hiring" section). Going from 8 to 12 daily hires
looked like a proportionate scale-up but is actually a 7x jump in daily
hire spend ($54/day -> $376/day, $1,620 -> $11,280 over 30 days) --
confirmed by a direct sizing sweep (not just theory) that N_HIRES=12
reliably bankrupts the farm to $0 with 0 hands by day ~19, every seed
tried. The cliff sits between N_HIRES=9 ($88/day, safe across every seed
tried) and N_HIRES=10 ($143/day, reliably bankrupts) -- N_HIRES=9 is the
ceiling this pass uses. This probably means top-10 agents ramp hiring up
over several days rather than paying the full target's fib-cost from day
0 the way this file's HIRE block always has (report.md §4 hints at this:
Family C "only crossing 10 hands around day 12-13" despite a field-high
peak of 14) -- a gradual hiring ramp is the natural next lever if 9 hires
turns out to be the real ceiling under a fixed-target policy, but it's a
new mechanism, not a sizing tweak, so it's out of scope for this pass.
SW is unused by 23 animals (fits entirely in NW's 25 tiles) -- it exists
as forward-compatible overflow room (see _LIVESTOCK_POOL) for whenever a
future pass raises the target past 25, not exercised by this pass's
numbers. A 10-seed sanity check (not a real AB confirm -- see 1st.md's
"Measurement problem" on why n=10 can't settle this) put COW=15/SHEEP=8/
hands=7 at a 72,931 mean vs the original 17-animal/6-hand config's 69,315
over the same 10 seeds, with no bankruptcies either side -- directionally
fine, not a confirmed win; that's what the next thorough AB pass is for.

The thorough pass: `harness.compare_paired` (seeded, n=30) against a
reconstructed pass-3 baseline (COW=11/SHEEP=6, LIVESTOCK_HANDS=6,
LAND_TARGET_QUADRANTS=2, MELON_EXIT_DAY effectively off) found pass 4
*worse*, not better -- mean diff -18,518 vs `random` (stdev 36,065, n=30;
a real signal, not noise) and a statistical tie vs `starter` (+102, stdev
34,522). The 10-seed sanity check's apparent +3,616 edge didn't survive
proper n. Root cause not isolated (the three changes were bundled and
reverted together, matching `leader_clone`'s own lesson in 1st.md: copying
the leaderboard's output numbers -- herd size, quadrant count -- doesn't
automatically transfer to an economy that didn't produce them), but the
bigger herd's daily hire/feed OPEX and the 3rd quadrant's $2k up-front
land cost are the leading suspects, both competing with the same cash pool
BUY_LAND and HIRE already contend for. Reverted to pass 3's values below;
if a future pass wants this lever again, isolate and AB-confirm each of
the three changes independently rather than bundling.
"""

try:
    import shared_features as _watcher_features
    import watcher_model as _watcher_model
except (ImportError, AttributeError):
    _watcher_features = None
    _watcher_model = None

# ---- Constants (the "easy to change" knobs) --------------------------------

ENABLE_DEADLINE_PLANNER = True        # step 1 layer; False preserves the legacy unit chunks exactly
ENABLE_ADAPTIVE_TARGETS = True        # step 2 layer; intended cumulatively with the deadline planner
ENABLE_TOWN_WATCHER = True            # step 3 layer; exact unlocked-shop demand signals
ENABLE_OPPONENT_WATCHER = True        # step 4 layer; six-day public-farm supply forecast
WATCHER_BACKEND = "rules"             # set to "linear_svm" after exporting a compatible watcher_model.py
TOWN_EXPOSURE_MULTIPLIER = 2.0        # production capacity relative to exact observed town demand
TOWN_WATCHER_START_DAY = 10           # preserve the profitable opening through melon's first harvest
TOWN_MAX_CROP_REDIRECT = 1            # best bounded township candidate from the n=60 tuning expansion
OPPONENT_SIGNAL_WEIGHT = 4.0          # confirmed crop-only forecast strength (round27/28)
OPPONENT_EXPANSION_THRESHOLD = 0.0    # require non-negative competitive case before buying more land
DEADLINE_CAPACITY_MARGIN = 0.20       # leave this fraction of daily unit-turns for travel/recovery
HIRING_RAMP_MODE = "workload"        # fixed, calendar, workload, or cash_workload
HIRING_CEILING = 8                    # conservative phase-4 ceiling; never exceeds configured crew
HIRING_PAYBACK_DAYS = 4.0             # minimum remaining productive days per marginal hire
LIVESTOCK_SPECIALISTS = 0             # phase-5 candidate: stable hand slots devoted to compact zones

ANIMAL_MIX = {"COW": 11, "SHEEP": 6}   # pass 3 value -- pass 4's 15/8 AB-confirmed worse (see docstring), reverted
ANIMAL_COST = {"COW": 400, "SHEEP": 500}
LIVESTOCK_HANDS = 6                    # pass 3 value -- pass 4's 7 (N_HIRES 8->9) AB-confirmed worse, reverted
MELON_HANDS = 2                        # splitting the 19 melon tiles across 2 units beat 1 hand at this scale --
                                        # a single melon hand couldn't service the tiles fast enough (round5/6/7);
                                        # a 3rd hand was tried and reliably broke the economics (round11/13) -- not used
MAX_MELON_TILES = 19                   # melon's own T=300 / 0.55 yield-per-tile-day / 30 days ~= 18 tiles -- keep near that;
                                        # expanding to 26 tiles under NE tested statistically tied, not worth the complexity (round14);
                                        # unchanged in pass 4 -- MELON_EXIT_DAY caps season exposure instead of tile count
MELON_YIELD_DAY = 10                   # melon: first_yield_day == max_yield_day == 10 (RULES.md Object Types)
MELON_EXIT_DAY = 999                   # pass 3 value (effectively off, day never reaches this) -- pass 4's day-10 exit
                                        # AB-confirmed worse alongside its other two changes (see docstring), reverted
LAND_TARGET_QUADRANTS = 2              # pass 3 value: NW+NE only (was BUY_NE_LAND) -- pass 4's 3rd quadrant (NW+NE+SW)
                                        # AB-confirmed worse alongside its other two changes (see docstring), reverted
LAND_MIN_DAY = 0                       # earliest day to attempt BUY_LAND -- 0 = as soon as affordable (swept, best default)
MAX_ANIMAL_BUY_PER_DAY = None          # None = uncapped -- buy every affordable animal every day (swept, best default)
SELL_MIN_PRICE_FRAC = 0.4              # hold melon/milk/wool instead of dumping while price is below 40% of base --
                                        # confirmed win over dump-immediate (round12/14/15) once production volume is high enough to move the price.
                                        # The sell gate is intentionally STATIC: the watcher SVM never modulates it. An
                                        # earlier experiment wired a model-driven `_svm_sell_frac` here; it was reverted
                                        # and the watcher now only influences production/expansion signals, not sales.
SALE_POLICY_MODE = "legacy"            # legacy or product_specific; phase-7 candidates remain gated
SALE_MIN_PRICE_FRAC_BY_PRODUCT = {      # independent routine-sale gates; decay toward zero before liquidation
    "WHEAT": 0.0, "CARROT": 0.35, "TOMATO": 0.45, "STRAWBERRY": 0.50,
    "MELON": 0.40, "EGG": 0.35, "MILK": 0.40, "WOOL": 0.45,
    "FERTILIZER": 0.0,
}
SALE_MAX_BATCH_BY_PRODUCT = {           # None means sell the full eligible quantity
    "WHEAT": 3, "FERTILIZER": 3, "CARROT": None, "TOMATO": None,
    "STRAWBERRY": 8, "MELON": 8, "EGG": 8, "MILK": 8, "WOOL": 8,
}
SALE_OVERFLOW_QUANTITY = 90             # cheap sale is preferable to shed-cap output loss
SALE_THRESHOLD_DECAY_START_DAY = 22
SALE_ROUTINE_ORDER_BUDGET = 3           # purchases are queued first; routine sales cannot consume every slot
LAND_EXPANSION_MODE = "legacy"          # legacy or conditional; phase-8 candidates remain gated
LAND_UTILIZATION_THRESHOLD = 0.80       # occupied fraction required before buying quadrant three
LAND_MIN_REPAY_DAYS = 8                 # conservative remaining-season payback window
LAND_OPERATING_RESERVE = 500            # retained in addition to adaptive feed/cash reserve
CROP_TRANSITION_MODE = "off"          # off or fixed; candidates stay gated until phase-6 promotion
MELON_LAST_SEED_DAY = 999              # stop buying melon seed independently from planting
MELON_LAST_PLANT_DAY = 999             # existing melon always matures; only vacant slots transition
TRANSITION_EMPTY_POLICY = "hold"       # hold, wheat, or strawberry after melon exit
STRAWBERRY_START_DAY = 10
STRAWBERRY_TILES = 0
STRAWBERRY_LAST_PLANT_DAY = 19          # first yield is age 10; day 19 is the last productive start
LATE_WHEAT_START_DAY = 999
LATE_WHEAT_TILES = 0
WHEAT_LAST_PLANT_DAY = 25               # peak harvest at age 4 before day-29 liquidation
_BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
               "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200}
_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
_CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
_CROP_MAX_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
_CROP_DAILY_YIELD = {"WHEAT": 0.80, "CARROT": 0.75, "TOMATO": 0.33, "STRAWBERRY": 0.24, "MELON": 0.55}
_ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
_ANIMAL_DAILY_YIELD = {"GOOSE": 1.0, "COW": 0.5, "SHEEP": 1.0 / 3.0}
_PRODUCT_DAILY_YIELD = dict(_CROP_DAILY_YIELD)
_PRODUCT_DAILY_YIELD.update({"EGG": 1.0, "MILK": 0.5, "WOOL": 1.0 / 3.0})
_ANIMAL_COST_ALL = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_ANIMAL_CAP = {"GOOSE": 17, "COW": 11, "SHEEP": 6}
_ANIMAL_FIRST_YIELD = {"GOOSE": 4, "COW": 8, "SHEEP": 6}
_ANIMAL_INTERVAL = {"GOOSE": 1, "COW": 2, "SHEEP": 3}
_ONGOING_CROP_INTERVAL = {"TOMATO": 1, "STRAWBERRY": 2}
_PRODUCTS = tuple(_BASE_PRICE)
_THROUGHPUT = {"WHEAT": 400, "CARROT": 450, "TOMATO": 200, "STRAWBERRY": 100,
               "MELON": 300, "EGG": 332, "MILK": 122, "WOOL": 105}
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

# ---- Crop diversification knobs --------------------------------------------
# Wheat's price barely reacts to glut (T=400, above_target=0.2 -- RULES.md
# Price Function table) unlike melon/carrot/livestock products, so growing
# our own feed instead of buying it is close to free money once tiles exist
# for it; carrot is a genuinely separate revenue pool (its own price/T) that
# doesn't compete with melon's for the same throughput ceiling.
WHEAT_TILES = 0                        # self-grow feed wheat (and sell any surplus) instead of buying all of it; 0 = off
WHEAT_HANDS = 0
CARROT_TILES = 0                       # secondary cash crop, separate price pool from melon; 0 = off
CARROT_HANDS = 0
CROP_MAX_YIELD_DAY = dict(_CROP_MAX_DAY)

ANIMAL_STRUCTURE = "PASTURE"
SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5)}

# NW quadrant, backward scan from the farmer's shed-adjacent spawn (4, 4) --
# same order as the proven livestock finding's tile pick. SW backward scan
# (mirrored from its own shed-adjacent corner (4, 5)) is pure overflow room:
# with ANIMAL_MIX <= 25 this pool is never reached and behavior is identical
# to pre-pass-4 (pure NW), same pattern melon already uses for its own
# NW-free + NE overflow below.
_NW_BACKWARD = [(x, y) for y in range(4, -1, -1) for x in range(4, -1, -1)]
_SW_BACKWARD = [(x, y) for y in range(9, 4, -1) for x in range(4, -1, -1)]
_LIVESTOCK_POOL = _NW_BACKWARD + _SW_BACKWARD


def _livestock_type_sequence():
    seq = []
    for animal, count in ANIMAL_MIX.items():
        seq += [animal] * count
    return seq


LIVESTOCK_TYPES = _livestock_type_sequence()
LIVESTOCK_POSITIONS = _LIVESTOCK_POOL[: len(LIVESTOCK_TYPES)]

# Crop pool: NW forward scan minus livestock's tiles, then NE forward scan
# (only reachable once LAND_TARGET_QUADRANTS unlocks it -- locked tiles are
# passable but PLANT no-ops on them, so listing them early is harmless, not
# incorrect).
# Melon claims the first MAX_MELON_TILES; wheat and carrot split whatever's
# left, in that order.
_claimed = set(LIVESTOCK_POSITIONS)
_NW_FREE = [(x, y) for y in range(0, 5) for x in range(0, 5) if (x, y) not in _claimed]
_NE_FORWARD = [(x, y) for y in range(0, 5) for x in range(5, 10)]
_ALL_FREE = _NW_FREE + _NE_FORWARD
MELON_POSITIONS = _ALL_FREE[:MAX_MELON_TILES]
_SPARE = _ALL_FREE[MAX_MELON_TILES:]
WHEAT_POSITIONS = _SPARE[:WHEAT_TILES]
CARROT_POSITIONS = _SPARE[WHEAT_TILES:WHEAT_TILES + CARROT_TILES]


def _chunk_indices(n_items, n_groups):
    """Split range(n_items) into n_groups contiguous, near-equal chunks."""
    if n_groups <= 0:
        return []
    base, extra = divmod(n_items, n_groups)
    groups, start = [], 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


# Crew split: farmer + LIVESTOCK_HANDS hands round-robin the livestock tiles
# in even chunks; MELON_HANDS/WHEAT_HANDS/CARROT_HANDS each round-robin their
# own crop's tiles. _CREW_PLAN flattens this into one job per hand-slot, in
# hire order, so the hands loop below just indexes into it -- adding a new
# job type only means appending another job-group list here.
LIVESTOCK_GROUPS = _chunk_indices(len(LIVESTOCK_POSITIONS), 1 + LIVESTOCK_HANDS)
MELON_GROUPS = _chunk_indices(len(MELON_POSITIONS), MELON_HANDS)
WHEAT_GROUPS = _chunk_indices(len(WHEAT_POSITIONS), WHEAT_HANDS)
CARROT_GROUPS = _chunk_indices(len(CARROT_POSITIONS), CARROT_HANDS)

_CREW_PLAN = (
    [("LIVESTOCK", g) for g in LIVESTOCK_GROUPS[1:]]  # group 0 is the farmer, handled separately
    + [("MELON", g) for g in MELON_GROUPS]
    + [("WHEAT", g) for g in WHEAT_GROUPS]
    + [("CARROT", g) for g in CARROT_GROUPS]
)
N_HIRES = len(_CREW_PLAN)


def _step_toward(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx:
        return "EAST"
    if x > tx:
        return "WEST"
    if y < ty:
        return "SOUTH"
    if y > ty:
        return "NORTH"
    return None


def _is_shed_adjacent(pos):
    return pos in SHED_ADJACENT


def _nearest_step(pos, positions):
    """Step toward the closest of `positions` (Manhattan), not just the
    first one found -- every wasted step here is an action turn not spent
    watering/feeding/harvesting.
    """
    if not positions:
        return None
    ux, uy = pos
    target = min(positions, key=lambda p: abs(p[0] - ux) + abs(p[1] - uy))
    return _step_toward(pos, target)


# ---- Livestock unit action (adapted from agent_lib.py::_unit_action, ------
# simplified since both COW and SHEEP use PASTURE -- no COOP branching) -----

def _livestock_action(pos, inv, tiles, shed, idxs):
    ux, uy = pos
    cur_idx = next((i for i in idxs if LIVESTOCK_POSITIONS[i] == (ux, uy)), None)
    if cur_idx is not None:
        atype = LIVESTOCK_TYPES[cur_idx]
        x, y = LIVESTOCK_POSITIONS[cur_idx]
        tile = tiles[y][x]
        if tile is None:
            return ["BUILD_PASTURE"]
        if tile.get("kind") == ANIMAL_STRUCTURE and not tile.get("animal"):
            if inv.get(atype, 0) > 0:
                return ["PLACE", atype]
        elif tile.get("animal") == atype:
            if not tile["fed_today"]:
                if inv.get("WHEAT", 0) > 0:
                    return ["FEED"]
            elif tile["yield_units"] > 0:
                return ["HARVEST"]
            elif tile["fertilizer_available"]:
                return ["COLLECT_FERTILIZER"]
            elif not tile["cared_today"]:
                return ["CARE"]

    if _is_shed_adjacent((ux, uy)):
        # Stock wheat for the WHOLE assigned chunk in one visit so the unit
        # never has to shuttle back mid-route to restock.
        wheat_needed = len(idxs)
        if wheat_needed > inv.get("WHEAT", 0) and shed.get("WHEAT", 0) > 0:
            n = min(shed["WHEAT"], wheat_needed - inv.get("WHEAT", 0))
            if n > 0:
                return ["PICKUP", "WHEAT", n]
        empty_needed = {}
        for i in idxs:
            atype = LIVESTOCK_TYPES[i]
            x, y = LIVESTOCK_POSITIONS[i]
            t = tiles[y][x]
            if isinstance(t, dict) and t.get("kind") == ANIMAL_STRUCTURE and not t.get("animal"):
                empty_needed[atype] = empty_needed.get(atype, 0) + 1
        for atype, cnt in empty_needed.items():
            need = cnt - inv.get(atype, 0)
            avail = shed.get(atype, 0)
            if need > 0 and avail > 0:
                return ["PICKUP", atype, min(need, avail)]

    if not _is_shed_adjacent((ux, uy)):
        unfed_remaining = 0
        missing_animal = False
        for i in idxs:
            atype = LIVESTOCK_TYPES[i]
            x, y = LIVESTOCK_POSITIONS[i]
            t = tiles[y][x]
            if isinstance(t, dict) and t.get("animal") == atype:
                if not t["fed_today"]:
                    unfed_remaining += 1
            elif isinstance(t, dict) and t.get("kind") == ANIMAL_STRUCTURE and not t.get("animal"):
                if inv.get(atype, 0) == 0:
                    missing_animal = True
        if unfed_remaining > inv.get("WHEAT", 0) or missing_animal:
            target = min(SHED_ADJACENT, key=lambda p: abs(p[0] - ux) + abs(p[1] - uy))
            step = _step_toward((ux, uy), target)
            if step:
                return [step]

    # Two priority tiers, nearest-first within each: an unfed animal is one
    # missed day from irrecoverable escape, so feeding always outranks
    # harvest/care/place -- but among several feed-needy tiles, still walk
    # to the closest one first, not just the first in index order.
    feed_needy = []
    other_needy = []
    for i in idxs:
        atype = LIVESTOCK_TYPES[i]
        x, y = LIVESTOCK_POSITIONS[i]
        t = tiles[y][x]
        if t is None:
            other_needy.append((x, y))
        elif isinstance(t, dict) and t.get("kind") == ANIMAL_STRUCTURE and not t.get("animal"):
            if inv.get(atype, 0) > 0:
                other_needy.append((x, y))
        elif isinstance(t, dict) and t.get("animal") == atype:
            if not t["fed_today"] and inv.get("WHEAT", 0) > 0:
                feed_needy.append((x, y))
            elif t["yield_units"] > 0:
                other_needy.append((x, y))
            elif t["fertilizer_available"]:
                other_needy.append((x, y))
            elif t["fed_today"] and not t["cared_today"]:
                other_needy.append((x, y))
    step = _nearest_step((ux, uy), feed_needy) or _nearest_step((ux, uy), other_needy)
    if step:
        return [step]
    return ["PASS"]


# ---- One-time crop unit action (adapted from engine.py::_tile_action) -----
# Shared by MELON, WHEAT, and CARROT -- all three are the same "plant / dig
# weed / water until max_yield_day / harvest" one-time-crop mechanic, just
# with different max_yield_day and seed price. Ongoing crops (tomato,
# strawberry) would need a different function -- not used here.

def _crop_action(crop, max_yield_day, pos, day, tiles, idxs, positions):
    ux, uy = pos
    cur_idx = next((i for i in idxs if positions[i] == (ux, uy)), None)
    if cur_idx is not None:
        x, y = positions[cur_idx]
        tile = tiles[y][x]
        if tile is None:
            return ["PLANT", crop]
        # A crop tile pool can reach into NE before land is bought (locked
        # tiles are passable, so a unit can end up standing on one); "LOCKED"
        # is a plain string, not a dict, so it must be excluded here or the
        # .get() calls below raise AttributeError.
        if isinstance(tile, dict):
            if tile.get("kind") == "WEED":
                return ["DIG"]
            if tile.get("kind") == "PLANT":
                age = day - tile["planted_day"]
                if age >= max_yield_day and tile["yield_units"] > 0:
                    return ["HARVEST"]
                if not tile["watered_today"]:
                    return ["WATER"]

    # Watering an unripe plant outranks harvest/plant/dig -- two consecutive
    # unwatered days turns it into a weed (irrecoverable), same reasoning as
    # livestock's feed-first tier. Nearest-first within each tier.
    water_needy = []
    other_needy = []
    for i in idxs:
        x, y = positions[i]
        t = tiles[y][x]
        if t is None:
            other_needy.append((x, y))
        elif isinstance(t, dict) and t.get("kind") == "WEED":
            other_needy.append((x, y))
        elif isinstance(t, dict) and t.get("kind") == "PLANT":
            age = day - t["planted_day"]
            if age >= max_yield_day and t["yield_units"] > 0:
                other_needy.append((x, y))
            elif not t["watered_today"]:
                water_needy.append((x, y))
    step = _nearest_step((ux, uy), water_needy) or _nearest_step((ux, uy), other_needy)
    if step:
        return [step]
    return ["PASS"]


def _melon_action(pos, day, tiles, idxs):
    # Past MELON_EXIT_DAY, empty tiles in this same pool get WHEAT instead --
    # no crew restructuring needed, the hand just keeps working its assigned
    # tiles. Already-growing melon is unaffected: HARVEST only fires once the
    # tile's real yield_units > 0 (set by the engine, not by max_yield_day
    # here), so passing WHEAT's shorter max_yield_day for a tile that's
    # actually still-growing melon can't trigger a premature harvest -- it
    # just keeps falling through to WATER, same as before.
    if day < MELON_EXIT_DAY:
        return _crop_action("MELON", MELON_YIELD_DAY, pos, day, tiles, idxs, MELON_POSITIONS)
    return _crop_action("WHEAT", CROP_MAX_YIELD_DAY["WHEAT"], pos, day, tiles, idxs, MELON_POSITIONS)


def _wheat_crop_action(pos, day, tiles, idxs):
    return _crop_action("WHEAT", CROP_MAX_YIELD_DAY["WHEAT"], pos, day, tiles, idxs, WHEAT_POSITIONS)


def _carrot_action(pos, day, tiles, idxs):
    return _crop_action("CARROT", CROP_MAX_YIELD_DAY["CARROT"], pos, day, tiles, idxs, CARROT_POSITIONS)


# ---- Adaptive runtime targets ----------------------------------------------

def _fib_hire_cost(n_hands):
    a, b, total = 1, 1, 0
    for _ in range(n_hands):
        total += a
        a, b = b, a + b
    return total


def _marginal_hire_cost(n_hands):
    """Cost of the next hand after ``n_hands`` have already been hired."""
    a, b = 1, 1
    for _ in range(n_hands):
        a, b = b, a + b
    return a


def _daily_hand_target(day, estimated_load, money, cash_reserve, remaining_days):
    """Bounded phase-4 hiring target using observation-only information."""
    workload = max(0, int((estimated_load + 19.1) // 19.2) - 1)
    ceiling = min(N_HIRES, HIRING_CEILING)
    if HIRING_RAMP_MODE == "fixed":
        return ceiling
    if HIRING_RAMP_MODE == "calendar":
        # Replay-family shape: four hands in the opening, five around day 6,
        # then the conservative experimental ceiling from day 9 onward.
        replay_target = 4 if day < 6 else 5 if day < 9 else ceiling
        return min(ceiling, replay_target)
    target = min(ceiling, workload)
    if HIRING_RAMP_MODE != "cash_workload":
        return target

    spendable = max(0.0, money - cash_reserve)
    affordable = 0
    spent = 0
    while affordable < target:
        marginal = _marginal_hire_cost(affordable)
        # A hand supplies at most 24 actions/day. Value the work conservatively
        # at wheat's base price and require time for the hire to repay itself.
        useful_actions = min(24.0, max(0.0, estimated_load - affordable * 19.2))
        if (spent + marginal > spendable or remaining_days < HIRING_PAYBACK_DAYS
                or useful_actions * _BASE_PRICE["WHEAT"] < marginal * HIRING_PAYBACK_DAYS):
            break
        spent += marginal
        affordable += 1
    return affordable


def _adaptive_targets(obs, me, tiles, prices, signals=None):
    """Return bounded targets derived from workload, cash, and live payback."""
    day = obs["day"]
    remaining_days = max(0.0, 30.0 - day - obs.get("hour", 0) / 24.0)
    wheat_price = max(1.0, prices.get("WHEAT", _BASE_PRICE["WHEAT"]))
    crop_counts = {crop: 0 for crop in _SEED_COST}
    animal_counts = {animal: 0 for animal in _ANIMAL_PRODUCT}
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop in crop_counts:
                crop_counts[crop] += 1
            if animal in animal_counts:
                animal_counts[animal] += 1

    crop_scores = {}
    replant_cutoffs = {}
    for crop, seed_cost in _SEED_COST.items():
        daily_gross = _CROP_DAILY_YIELD[crop] * prices.get(crop, _BASE_PRICE[crop])
        cycle_days = max(1, _CROP_MAX_DAY[crop])
        daily_profit = daily_gross - seed_cost / cycle_days
        payback = seed_cost / daily_profit if daily_profit > 0 else 999.0
        replant_cutoffs[crop] = max(0, int(30 - _CROP_FIRST_YIELD[crop] - payback))
        watcher_factor = 1.0
        if signals is not None:
            watcher_factor += 0.35 * signals["product_attractiveness"].get(crop, 0.0)
        crop_scores[crop] = daily_profit * max(0.5, min(1.35, watcher_factor)) if day <= replant_cutoffs[crop] else -1.0

    product_tiles = dict(crop_counts)
    open_crop_slots = max(0, MAX_MELON_TILES - sum(crop_counts.values()))
    ranked_crops = sorted(crop_scores, key=lambda name: (-crop_scores[name], name))
    if signals is not None and day >= TOWN_WATCHER_START_DAY and ranked_crops:
        profit_leader = ranked_crops[0]
        existing_redirects = sum(count for crop, count in crop_counts.items() if crop != profit_leader)
        redirect_budget = max(0, TOWN_MAX_CROP_REDIRECT - existing_redirects)
        town_ranked = sorted(
            (crop for crop in crop_scores if crop != profit_leader),
            key=lambda name: (-signals["product_attractiveness"].get(name, 0.0), -crop_scores[name], name),
        )
        for crop in town_ranked:
            if crop_scores[crop] <= 0 or open_crop_slots <= 0 or redirect_budget <= 0:
                continue
            exposure = signals["recommended_limits"]["product_exposure"].get(crop, 0)
            add = min(open_crop_slots, redirect_budget, max(0, exposure - product_tiles[crop]))
            product_tiles[crop] += add
            open_crop_slots -= add
            redirect_budget -= add
    for crop in ranked_crops:
        if crop_scores[crop] <= 0 or open_crop_slots <= 0:
            continue
        room = open_crop_slots
        product_tiles[crop] += room
        open_crop_slots -= room

    animal_scores = {}
    for animal, product in _ANIMAL_PRODUCT.items():
        daily_profit = _ANIMAL_DAILY_YIELD[animal] * prices.get(product, _BASE_PRICE[product]) - wheat_price
        payback = _ANIMAL_COST_ALL[animal] / daily_profit if daily_profit > 0 else 999.0
        watcher_factor = 1.0
        if signals is not None:
            watcher_factor += 0.35 * signals["product_attractiveness"].get(product, 0.0)
        animal_scores[animal] = daily_profit * max(0.5, min(1.35, watcher_factor)) if payback < remaining_days else -1.0

    animal_targets = dict(animal_counts)
    open_animal_slots = max(0, len(LIVESTOCK_POSITIONS) - sum(animal_counts.values()))
    for animal in sorted(animal_scores, key=lambda name: (-animal_scores[name], name)):
        room = max(0, _ANIMAL_CAP[animal] - animal_targets[animal])
        if signals is not None:
            product = _ANIMAL_PRODUCT[animal]
            exposure = signals["recommended_limits"]["product_exposure"].get(product, 0)
            room = min(room, max(0, exposure - animal_targets[animal]))
        add = min(open_animal_slots, room) if animal_scores[animal] > 0 else 0
        animal_targets[animal] += add
        open_animal_slots -= add

    target_animals = sum(animal_targets.values())
    target_crops = sum(product_tiles.values())
    estimated_load = target_animals * 4.0 + target_crops * 1.75
    cash_reserve = sum(animal_counts.values()) * wheat_price * 2
    daily_hands = _daily_hand_target(
        day, estimated_load, me.get("money", 0), cash_reserve, remaining_days
    )
    if signals is not None and signals.get("backend") == "linear_svm":
        labor_score = signals.get("labor_expansion_pressure", 0.0)
        labor_delta = 1 if labor_score > 0.25 else -1 if labor_score < -0.25 else 0
        daily_hands = max(0, min(N_HIRES, daily_hands + labor_delta))
    # A conditional land experiment deliberately keeps production targets
    # separate from the purchase decision.  Its stricter utilization/layout
    # gate below decides whether the configured extra quadrant is justified.
    if LAND_EXPANSION_MODE == "conditional":
        quadrants = LAND_TARGET_QUADRANTS
    else:
        quadrants = min(LAND_TARGET_QUADRANTS, max(1, (target_animals + target_crops + 24) // 25))
    if signals is not None and ENABLE_OPPONENT_WATCHER:
        quadrants = min(quadrants, signals["recommended_limits"]["quadrants"])

    return {
        "day": day,
        "product_tiles": product_tiles,
        "animal_counts": animal_targets,
        "daily_hands": daily_hands,
        "quadrants": quadrants,
        "cash_reserve": cash_reserve,
        "replant_cutoffs": replant_cutoffs,
        "crop_scores": crop_scores,
        "animal_scores": animal_scores,
        "watcher_adjustments": {
            "labor_hand_delta": (daily_hands - _daily_hand_target(
                day, estimated_load, me.get("money", 0), cash_reserve, remaining_days
            )) if signals is not None and signals.get("backend") == "linear_svm" else 0,
            "land_target": quadrants,
        },
    }


# ---- Reversible watcher ----------------------------------------------------

def _scheduled_outputs(age, first_day, interval, horizon):
    """Count scheduled outputs strictly after now through the horizon."""
    count = 0
    for offset in range(1, horizon + 1):
        future_age = age + offset
        if future_age >= first_day and (future_age - first_day) % interval == 0:
            count += 1
    return count


def _forecast_opponent_supply(obs, horizon=6):
    """Forecast visible opponent output arriving over the next `horizon` days."""
    forecast = {product: 0.0 for product in _PRODUCTS}
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    opponent = 1 - player
    if opponent >= len(farms):
        return forecast
    day = obs.get("day", 0)
    for row in farms[opponent].get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            ready = max(0.0, float(tile.get("yield_units", 0)))
            if crop in forecast:
                forecast[crop] += ready
                age = max(0, day - tile.get("planted_day", day))
                if crop in _ONGOING_CROP_INTERVAL:
                    forecast[crop] += _scheduled_outputs(
                        age, _CROP_FIRST_YIELD[crop], _ONGOING_CROP_INTERVAL[crop], horizon
                    )
                elif ready <= 0 and age < _CROP_MAX_DAY[crop] <= age + horizon:
                    # At peak, an unfertilized one-time crop holds roughly
                    # daily_yield * occupancy output; current ready yield is
                    # used instead whenever it is already public.
                    forecast[crop] += max(1.0, _CROP_DAILY_YIELD[crop] * _CROP_MAX_DAY[crop])
            elif animal in _ANIMAL_PRODUCT:
                product = _ANIMAL_PRODUCT[animal]
                forecast[product] += ready
                age = max(0, day - tile.get("placed_day", day))
                forecast[product] += _scheduled_outputs(
                    age, _ANIMAL_FIRST_YIELD[animal], _ANIMAL_INTERVAL[animal], horizon
                )
    return forecast


def watcher_signals(obs, backend):
    """Return bounded, observation-only watcher recommendations.

    Phase 3 enables exact township demand. Opponent forecasts remain neutral
    until their independent layer is enabled. Compatible exported SVM models
    override individual signals; missing/invalid outputs retain rule values.
    """
    requested_backend = backend
    if backend not in ("off", "rules", "linear_svm"):
        backend = "rules"
    svm_ready = bool(
        backend == "linear_svm" and
        _watcher_features is not None and _watcher_model is not None and
        getattr(_watcher_model, "SCHEMA_VERSION", None) == _watcher_features.SCHEMA_VERSION and
        getattr(_watcher_model, "SCHEMA_HASH", None) == _watcher_features.SCHEMA_HASH and
        tuple(getattr(_watcher_model, "FEATURE_ORDER", ())) == tuple(_watcher_features.FEATURE_ORDER) and
        bool(getattr(_watcher_model, "MODELS", {}))
    )
    if backend == "linear_svm" and not svm_ready:
        backend = "rules"

    zeros = {product: 0.0 for product in _PRODUCTS}
    if backend == "off":
        return {
            "product_attractiveness": dict(zeros),
            "town_demand_pressure": dict(zeros),
            "opponent_supply_pressure": dict(zeros),
            "market_glut_risk": dict(zeros),
            "competitive_expansion_pressure": 0.0,
            "land_expansion_pressure": 0.0,
            "labor_expansion_pressure": 0.0,
            "recommended_limits": {
                "product_exposure": {product: 0 for product in _PRODUCTS},
                "hands": 0, "quadrants": 1, "cash_reserve": 0.0,
            },
            "backend": "off",
        }

    demand_per_day = {product: 1.0 for product in _PRODUCTS}  # town center
    if ENABLE_TOWN_WATCHER:
        for shop in obs.get("town", {}).get("unlocked_shops", []):
            products = _SHOP_PRODUCTS.get(shop, ())
            rate = 12.0 if len(products) == 1 else 6.0
            for product in products:
                demand_per_day[product] += rate
    else:
        demand_per_day = dict(zeros)

    town_pressure = {
        product: max(0.0, min(1.0, demand_per_day[product] / (_THROUGHPUT[product] / 24.0)))
        for product in _PRODUCTS
    }
    opponent_forecast = _forecast_opponent_supply(obs) if ENABLE_OPPONENT_WATCHER else dict(zeros)
    opponent_pressure = {
        product: max(0.0, min(1.0, opponent_forecast[product] / (_THROUGHPUT[product] * 6.0 / 24.0)))
        for product in _PRODUCTS
    }
    market = obs.get("market", {})
    prices = market.get("prices", {})
    inventory = market.get("inventory", {})
    glut_risk = {}
    attractiveness = {}
    for product in _PRODUCTS:
        price_ratio = prices.get(product, _BASE_PRICE[product]) / float(_BASE_PRICE[product])
        displacement = max(0.0, inventory.get(product, 10000) - 10000) / float(_THROUGHPUT[product])
        glut = max(0.0, min(1.0, 0.45 * max(0.0, 1.0 - price_ratio) +
                            0.35 * min(1.0, displacement) + 0.20 * opponent_pressure[product]))
        glut_risk[product] = glut
        score = (0.55 * town_pressure[product] +
                 0.45 * max(-1.0, min(1.0, price_ratio - 1.0)) -
                 0.65 * glut - OPPONENT_SIGNAL_WEIGHT * opponent_pressure[product])
        attractiveness[product] = max(-1.0, min(1.0, score))

    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    me = farms[player] if player < len(farms) else {}
    live_animals = sum(
        1 for row in me.get("tiles", []) for tile in row
        if isinstance(tile, dict) and tile.get("animal")
    )
    cash_reserve = live_animals * prices.get("WHEAT", _BASE_PRICE["WHEAT"]) * 2.0
    exposure = {}
    for product in _PRODUCTS:
        # Existing herd economics and structures are sunk commitments. Phase
        # 4 may redirect future crop vacancies, but it must not strand the
        # phase-3 livestock build merely because the opponent shows supply.
        competing_supply = opponent_forecast[product] / 6.0 if product in _SEED_COST else 0.0
        units_per_day = max(0.0, demand_per_day[product] - competing_supply)
        assets = units_per_day * TOWN_EXPOSURE_MULTIPLIER / _PRODUCT_DAILY_YIELD[product]
        exposure[product] = max(0, int(assets + 0.999999))
    if obs.get("day", 0) < TOWN_WATCHER_START_DAY:
        # The watcher redirects future vacancies; it does not suppress the
        # proven opening build before township demand has differentiated.
        for crop in _SEED_COST:
            exposure[crop] = MAX_MELON_TILES
        for animal, product in _ANIMAL_PRODUCT.items():
            exposure[product] = _ANIMAL_CAP[animal]
    own_assets = sum(
        1 for row in me.get("tiles", []) for tile in row
        if isinstance(tile, dict) and (tile.get("crop") or tile.get("animal"))
    )
    opponent_farm = farms[1 - player] if len(farms) > 1 else {}
    opponent_assets = sum(
        1 for row in opponent_farm.get("tiles", []) for tile in row
        if isinstance(tile, dict) and (tile.get("crop") or tile.get("animal"))
    )
    rule_attractiveness = dict(attractiveness)
    unmet_pressure = max(attractiveness.values()) if attractiveness else 0.0
    scale_gap = (opponent_assets - own_assets) / 25.0
    rule_expansion = max(-1.0, min(1.0, 0.65 * unmet_pressure + 0.35 * scale_gap))
    land_expansion = rule_expansion
    labor_expansion = rule_expansion
    model_scores = {}
    fallback_reasons = {}
    changed_outputs = []
    if svm_ready:
        try:
            features = _watcher_features.extract_features(obs)
            if (len(features) != len(_watcher_features.FEATURE_ORDER) or
                    not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and value == value and abs(value) != float("inf") for value in features)):
                raise ValueError("invalid_features")

            def valid_score(output_id):
                try:
                    value = _watcher_model.score(output_id, features)
                except (KeyError, TypeError, ValueError, ArithmeticError):
                    fallback_reasons[output_id] = "model_error"
                    return None
                if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                        value != value or abs(value) == float("inf")):
                    fallback_reasons[output_id] = "missing_or_invalid_output"
                    return None
                value = max(-1.0, min(1.0, float(value)))
                model_scores[output_id] = value
                return value

            for product in _PRODUCTS:
                increase = valid_score(f"product:{product}:increase")
                avoid = valid_score(f"product:{product}:avoid")
                if increase is not None or avoid is not None:
                    attractiveness[product] = max(
                        -1.0, min(1.0, (increase or 0.0) - (avoid or 0.0))
                    )
                    changed_outputs.append(f"product:{product}")
            model_land = valid_score("land_expansion")
            model_labor = valid_score("labor_expansion")
            if model_land is not None:
                land_expansion = model_land
                changed_outputs.append("land_expansion")
            if model_labor is not None:
                labor_expansion = model_labor
                changed_outputs.append("labor_expansion")
            backend = "linear_svm" if changed_outputs else "rules"
        except (KeyError, TypeError, ValueError, ArithmeticError):
            backend = "rules"
            fallback_reasons["features"] = "extraction_or_validation_failed"
    expansion = land_expansion
    return {
        "product_attractiveness": attractiveness,
        "town_demand_pressure": town_pressure,
        "opponent_supply_pressure": opponent_pressure,
        "market_glut_risk": glut_risk,
        "competitive_expansion_pressure": expansion,
        "land_expansion_pressure": land_expansion,
        "labor_expansion_pressure": labor_expansion,
        "recommended_limits": {
            "product_exposure": exposure,
            "hands": min(N_HIRES, len(me.get("hands", [])) or N_HIRES),
            "quadrants": min(LAND_TARGET_QUADRANTS, len(me.get("unlocked_quadrants", ["NW"])) +
                             (1 if expansion > OPPONENT_EXPANSION_THRESHOLD else 0)),
            "cash_reserve": cash_reserve,
        },
        "backend": backend if requested_backend != "off" else "off",
        "watcher_trace": {
            "rule_scores": {
                "product_attractiveness": rule_attractiveness,
                "land_expansion": rule_expansion,
                "labor_expansion": rule_expansion,
            },
            "model_scores": model_scores,
            "final_scores": {
                "product_attractiveness": dict(attractiveness),
                "land_expansion": land_expansion,
                "labor_expansion": labor_expansion,
            },
            "changed_outputs": changed_outputs,
            "changed_signals": [
                output_id for output_id in changed_outputs
                if ((output_id == "land_expansion" and land_expansion != rule_expansion) or
                    (output_id == "labor_expansion" and labor_expansion != rule_expansion) or
                    (output_id.startswith("product:") and
                     attractiveness[output_id.split(":", 1)[1]] !=
                     rule_attractiveness[output_id.split(":", 1)[1]]))
            ],
            "fallback_reasons": fallback_reasons,
        },
    }


def _adaptive_animal_slots(tiles, targets):
    """Keep every existing animal; allocate only currently vacant slots."""
    assignments = {}
    remaining = dict(targets["animal_counts"])
    for pos in LIVESTOCK_POSITIONS:
        x, y = pos
        tile = tiles[y][x]
        if isinstance(tile, dict) and tile.get("animal") in remaining:
            animal = tile["animal"]
            assignments[pos] = animal
            remaining[animal] = max(0, remaining[animal] - 1)
    sequence = []
    for animal in sorted(remaining, key=lambda name: (-targets["animal_scores"][name], name)):
        sequence.extend([animal] * remaining[animal])
    for pos in LIVESTOCK_POSITIONS:
        if pos not in assignments and sequence:
            assignments[pos] = sequence.pop(0)
    return assignments


def _adaptive_crop_slots(tiles, targets):
    """Healthy crops keep their type; empty slots take the largest deficit."""
    assignments = {}
    remaining = dict(targets["product_tiles"])
    crop_pool = _ALL_FREE[:MAX_MELON_TILES]
    for pos in crop_pool:
        x, y = pos
        tile = tiles[y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") in remaining:
            crop = tile["crop"]
            assignments[pos] = crop
            remaining[crop] = max(0, remaining[crop] - 1)
    for pos in crop_pool:
        x, y = pos
        tile = tiles[y][x]
        if pos in assignments or (isinstance(tile, dict) and tile.get("kind") not in ("WEED",)):
            continue
        if CROP_TRANSITION_MODE == "fixed":
            scheduled = _fixed_transition_crop(targets["day"], crop_pool.index(pos))
            if scheduled is None:
                continue
            assignments[pos] = scheduled
            if scheduled in remaining:
                remaining[scheduled] = max(0, remaining[scheduled] - 1)
            continue
        choices = [crop for crop, deficit in remaining.items() if deficit > 0]
        if not choices:
            break
        crop = max(choices, key=lambda name: (remaining[name], targets["crop_scores"][name], name))
        assignments[pos] = crop
        remaining[crop] -= 1
    return assignments


def _planned_tile_positions():
    """Unique coordinates the configured production layout can occupy."""
    return set(LIVESTOCK_POSITIONS) | set(MELON_POSITIONS) | set(WHEAT_POSITIONS) | set(CARROT_POSITIONS)


def _fixed_transition_crop(day, slot_index):
    """Crop for a vacant phase-6 slot; living plants are never dug early."""
    if day <= MELON_LAST_PLANT_DAY:
        return "MELON"
    if (day >= LATE_WHEAT_START_DAY and day <= WHEAT_LAST_PLANT_DAY
            and slot_index < LATE_WHEAT_TILES):
        return "WHEAT"
    if (TRANSITION_EMPTY_POLICY == "strawberry" and day >= STRAWBERRY_START_DAY
            and day <= STRAWBERRY_LAST_PLANT_DAY and slot_index < STRAWBERRY_TILES):
        return "STRAWBERRY"
    if TRANSITION_EMPTY_POLICY == "wheat" and day <= WHEAT_LAST_PLANT_DAY:
        return "WHEAT"
    return None


# ---- Deadline-safe global planner ------------------------------------------

def _planner_crop_name(day, pos, crop_assignments=None):
    if crop_assignments is not None:
        return crop_assignments.get(pos)
    if pos in MELON_POSITIONS:
        if CROP_TRANSITION_MODE == "fixed":
            return _fixed_transition_crop(day, MELON_POSITIONS.index(pos))
        return "MELON" if day < MELON_EXIT_DAY else "WHEAT"
    if pos in WHEAT_POSITIONS:
        return "WHEAT"
    if pos in CARROT_POSITIONS:
        return "CARROT"
    return None


def _planner_daily_load(tiles, extra_animals=0, extra_crops=0):
    """Conservative action estimate for assets that must be serviced daily.

    Animals need feed and care, normally offer fertilizer, and periodically
    need harvest. Crops need water plus amortized harvest/replant work. The
    final term prices in local travel even when every tile action succeeds.
    """
    animals = 0
    crops = 0
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("animal"):
                animals += 1
            elif tile.get("kind") == "PLANT":
                crops += 1
    animals += extra_animals
    crops += extra_crops
    obligations = animals * 3.5 + crops * 1.25
    travel = (animals + crops) * 0.50
    return obligations + travel


def _planner_has_capacity(tiles, unit_count, extra_animals=0, extra_crops=0):
    usable_turns = max(1, unit_count) * 24 * (1.0 - DEADLINE_CAPACITY_MARGIN)
    return _planner_daily_load(tiles, extra_animals, extra_crops) <= usable_turns


def _planner_tasks(day, tiles, seeds, targets=None):
    """Build the ordered, global task board for the current observation."""
    tasks = []
    animal_assignments = _adaptive_animal_slots(tiles, targets) if targets else dict(zip(LIVESTOCK_POSITIONS, LIVESTOCK_TYPES))
    crop_assignments = _adaptive_crop_slots(tiles, targets) if targets else None
    livestock_set = set(animal_assignments)
    crop_positions = list(crop_assignments) if crop_assignments is not None else MELON_POSITIONS + WHEAT_POSITIONS + CARROT_POSITIONS

    for pos, atype in animal_assignments.items():
        x, y = pos
        tile = tiles[y][x]
        if tile is None:
            structure = "BUILD_COOP" if atype == "GOOSE" else "BUILD_PASTURE"
            tasks.append((6, pos, structure, None))
        elif isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and not tile.get("animal"):
            wanted_structure = "COOP" if atype == "GOOSE" else "PASTURE"
            if tile.get("kind") == wanted_structure:
                tasks.append((4, pos, "PLACE", atype))
            else:
                tasks.append((6, pos, "DIG", None))
        elif isinstance(tile, dict) and tile.get("animal"):
            if not tile.get("fed_today", False):
                priority = 0 if tile.get("consecutive_unfed", 0) >= 1 else 2
                tasks.append((priority, pos, "FEED", "WHEAT"))
            if tile.get("yield_units", 0) > 0:
                tasks.append((3, pos, "HARVEST", None))
            if tile.get("fertilizer_available", False):
                tasks.append((4, pos, "COLLECT_FERTILIZER", None))
            if tile.get("fed_today", False) and not tile.get("cared_today", False):
                tasks.append((5, pos, "CARE", None))

    for pos in crop_positions:
        if pos in livestock_set:
            continue
        x, y = pos
        tile = tiles[y][x]
        crop = _planner_crop_name(day, pos, crop_assignments)
        if crop is None:
            continue
        if tile is None:
            tasks.append((6, pos, "PLANT", crop))
        elif not isinstance(tile, dict):
            continue
        elif tile.get("kind") == "WEED":
            tasks.append((6, pos, "DIG", None))
        elif tile.get("kind") == "PLANT":
            actual_crop = tile.get("crop", crop)
            max_day = CROP_MAX_YIELD_DAY.get(actual_crop, MELON_YIELD_DAY)
            age = day - tile.get("planted_day", day)
            if not tile.get("watered_today", False):
                missed = tile.get("consecutive_unwatered", 0) >= 1 or tile.get("planted_day") == day
                tasks.append((1 if missed else 2, pos, "WATER", None))
            if tile.get("yield_units", 0) > 0 and age >= max_day:
                tasks.append((3, pos, "HARVEST", None))

    # Stable sorting preserves the configured contiguous zone order inside a
    # priority, while assignment below still chooses the nearest unit.
    tasks.sort(key=lambda task: task[0])
    return tasks


def _planner_distance(pos, target, required_item, inv):
    direct = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
    if required_item is None or inv.get(required_item, 0) > 0:
        return direct
    via_shed = min(abs(pos[0] - sx) + abs(pos[1] - sy) for sx, sy in SHED_ADJACENT)
    shed_to_target = min(abs(target[0] - sx) + abs(target[1] - sy) for sx, sy in SHED_ADJACENT)
    return via_shed + 1 + shed_to_target


def _deadline_actions(me, private, day, hour, tiles, shed, seeds, targets=None):
    """Greedily assign each target once, in deadline order, to its cheapest unit."""
    positions = [tuple(me["farmer"])] + [tuple(p) for p in me.get("hands", [])]
    inventories = private.get("inventories", [{}])
    invs = [inventories[i] if i < len(inventories) else {} for i in range(len(positions))]
    tasks = _planner_tasks(day, tiles, seeds, targets)
    feed_tasks = sum(1 for task in tasks if task[2] == "FEED")
    actions = [["PASS"] for _ in positions]
    free_units = set(range(len(positions)))
    claimed_targets = set()
    reserved_items = {}
    reserved_seeds = {}
    animal_zone = set(LIVESTOCK_POSITIONS)
    animal_units = len(positions)
    if targets and len(positions) > 1:
        # Base the split on fixed maximum workloads, not live prices or
        # partially completed purchases, so hand-slot roles cannot drift
        # during a day as targets are recomputed.
        animal_load = len(LIVESTOCK_POSITIONS) * 4.0
        crop_load = MAX_MELON_TILES * 1.75
        animal_units = max(1, min(len(positions) - 1, round(len(positions) * animal_load / max(1.0, animal_load + crop_load))))
    specialist_count = min(LIVESTOCK_SPECIALISTS, max(0, animal_units - 1))
    specialist_zones = _chunk_indices(len(LIVESTOCK_POSITIONS), specialist_count + 1)
    specialist_owner = {}
    for specialist_offset, zone in enumerate(specialist_zones[1:], start=1):
        for idx in zone:
            specialist_owner[LIVESTOCK_POSITIONS[idx]] = specialist_offset

    for priority, target, operation, item in tasks:
        if not free_units:
            break
        if target in claimed_targets:
            continue
        if operation == "PLANT":
            if reserved_seeds.get(item, 0) >= seeds.get(item, 0):
                continue
            if not _planner_has_capacity(tiles, len(positions), extra_crops=reserved_seeds.get("_new", 0) + 1):
                continue
        if operation == "PLACE":
            total_item = shed.get(item, 0) + sum(inv.get(item, 0) for inv in invs)
            if reserved_items.get(item, 0) >= total_item:
                continue
            if not _planner_has_capacity(tiles, len(positions), extra_animals=reserved_items.get("_animals", 0) + 1):
                continue
        if operation == "FEED":
            total_wheat = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in invs)
            if reserved_items.get("WHEAT", 0) >= total_wheat:
                continue

        candidates = []
        for unit in free_units:
            if targets:
                livestock_task = target in animal_zone
                if livestock_task != (unit < animal_units):
                    continue
                owner = specialist_owner.get(target)
                emergency = priority <= 1
                if not emergency:
                    if 1 <= unit <= specialist_count and owner != unit:
                        continue
                    if owner is not None and unit != owner:
                        continue
            inv = invs[unit]
            carried_item = item if operation != "PLANT" else None
            if carried_item is not None and inv.get(carried_item, 0) <= 0 and shed.get(carried_item, 0) <= reserved_items.get(carried_item, 0):
                continue
            if operation == "PLANT":
                direct = abs(positions[unit][0] - target[0]) + abs(positions[unit][1] - target[1])
                if hour + direct + 2 > 24:  # travel, plant, then water before refresh
                    continue
            candidates.append((_planner_distance(positions[unit], target, carried_item, inv), unit))
        if not candidates:
            continue
        _, unit = min(candidates)
        free_units.remove(unit)
        claimed_targets.add(target)
        pos = positions[unit]
        inv = invs[unit]

        if item is not None and operation != "PLANT" and inv.get(item, 0) <= 0:
            pickup_n = 1
            if operation == "FEED":
                pickup_n = max(1, (feed_tasks + len(positions) - 1) // len(positions))
            pickup_n = min(pickup_n, max(0, shed.get(item, 0) - reserved_items.get(item, 0)))
            if _is_shed_adjacent(pos):
                actions[unit] = ["PICKUP", item, pickup_n]
            else:
                shed_target = min(SHED_ADJACENT, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
                actions[unit] = [_step_toward(pos, shed_target)]
            reserved_items[item] = reserved_items.get(item, 0) + pickup_n
        elif pos == target:
            actions[unit] = [operation, item] if operation in ("PLACE", "PLANT") else [operation]
            if operation == "PLANT":
                reserved_seeds[item] = reserved_seeds.get(item, 0) + 1
                reserved_seeds["_new"] = reserved_seeds.get("_new", 0) + 1
            elif operation == "PLACE":
                reserved_items[item] = reserved_items.get(item, 0) + 1
                reserved_items["_animals"] = reserved_items.get("_animals", 0) + 1
            elif operation == "FEED":
                reserved_items["WHEAT"] = reserved_items.get("WHEAT", 0) + 1
        else:
            actions[unit] = [_step_toward(pos, target)]

    return actions[0], actions[1:]


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    hour = obs["hour"]
    tiles = me["tiles"]
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    money = me["money"]
    inventories = private.get("inventories", [{}])

    buy_orders = []
    sell_orders = []
    liquidating = day >= 29

    # Only HIRE truly needs hour 0 (hands are hired for the whole day, once).
    # Every other purchase type is spread across its own hour instead of all
    # piling into hour 0 -- maxMarketOrdersPerTurn (10) is a *per-turn* cap,
    # and with N_HIRES approaching double digits, HIRE alone can fill it,
    # silently dropping every other buy (and every sell) that day if they
    # all queued at hour 0 too. One purchase category per hour sidesteps the
    # cap entirely regardless of how many hands/crops are configured.
    prices = obs.get("market", {}).get("prices", {})
    signals = watcher_signals(obs, WATCHER_BACKEND) if ENABLE_TOWN_WATCHER or ENABLE_OPPONENT_WATCHER else None
    targets = _adaptive_targets(obs, me, tiles, prices, signals) if ENABLE_ADAPTIVE_TARGETS else None

    wheat_reserve = 0 if liquidating else (
        sum(targets["animal_counts"].values()) if targets else len(LIVESTOCK_TYPES)
    ) * 2
    if SALE_POLICY_MODE == "product_specific":
        # Product-specific gates distinguish high-volume staples from premium
        # goods. Thresholds decay late, overflow bypasses them, and day 29
        # liquidates every product regardless of the normal policy.
        for item, n in sorted(shed.items()):
            eligible = n - (wheat_reserve if item == "WHEAT" else 0)
            if eligible <= 0:
                continue
            base = _BASE_PRICE.get(item)
            cur = prices.get(item)
            threshold = SALE_MIN_PRICE_FRAC_BY_PRODUCT.get(item, SELL_MIN_PRICE_FRAC)
            if day >= SALE_THRESHOLD_DECAY_START_DAY and not liquidating:
                span = max(1, 29 - SALE_THRESHOLD_DECAY_START_DAY)
                threshold *= max(0.0, (29 - day) / span)
            if (not liquidating and n < SALE_OVERFLOW_QUANTITY and base and cur is not None
                    and cur < threshold * base):
                continue
            batch = SALE_MAX_BATCH_BY_PRODUCT.get(item)
            quantity = eligible if liquidating or batch is None else min(eligible, batch)
            if quantity > 0:
                sell_orders.append(["SELL", item, quantity])
        if not liquidating:
            sell_orders = sell_orders[:SALE_ROUTINE_ORDER_BUDGET]
    else:
        for item, n in list(shed.items()):
            if n <= 0 or item == "WHEAT":
                continue
            base = _BASE_PRICE.get(item)
            cur = prices.get(item)
            if (not liquidating and SELL_MIN_PRICE_FRAC > 0 and base and cur is not None
                    and cur < SELL_MIN_PRICE_FRAC * base and n < 90):
                continue
            sell_orders.append(["SELL", item, n])
        wheat_surplus = shed.get("WHEAT", 0) - wheat_reserve
        if wheat_surplus > 0:
            sell_orders.append(["SELL", "WHEAT", wheat_surplus])

    if (targets and hour <= 3) or (not targets and hour == 0):
        daily_hands = targets["daily_hands"] if targets else N_HIRES
        if targets:
            live_animals = sum(
                1 for row in tiles for tile in row
                if isinstance(tile, dict) and tile.get("animal")
            )
            wheat_on_hand = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
            today_feed_reserve = max(0, live_animals - wheat_on_hand) * prices.get("WHEAT", _BASE_PRICE["WHEAT"])
            hires_today = me.get("hires_today", len(me.get("hands", [])))
            a, b = 1, 1
            for _ in range(hires_today):
                a, b = b, a + b
            spent = 0
            for _ in range(max(0, daily_hands - hires_today)):
                if money - spent - a < today_feed_reserve:
                    break
                buy_orders.append(["HIRE"])
                spent += a
                a, b = b, a + b
        else:
            for _ in range(daily_hands):
                buy_orders.append(["HIRE"])

    if hour == 1:
        # Wheat feed is funded before any other spend, at the live market
        # price -- an existing herd starving because some other purchase ate
        # the day's cash is an unrecoverable death spiral (2 unfed days =
        # animal escapes for good), so feeding what's alive always wins.
        # Self-grown wheat (if any) already lowered `wheat_have`, so this
        # naturally only tops up whatever the wheat plot didn't cover.
        if targets:
            wheat_needed = sum(
                1 for row in tiles for tile in row
                if isinstance(tile, dict) and tile.get("animal")
            )
        else:
            wheat_needed = len(LIVESTOCK_TYPES)
        wheat_have = shed.get("WHEAT", 0)
        for inv in inventories:
            wheat_have += inv.get("WHEAT", 0)
        deficit = wheat_needed - wheat_have
        if deficit > 0:
            wheat_price = prices.get("WHEAT", 25)
            afford = int(money // wheat_price) if wheat_price > 0 else deficit
            n_buy = min(deficit, afford)
            if n_buy > 0:
                buy_orders.append(["BUY_PRODUCT", "WHEAT", n_buy])

    if hour == 2:
        # Buy missing COW/SHEEP up to ANIMAL_MIX, capped by affordability
        # and (if set) a per-day purchase cap.
        needed_counts = {}
        if targets:
            assignments = _adaptive_animal_slots(tiles, targets)
            for pos, atype in assignments.items():
                x, y = pos
                t = tiles[y][x]
                if not (isinstance(t, dict) and t.get("animal") == atype):
                    needed_counts[atype] = needed_counts.get(atype, 0) + 1
        else:
            for i, atype in enumerate(LIVESTOCK_TYPES):
                x, y = LIVESTOCK_POSITIONS[i]
                t = tiles[y][x]
                if not (isinstance(t, dict) and t.get("animal") == atype):
                    needed_counts[atype] = needed_counts.get(atype, 0) + 1
        remaining_cap = MAX_ANIMAL_BUY_PER_DAY if MAX_ANIMAL_BUY_PER_DAY is not None else sum(needed_counts.values())
        cash = money
        for atype, need in needed_counts.items():
            if remaining_cap <= 0:
                break
            have = shed.get(atype, 0)
            for inv in inventories:
                have += inv.get(atype, 0)
            to_buy = max(0, need - have)
            cost_each = _ANIMAL_COST_ALL[atype] if targets else ANIMAL_COST[atype]
            spendable = max(0, cash - (targets["cash_reserve"] if targets else 0))
            afford = int(spendable // cost_each)
            n_buy = min(to_buy, afford, remaining_cap)
            if ENABLE_DEADLINE_PLANNER or ENABLE_ADAPTIVE_TARGETS:
                intended_hands = targets["daily_hands"] if targets else N_HIRES
                unit_count = 1 + max(len(me.get("hands", [])), intended_hands)
                while n_buy > 0 and not _planner_has_capacity(tiles, unit_count, extra_animals=n_buy):
                    n_buy -= 1
            if n_buy > 0:
                buy_orders.append(["BUY_ANIMAL", atype, n_buy])
                cash -= n_buy * cost_each
                remaining_cap -= n_buy

    target_quadrants = targets["quadrants"] if targets else LAND_TARGET_QUADRANTS
    intended_hands = targets["daily_hands"] if targets else N_HIRES
    unlocked_count = len(me.get("unlocked_quadrants", ["NW"]))
    next_land_cost = {1: 1000, 2: 2000, 3: 4000}.get(unlocked_count, 999999)
    land_allowed = True
    if LAND_EXPANSION_MODE == "conditional" and unlocked_count >= 2:
        occupied = sum(
            1 for row in tiles for tile in row
            if isinstance(tile, dict) and (tile.get("crop") or tile.get("animal") or tile.get("kind"))
        )
        utilization = occupied / max(1, 25 * unlocked_count)
        reserve = (targets["cash_reserve"] if targets else 0) + LAND_OPERATING_RESERVE
        enough_time = (29 - day) >= LAND_MIN_REPAY_DAYS
        # Transition crops replace occupants of the melon pool; count unique
        # coordinates rather than phase-specific crop counts.
        concrete_layout = len(_planned_tile_positions()) > 25 * unlocked_count
        land_allowed = (utilization >= LAND_UTILIZATION_THRESHOLD and enough_time and
                        money - next_land_cost >= reserve and concrete_layout)
    if hour == 3 and day >= LAND_MIN_DAY and unlocked_count < target_quadrants and land_allowed and (
        not (ENABLE_DEADLINE_PLANNER or ENABLE_ADAPTIVE_TARGETS) or _planner_has_capacity(tiles, 1 + max(len(me.get("hands", [])), intended_hands))
    ) and (
        not targets or money - next_land_cost >= targets["cash_reserve"]
    ):
        # No money precheck: the engine silently no-ops an unaffordable
        # BUY_LAND, and re-issuing it every day until it lands is the exact
        # idiom every top-10 leaderboard agent uses (report.md) -- it
        # guarantees the purchase clears on the earliest affordable day with
        # no risk of a stale one-shot check missing the window.
        buy_orders.append(["BUY_LAND"])

    if hour == 4:
        if targets:
            assignments = _adaptive_crop_slots(tiles, targets)
            needed_seeds = {}
            for (x, y), crop in assignments.items():
                if tiles[y][x] is None:
                    needed_seeds[crop] = needed_seeds.get(crop, 0) + 1
            cash = money
            for crop in sorted(needed_seeds, key=lambda name: (-targets["crop_scores"][name], name)):
                if crop == "MELON" and CROP_TRANSITION_MODE == "fixed" and day > MELON_LAST_SEED_DAY:
                    continue
                if crop == "STRAWBERRY" and day > STRAWBERRY_LAST_PLANT_DAY:
                    continue
                if crop == "WHEAT" and day > WHEAT_LAST_PLANT_DAY:
                    continue
                need = max(0, needed_seeds[crop] - seeds.get(crop, 0))
                cost = _SEED_COST[crop]
                spendable = max(0, cash - targets["cash_reserve"])
                n_buy = min(need, int(spendable // cost))
                if n_buy > 0:
                    buy_orders.append(["BUY_SEED", crop, n_buy])
                    cash -= n_buy * cost
        else:
            empty_positions = [(x, y) for x, y in MELON_POSITIONS if tiles[y][x] is None]
            if ENABLE_DEADLINE_PLANNER:
                unit_count = 1 + max(len(me.get("hands", [])), N_HIRES)
                empty_positions = empty_positions[:max(0, int((unit_count * 24 * (1.0 - DEADLINE_CAPACITY_MARGIN) - _planner_daily_load(tiles)) // 1.75))]
            if day < MELON_EXIT_DAY:
                have_seeds = seeds.get("MELON", 0)
                if len(empty_positions) > have_seeds:
                    buy_orders.append(["BUY_SEED", "MELON", len(empty_positions) - have_seeds])
            else:
                have_seeds = seeds.get("WHEAT", 0)
                if len(empty_positions) > have_seeds:
                    buy_orders.append(["BUY_SEED", "WHEAT", len(empty_positions) - have_seeds])

    if not targets and hour == 5 and WHEAT_TILES > 0:
        empty_wheat = sum(1 for x, y in WHEAT_POSITIONS if tiles[y][x] is None)
        if ENABLE_DEADLINE_PLANNER:
            unit_count = 1 + max(len(me.get("hands", [])), N_HIRES)
            spare = max(0, int((unit_count * 24 * (1.0 - DEADLINE_CAPACITY_MARGIN) - _planner_daily_load(tiles)) // 1.75))
            empty_wheat = min(empty_wheat, spare)
        have_seeds = seeds.get("WHEAT", 0)
        if empty_wheat > have_seeds:
            buy_orders.append(["BUY_SEED", "WHEAT", empty_wheat - have_seeds])

    if not targets and hour == 6 and CARROT_TILES > 0:
        empty_carrot = sum(1 for x, y in CARROT_POSITIONS if tiles[y][x] is None)
        if ENABLE_DEADLINE_PLANNER:
            unit_count = 1 + max(len(me.get("hands", [])), N_HIRES)
            spare = max(0, int((unit_count * 24 * (1.0 - DEADLINE_CAPACITY_MARGIN) - _planner_daily_load(tiles)) // 1.75))
            empty_carrot = min(empty_carrot, spare)
        have_seeds = seeds.get("CARROT", 0)
        if empty_carrot > have_seeds:
            buy_orders.append(["BUY_SEED", "CARROT", empty_carrot - have_seeds])

    # Closing sales take precedence; always enforce the engine's silent
    # maxMarketOrdersPerTurn=10 truncation explicitly.
    market = (sell_orders + buy_orders if liquidating else buy_orders + sell_orders)[:10]

    if ENABLE_DEADLINE_PLANNER or ENABLE_ADAPTIVE_TARGETS:
        farmer_action, hands_actions = _deadline_actions(me, private, day, hour, tiles, shed, seeds, targets)
        return {"farmer": farmer_action, "hands": hands_actions, "market": market}

    fx, fy = me["farmer"]
    farmer_inv = inventories[0] if inventories else {}
    farmer_idxs = LIVESTOCK_GROUPS[0] if LIVESTOCK_GROUPS else []
    farmer_action = _livestock_action((fx, fy), farmer_inv, tiles, shed, farmer_idxs) if farmer_idxs else ["PASS"]

    _DISPATCH = {
        "LIVESTOCK": lambda pos, inv, idxs: _livestock_action(pos, inv, tiles, shed, idxs),
        "MELON": lambda pos, inv, idxs: _melon_action(pos, day, tiles, idxs),
        "WHEAT": lambda pos, inv, idxs: _wheat_crop_action(pos, day, tiles, idxs),
        "CARROT": lambda pos, inv, idxs: _carrot_action(pos, day, tiles, idxs),
    }

    hands = me.get("hands", [])
    hands_actions = []
    for h_idx, hpos in enumerate(hands):
        hinv = inventories[h_idx + 1] if h_idx + 1 < len(inventories) else {}
        if h_idx < len(_CREW_PLAN):
            job, idxs = _CREW_PLAN[h_idx]
            hands_actions.append(_DISPATCH[job](tuple(hpos), hinv, idxs) if idxs else ["PASS"])
        else:
            hands_actions.append(["PASS"])

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}


def _demo():
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 100}, debug=False)
    env.run([agent, "random"])
    final = env.steps[-1][0].observation["farms"][0]["money"]
    assert final >= 0
    print("strategy_follower self-check OK, final money:", final)


if __name__ == "__main__":
    _demo()
