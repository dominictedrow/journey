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

Current replay-driven revision: expansion is now isolated from the earlier
herd-size experiment. It can expand through NW+NE+SW when the land purchase is
affordable, assigns the overflow crop slice in SW, stops melon/wheat planting
when the remaining season cannot repay them, and liquidates on day 29. Routing
now uses owned-cell paths, including recovery from a hand's locked spawn tile.
These are behavior/correctness changes requested from replay review; they are
not yet a paired-performance claim.
"""

try:
    import shared_features
    import watcher_model
except ImportError:
    shared_features = None
    watcher_model = None

# ---- Constants (the "easy to change" knobs) --------------------------------

ANIMAL_MIX = {"COW": 11, "SHEEP": 6}   # pass 3 value -- pass 4's 15/8 AB-confirmed worse (see docstring), reverted
ANIMAL_COST = {"COW": 400, "SHEEP": 500}
LIVESTOCK_HANDS = 6                    # pass 3 value -- pass 4's 7 (N_HIRES 8->9) AB-confirmed worse, reverted
MELON_HANDS = 3                        # scale crop labor with the third quadrant: 45 tiles split 15/hand keeps each route
                                        # below the known-bad 19-tile single-hand load while holding total hires at the
                                        # documented safe ceiling of 9 (6 livestock + 3 crop)
MAX_MELON_TILES = 45                   # 8 NW + 25 NE + 12 SW: the $2k third quadrant now adds a meaningful production
                                        # block instead of only three tiles; MELON_EXIT_DAY still limits late exposure
MELON_YIELD_DAY = 10                   # melon: first_yield_day == max_yield_day == 10 (RULES.md Object Types)
MELON_EXIT_DAY = 19                    # switch away from melon on day 19; later plantings cannot close cleanly
WHEAT_EXIT_DAY = 26                    # day 25 is the last wheat planting day that can mature before season end
LIQUIDATION_DAY = 29                   # stop routine work, bank carried goods, and force final sales
LAND_TARGET_QUADRANTS = 3              # NW+NE+SW; SW is represented by the overflow crop slice above
LAND_UTILIZATION_TRIGGER = 0.65        # observed baseline fills 68% of NW; expand only once that capacity is reached
LAND_MIN_DAY = 0                       # earliest day to attempt BUY_LAND -- 0 = as soon as affordable (swept, best default)
MAX_ANIMAL_BUY_PER_DAY = None          # None = uncapped -- buy every affordable animal every day (swept, best default)
SELL_MIN_PRICE_FRAC = 0.4              # hold melon/milk/wool instead of dumping while price is below 40% of base --
                                        # confirmed win over dump-immediate (round12/14/15) once production volume is high enough to move the price
_BASE_PRICE = {"MELON": 250, "MILK": 160, "WOOL": 200, "CARROT": 35}  # RULES.md Object Types table -- for the sell-gate check above
SELLABLE_PRODUCTS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                     "EGG", "MILK", "WOOL", "FERTILIZER"}

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
CROP_MAX_YIELD_DAY = {"MELON": MELON_YIELD_DAY, "WHEAT": 4, "CARROT": 3}  # RULES.md Object Types table

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

# Crop pool: NW forward scan minus livestock's tiles, then NE and SW forward
# scans. The final twelve melon positions are in SW and remain locked until the
# third land purchase succeeds; the third crop hand then has the capacity to
# bring that new block into production without overloading the original crew.
# Melon claims the first MAX_MELON_TILES; wheat and carrot split whatever's
# left, in that order.
_claimed = set(LIVESTOCK_POSITIONS)
_NW_FREE = [(x, y) for y in range(0, 5) for x in range(0, 5) if (x, y) not in _claimed]
_NE_FORWARD = [(x, y) for y in range(0, 5) for x in range(5, 10)]
_SW_FORWARD = [(x, y) for y in range(5, 10) for x in range(0, 5)]
_ALL_FREE = _NW_FREE + _NE_FORWARD + _SW_FORWARD
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


def _owned_step_toward(pos, targets, tiles):
    """Shortest next step that stays on owned land.

    A newly hired hand may spawn on a locked shed tile; that starting square
    is allowed, but every square entered by the route must be owned.
    """
    targets = set(targets)
    if not targets or pos in targets:
        return None
    height, width = len(tiles), len(tiles[0])
    queue = [pos]
    first_step = {pos: None}
    for cur in queue:
        x, y = cur
        for action, nx, ny in (("WEST", x - 1, y), ("EAST", x + 1, y),
                               ("NORTH", x, y - 1), ("SOUTH", x, y + 1)):
            nxt = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height) or nxt in first_step:
                continue
            # A hand can spawn on any of the four shed-adjacent squares,
            # including a locked one. Permit locked shed squares only as the
            # short escape bridge; normal routing remains owned-land-only.
            if tiles[ny][nx] == "LOCKED" and nxt not in SHED_ADJACENT:
                continue
            first_step[nxt] = action if cur == pos else first_step[cur]
            if nxt in targets:
                return first_step[nxt]
            queue.append(nxt)
    return None


def _is_shed_adjacent(pos):
    return pos in SHED_ADJACENT


def _nearest_step(pos, positions, tiles):
    """Step toward the closest of `positions` (Manhattan), not just the
    first one found -- every wasted step here is an action turn not spent
    watering/feeding/harvesting.
    """
    if not positions:
        return None
    owned = [p for p in positions if tiles[p[1]][p[0]] != "LOCKED"]
    return _owned_step_toward(pos, owned, tiles)


def _return_to_shed(pos, inv, tiles):
    if not any(n > 0 for n in inv.values()):
        return None
    if _is_shed_adjacent(pos):
        return ["DROP"]
    owned_shed = [p for p in SHED_ADJACENT if tiles[p[1]][p[0]] != "LOCKED"]
    step = _owned_step_toward(pos, owned_shed, tiles)
    return [step] if step else ["PASS"]


# ---- Livestock unit action (adapted from agent_lib.py::_unit_action, ------
# simplified since both COW and SHEEP use PASTURE -- no COOP branching) -----

def _livestock_action(pos, inv, tiles, shed, idxs, liquidation=False):
    ux, uy = pos
    if liquidation:
        returning = _return_to_shed(pos, inv, tiles)
        if returning:
            return returning
        ready = []
        for i in idxs:
            x, y = LIVESTOCK_POSITIONS[i]
            t = tiles[y][x]
            if isinstance(t, dict) and t.get("animal") and t.get("yield_units", 0) > 0:
                ready.append((x, y))
        if (ux, uy) in ready:
            return ["HARVEST"]
        step = _nearest_step(pos, ready, tiles)
        return [step] if step else ["PASS"]

    cur_idx = next((i for i in idxs if LIVESTOCK_POSITIONS[i] == (ux, uy)), None)
    if cur_idx is not None:
        atype = LIVESTOCK_TYPES[cur_idx]
        x, y = LIVESTOCK_POSITIONS[cur_idx]
        tile = tiles[y][x]
        if tile == "LOCKED":
            pass
        elif tile is None:
            return ["BUILD_PASTURE"]
        elif tile.get("kind") == ANIMAL_STRUCTURE and not tile.get("animal"):
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
            owned_shed = [p for p in SHED_ADJACENT if tiles[p[1]][p[0]] != "LOCKED"]
            step = _owned_step_toward((ux, uy), owned_shed, tiles)
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
    step = _nearest_step((ux, uy), feed_needy, tiles) or _nearest_step((ux, uy), other_needy, tiles)
    if step:
        return [step]
    return ["PASS"]


# ---- One-time crop unit action (adapted from engine.py::_tile_action) -----
# Shared by MELON, WHEAT, and CARROT -- all three are the same "plant / dig
# weed / water until max_yield_day / harvest" one-time-crop mechanic, just
# with different max_yield_day and seed price. Ongoing crops (tomato,
# strawberry) would need a different function -- not used here.

def _crop_action(crop, max_yield_day, pos, inv, day, tiles, idxs, positions, plant_allowed=True, liquidation=False):
    ux, uy = pos
    if liquidation:
        returning = _return_to_shed(pos, inv, tiles)
        if returning:
            return returning
        ready = []
        for i in idxs:
            x, y = positions[i]
            t = tiles[y][x]
            if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("yield_units", 0) > 0:
                ready.append((x, y))
        if (ux, uy) in ready:
            return ["HARVEST"]
        step = _nearest_step(pos, ready, tiles)
        return [step] if step else ["PASS"]

    cur_idx = next((i for i in idxs if positions[i] == (ux, uy)), None)
    if cur_idx is not None:
        x, y = positions[cur_idx]
        tile = tiles[y][x]
        if tile is None and plant_allowed:
            return ["PLANT", crop]
        # A crop tile pool can reach into NE before land is bought (locked
        # tiles are passable, so a unit can end up standing on one); "LOCKED"
        # is a plain string, not a dict, so it must be excluded here or the
        # .get() calls below raise AttributeError.
        if isinstance(tile, dict):
            if tile.get("kind") == "WEED":
                return ["DIG"]
            if tile.get("kind") == "PLANT":
                tile_crop = tile.get("crop", crop)
                tile_max_yield_day = CROP_MAX_YIELD_DAY.get(tile_crop, max_yield_day)
                age = day - tile["planted_day"]
                if age >= tile_max_yield_day and tile["yield_units"] > 0:
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
        if t is None and plant_allowed:
            other_needy.append((x, y))
        elif isinstance(t, dict) and t.get("kind") == "WEED":
            other_needy.append((x, y))
        elif isinstance(t, dict) and t.get("kind") == "PLANT":
            tile_crop = t.get("crop", crop)
            tile_max_yield_day = CROP_MAX_YIELD_DAY.get(tile_crop, max_yield_day)
            age = day - t["planted_day"]
            if age >= tile_max_yield_day and t["yield_units"] > 0:
                other_needy.append((x, y))
            elif not t["watered_today"]:
                water_needy.append((x, y))
    step = _nearest_step((ux, uy), water_needy, tiles) or _nearest_step((ux, uy), other_needy, tiles)
    if step:
        return [step]
    return ["PASS"]


def _melon_action(pos, inv, day, tiles, idxs):
    # Past MELON_EXIT_DAY, empty tiles in this same pool get WHEAT instead --
    # no crew restructuring needed, the hand just keeps working its assigned
    # tiles. Already-growing melon is unaffected: HARVEST only fires once the
    # tile's real yield_units > 0 (set by the engine, not by max_yield_day
    # here), so passing WHEAT's shorter max_yield_day for a tile that's
    # actually still-growing melon can't trigger a premature harvest -- it
    # just keeps falling through to WATER, same as before.
    if day < MELON_EXIT_DAY:
        return _crop_action("MELON", MELON_YIELD_DAY, pos, inv, day, tiles, idxs, MELON_POSITIONS)
    return _crop_action("WHEAT", CROP_MAX_YIELD_DAY["WHEAT"], pos, inv, day, tiles, idxs,
                        MELON_POSITIONS, plant_allowed=day < WHEAT_EXIT_DAY,
                        liquidation=day >= LIQUIDATION_DAY)


def _wheat_crop_action(pos, inv, day, tiles, idxs):
    return _crop_action("WHEAT", CROP_MAX_YIELD_DAY["WHEAT"], pos, inv, day, tiles, idxs, WHEAT_POSITIONS,
                        plant_allowed=day < WHEAT_EXIT_DAY, liquidation=day >= LIQUIDATION_DAY)


def _carrot_action(pos, inv, day, tiles, idxs):
    return _crop_action("CARROT", CROP_MAX_YIELD_DAY["CARROT"], pos, inv, day, tiles, idxs, CARROT_POSITIONS,
                        plant_allowed=day < 27, liquidation=day >= LIQUIDATION_DAY)


def _svm_sell_frac(item, features):
    """Blend SELL_MIN_PRICE_FRAC toward the exported watcher's per-product
    increase/avoid scores: hold longer (raise the gate) when it favors
    growing `item`, sell sooner (lower the gate) when it favors avoiding it.
    Falls back to the static SELL_MIN_PRICE_FRAC on any missing/invalid signal.
    """
    if watcher_model is None or features is None:
        return SELL_MIN_PRICE_FRAC
    increase = watcher_model.score(f"product:{item}:increase", features)
    avoid = watcher_model.score(f"product:{item}:avoid", features)
    if increase is None and avoid is None:
        return SELL_MIN_PRICE_FRAC
    attractiveness = max(-1.0, min(1.0, (increase or 0.0) - (avoid or 0.0)))
    return max(0.0, min(1.0, SELL_MIN_PRICE_FRAC * (1.0 - attractiveness)))


# Kaggle's real submission loader (kaggle_environments/agent.py::get_last_callable)
# execs this file and picks the LAST top-level callable defined, not something
# named "agent" specifically -- so nothing may be defined below this function,
# or that (0-arg, non-observation-taking) callable silently becomes "the agent"
# instead, called with no args every turn.
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
    liquidating = day >= LIQUIDATION_DAY

    # Only HIRE truly needs hour 0 (hands are hired for the whole day, once).
    # Every other purchase type is spread across its own hour instead of all
    # piling into hour 0 -- maxMarketOrdersPerTurn (10) is a *per-turn* cap,
    # and with N_HIRES approaching double digits, HIRE alone can fill it,
    # silently dropping every other buy (and every sell) that day if they
    # all queued at hour 0 too. One purchase category per hour sidesteps the
    # cap entirely regardless of how many hands/crops are configured.
    prices = obs.get("market", {}).get("prices", {})

    svm_features = None
    if (shared_features is not None and watcher_model is not None and
            getattr(watcher_model, "SCHEMA_HASH", None) == shared_features.SCHEMA_HASH):
        try:
            svm_features = shared_features.extract_features(obs)
        except Exception:
            svm_features = None

    for item, n in list(shed.items()):
        if n <= 0 or item == "WHEAT" or item not in SELLABLE_PRODUCTS:
            continue
        base = _BASE_PRICE.get(item)
        cur = prices.get(item)
        sell_frac = _svm_sell_frac(item, svm_features)
        # Hold instead of dumping while the price is depressed below the
        # gate -- but never past shedCapacity (100): losing product to
        # overflow is strictly worse than selling it cheap.
        if not liquidating and sell_frac > 0 and base and cur is not None and cur < sell_frac * base and n < 90:
            continue
        sell_orders.append(["SELL", item, n])

    # Wheat barely reacts to glut (RULES.md Price Function table), so any
    # surplus beyond a 2-day feed reserve is free money to sell, not just a
    # cost saved on feed. Not gated on WHEAT_TILES (0 by default) because
    # MELON_EXIT_DAY routes ex-melon tiles into WHEAT too -- this check is
    # self-gating (wheat_surplus > 0) regardless of the source.
    wheat_reserve = 0 if liquidating else len(LIVESTOCK_TYPES) * 2
    wheat_surplus = shed.get("WHEAT", 0) - wheat_reserve
    if wheat_surplus > 0:
        sell_orders.append(["SELL", "WHEAT", wheat_surplus])

    # A static crew plan can include work in the next quadrant. Hire only the
    # usable prefix, preserving hand-slot/job alignment while avoiding wages
    # for a hand whose entire assignment is still locked.
    active_hires = 0
    for job, idxs in _CREW_PLAN:
        positions = LIVESTOCK_POSITIONS if job == "LIVESTOCK" else {
            "MELON": MELON_POSITIONS, "WHEAT": WHEAT_POSITIONS,
            "CARROT": CARROT_POSITIONS,
        }[job]
        if any(tiles[positions[i][1]][positions[i][0]] != "LOCKED" for i in idxs):
            active_hires += 1
        else:
            break

    if hour == 0:
        for _ in range(active_hires):
            buy_orders.append(["HIRE"])

    if hour == 1 and not liquidating:
        # Wheat feed is funded before any other spend, at the live market
        # price -- an existing herd starving because some other purchase ate
        # the day's cash is an unrecoverable death spiral (2 unfed days =
        # animal escapes for good), so feeding what's alive always wins.
        # Self-grown wheat (if any) already lowered `wheat_have`, so this
        # naturally only tops up whatever the wheat plot didn't cover.
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

    if hour == 2 and day < 21:
        # Buy missing COW/SHEEP up to ANIMAL_MIX, capped by affordability
        # and (if set) a per-day purchase cap.
        needed_counts = {}
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
            cost_each = ANIMAL_COST[atype]
            afford = int(cash // cost_each)
            n_buy = min(to_buy, afford, remaining_cap)
            if n_buy > 0:
                buy_orders.append(["BUY_ANIMAL", atype, n_buy])
                cash -= n_buy * cost_each
                remaining_cap -= n_buy

    owned_tiles = [t for row in tiles for t in row if t != "LOCKED"]
    occupied_owned = sum(t is not None for t in owned_tiles)
    land_utilization = occupied_owned / len(owned_tiles) if owned_tiles else 0
    unlocked_count = len(me.get("unlocked_quadrants", ["NW"]))
    expansion_ready = unlocked_count == 1 or land_utilization >= LAND_UTILIZATION_TRIGGER
    if (hour == 3 and not liquidating and day >= LAND_MIN_DAY
            and unlocked_count < LAND_TARGET_QUADRANTS and expansion_ready):
        # No money precheck: the engine silently no-ops an unaffordable
        # BUY_LAND, and re-issuing it every day until it lands is the exact
        # idiom every top-10 leaderboard agent uses (report.md) -- it
        # guarantees the purchase clears on the earliest affordable day with
        # no risk of a stale one-shot check missing the window.
        buy_orders.append(["BUY_LAND"])

    if hour == 4 and day < WHEAT_EXIT_DAY:
        empty_positions = [(x, y) for x, y in MELON_POSITIONS if tiles[y][x] is None]
        if day < MELON_EXIT_DAY:
            have_seeds = seeds.get("MELON", 0)
            if len(empty_positions) > have_seeds:
                buy_orders.append(["BUY_SEED", "MELON", len(empty_positions) - have_seeds])
        else:
            # Past the exit day, _melon_action plants WHEAT on this same
            # pool instead -- keep its seed supply matched here too.
            have_seeds = seeds.get("WHEAT", 0)
            if len(empty_positions) > have_seeds:
                buy_orders.append(["BUY_SEED", "WHEAT", len(empty_positions) - have_seeds])

    if hour == 5 and WHEAT_TILES > 0 and day < WHEAT_EXIT_DAY:
        empty_wheat = sum(1 for x, y in WHEAT_POSITIONS if tiles[y][x] is None)
        have_seeds = seeds.get("WHEAT", 0)
        if empty_wheat > have_seeds:
            buy_orders.append(["BUY_SEED", "WHEAT", empty_wheat - have_seeds])

    if hour == 6 and CARROT_TILES > 0 and day < 27:
        empty_carrot = sum(1 for x, y in CARROT_POSITIONS if tiles[y][x] is None)
        have_seeds = seeds.get("CARROT", 0)
        if empty_carrot > have_seeds:
            buy_orders.append(["BUY_SEED", "CARROT", empty_carrot - have_seeds])

    # Closing sales take precedence; at other times leave enough room after
    # buys and explicitly enforce the engine's silent 10-order limit.
    market = (sell_orders + buy_orders if liquidating else buy_orders + sell_orders)[:10]

    fx, fy = me["farmer"]
    farmer_inv = inventories[0] if inventories else {}
    farmer_idxs = LIVESTOCK_GROUPS[0] if LIVESTOCK_GROUPS else []
    farmer_action = (_livestock_action((fx, fy), farmer_inv, tiles, shed, farmer_idxs, liquidating)
                     if farmer_idxs else ["PASS"])

    _DISPATCH = {
        "LIVESTOCK": lambda pos, inv, idxs: _livestock_action(pos, inv, tiles, shed, idxs, liquidating),
        "MELON": lambda pos, inv, idxs: _melon_action(pos, inv, day, tiles, idxs),
        "WHEAT": lambda pos, inv, idxs: _wheat_crop_action(pos, inv, day, tiles, idxs),
        "CARROT": lambda pos, inv, idxs: _carrot_action(pos, inv, day, tiles, idxs),
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
