#!/usr/bin/env python
"""
Probe the three "hard" (currently out-of-scope) imbalance classes to size how much of each
is actually TRACTABLE with a mechanism signal (EC or a named carrier), vs genuinely
curation-level. Also quantify how much predicted-EC would unlock (the EC=null fraction).

Classes:
  A. O2-only hydroxylations   -- monooxygenase S + O2 + [reductant]H + H+ -> S-OH + H2O + ...
                                 imbalanced because the REDUCTANT is missing; which one is
                                 pinned by EC subclass (1.14.13 NADPH / 1.14.12 NADH / ...).
  B. redox charge-only        -- mass balances, charge is off by an even |dq|: an implicit
                                 electron transfer. Tractable IFF a SMALL-MOLECULE redox
                                 couple is pinned (EC 1.x.1 NAD(P) / 1.x.5 quinone / ...);
                                 protein carriers (ferredoxin/cytochrome) are not fixable.
  C. heavy-skeleton           -- C/N/P/S residual: a whole fragment missing. Tractable sub-
                                 pattern = residual equals ONE clean building block
                                 (hexose C6H10O5, pentose, acetyl, methylene, ...).

Read-only. Prints counts; writes nothing (diagnostic).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from collections import Counter
from mcb import io as msbio
from mcb import closures as cl

O2 = "cpd00007"
# reductant couples (reduced -> oxidized) keyed by the EC subclass that pins them
REDUCTANTS = {
    "1.14.13": ("cpd00005", "cpd00006", "NADPH/NADP"),   # NADPH
    "1.14.14": ("cpd00982", "cpd00015", "FADH2/FAD"),    # flavin (approx)
    "1.14.12": ("cpd00004", "cpd00003", "NADH/NAD"),     # NADH
}
# redox couples that are SMALL MOLECULES (fixable); pinned by EC sub-subclass ".1/.5/.3"
REDOX_SMALL = {
    "NAD":  ("cpd00004", "cpd00003"),
    "NADP": ("cpd00005", "cpd00006"),
}
PROTEIN_CARRIER_HINTS = ("ferredoxin", "cytochrome", "thioredoxin", "flavodoxin",
                         "acceptor", "donor", "reduced", "oxidized")
# common building blocks a heavy residual might equal (missing polymer/transfer unit)
BUILDING_BLOCKS = {
    "hexose C6H10O5": {"C": 6, "H": 10, "O": 5},
    "pentose C5H8O4": {"C": 5, "H": 8, "O": 4},
    "deoxyhexose C6H10O4": {"C": 6, "H": 10, "O": 4},
    "acetyl C2H2O":  {"C": 2, "H": 2, "O": 1},
    "methylene CH2": {"C": 1, "H": 2},
    "CO2 C1O2":      {"C": 1, "O": 2},
    "formyl CHO":    {"C": 1, "H": 1, "O": 1},
}


def ec_prefixes(ec):
    return [e.strip() for e in ec.split("|")] if ec not in ("", "null") else []


def sig_equals_block(res):
    heavy = {k: v for k, v in res.items() if k not in ("charge",)}
    for name, bl in BUILDING_BLOCKS.items():
        for sign in (1, -1):
            if all(abs(heavy.get(k, 0) - sign * v) < 1e-6 for k, v in bl.items()) and \
               all(abs(heavy.get(k, 0)) < 1e-6 for k in heavy if k not in bl):
                return name
    return None


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))

    A = Counter(); A_ec = Counter()
    B = Counter(); B_names = Counter()
    C = Counter(); C_blocks = Counter()
    ec_null_total = Counter()

    for _, r in rxn.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry"])
        if cl.is_refused_reaction(sp):
            continue
        res, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res:
            continue
        if cl.proton_water_close(res) is not None:
            continue  # already handled by stage 1
        ec = r["ec_numbers"]
        heavy = {k for k in res if k not in ("H", "O", "charge")}
        cpds = {s["cpd"] for s in sp}

        # ---- A: O2-only hydroxylation (O2 present, reductant missing) ----
        if O2 in cpds and heavy <= set():   # only H/O/charge residual w/ O2 present
            A["total(O2 present, H/O/charge resid)"] += 1
            pinned = [lbl for pre, (red, ox, lbl) in REDUCTANTS.items()
                      if any(e.startswith(pre) for e in ec_prefixes(ec))]
            if pinned:
                A_ec[pinned[0]] += 1
                A["EC-pinned reductant"] += 1
            elif ec in ("", "null"):
                A["EC=null (predicted-EC would help)"] += 1
                ec_null_total["A"] += 1
            else:
                A["other EC"] += 1

        # ---- B: redox charge-only (mass ok, charge off, even) ----
        elif not heavy and res.get("charge") and abs(res["charge"]) % 2 == 0 \
                and abs(res.get("H", 0)) < 1e-6 and abs(res.get("O", 0)) < 1e-6:
            B["total(charge-only, even)"] += 1
            nm = (r["name"] or "").lower()
            has_small = any(red in cpds or ox in cpds for red, ox in REDOX_SMALL.values())
            if any(h in nm for h in PROTEIN_CARRIER_HINTS):
                B["protein-carrier (NOT fixable)"] += 1
            elif has_small:
                B["small-molecule couple present (TRACTABLE)"] += 1
                B_names["small-couple"] += 1
            elif ec in ("", "null"):
                B["EC=null (predicted-EC would help)"] += 1
                ec_null_total["B"] += 1
            else:
                B["other"] += 1

        # ---- C: heavy-skeleton ----
        elif heavy:
            C["total(heavy residual)"] += 1
            blk = sig_equals_block(res)
            if blk:
                C["residual == one building block (TRACTABLE sub-pattern)"] += 1
                C_blocks[blk] += 1
            else:
                C["multi/other skeleton residual"] += 1

    def show(title, ctr, extra=None):
        print(f"\n===== {title} =====")
        for k, v in ctr.most_common():
            print(f"  {k:52s} {v}")
        if extra:
            for k, v in extra.most_common():
                print(f"       - {k:30s} {v}")

    show("A. O2-only hydroxylations (missing reductant)", A, A_ec)
    show("B. redox charge-only (implicit electron transfer)", B)
    show("C. heavy-skeleton errors", C, C_blocks)
    print(f"\n>>> EC=null reactions in the tractable A+B sub-patterns "
          f"(predicted EC would directly unlock): A={ec_null_total['A']} B={ec_null_total['B']}")


if __name__ == "__main__":
    main()
