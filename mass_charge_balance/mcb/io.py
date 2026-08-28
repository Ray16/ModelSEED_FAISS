#!/usr/bin/env python
"""Shared ModelSEED biochemistry loaders + formula/stoichiometry parsing.

Used by every correction tier. Nothing here mutates the database.
"""
import glob
import os
import re
from collections import defaultdict

import pandas as pd

BIOCHEM = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"

# Well-known compound IDs used as balancing reagents.
CPD_PROTON = "cpd00067"   # H+   : formula H, charge +1
CPD_WATER = "cpd00001"    # H2O  : formula H2O, charge 0

# ChemAxon "unknown" sentinels seen in the charge column.
UNKNOWN_SENTINELS = {10000000, 10000001, 10000002, 9999999, 9999998,
                     -10000000, -10000001, -10000002, -9999999}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_compounds():
    frames = [pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
              for f in sorted(glob.glob(os.path.join(BIOCHEM, "compound_*.tsv")))]
    return pd.concat(frames, ignore_index=True)


def load_reactions(active_only=True):
    frames = [pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
              for f in sorted(glob.glob(os.path.join(BIOCHEM, "reaction_*.tsv")))]
    rxn = pd.concat(frames, ignore_index=True)
    if active_only:
        rxn = rxn[rxn["is_obsolete"].isin(["0", "false", "False", ""])].copy()
    return rxn


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------
# Real 1-2 letter elements only. Anything else (R, X, *, parens, lowercase n,
# polymer markers) -> generic/unscoreable -> return None.
_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_TOKEN_RE = re.compile(r"[A-Z][a-z]?\d*")


def parse_formula(formula):
    """dict element->int count, or None if formula is empty/generic/invalid."""
    if formula is None:
        return None
    f = formula.strip()
    if f in ("", "null", "noformula", "*"):
        return None
    # Generic markers => cannot mass-balance.
    if any(sym in f for sym in ("R", "X", "*", "(", ")", ".")):
        return None
    # Leftover chars after stripping valid element tokens => invalid (e.g. 'n').
    if _TOKEN_RE.sub("", f).strip():
        return None
    counts = defaultdict(int)
    for m in _ELEM_RE.finditer(f):
        sym, num = m.group(1), m.group(2)
        counts[sym] += int(num) if num else 1
    return dict(counts) if counts else None


# ---------------------------------------------------------------------------
# Stoichiometry parsing / serialization
#   format:  n:cpdid:cmpt:comm:"name"  joined by ';'
# ---------------------------------------------------------------------------
def parse_stoich(s):
    """List of dicts: {coeff, cpd, cmpt, comm, name}. Reactant coeff is negative.

    The actual DB format is 4-field  coeff:cpd:cmpt:"name"  (no community index),
    though REACTIONS.md documents an optional 5th field coeff:cpd:cmpt:comm:"name".
    We handle both: the quoted trailing token is always the name; any numeric field
    between cmpt and the name is the community index (defaults to "0").
    """
    out = []
    if not s or s == "null":
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        # Split the quoted name (if any) from the colon-delimited prefix.
        if '"' in part:
            first_q = part.index('"')
            last_q = part.rindex('"')
            name = part[first_q + 1:last_q]
            prefix = part[:first_q].rstrip(":")
        else:
            bits = part.rsplit(":", 1)
            prefix, name = bits[0], (bits[1] if len(bits) > 1 else "")
        fields = prefix.split(":")
        try:
            coeff = float(fields[0])
        except (ValueError, IndexError):
            continue
        cpd = fields[1] if len(fields) > 1 else ""
        cmpt = fields[2] if len(fields) > 2 else "0"
        comm = fields[3] if len(fields) > 3 else ""   # community index, usually absent
        out.append({"coeff": coeff, "cpd": cpd, "cmpt": cmpt, "comm": comm, "name": name})
    return out


def format_coeff(c):
    """Match ModelSEED's integer-when-whole formatting."""
    if abs(c - round(c)) < 1e-9:
        return str(int(round(c)))
    return f"{c:g}"


def is_valid_reaction(species):
    """Reject degenerate results a balance check would wrongly pass: an empty/one-species
    reaction (the fix cancelled everything) or an exact net no-op (reactants == products)."""
    sp = [s for s in species if abs(s["coeff"]) > 1e-9]
    if len(sp) < 2:
        return False
    react = sorted((s["cpd"], round(-s["coeff"], 6)) for s in sp if s["coeff"] < 0)
    prod = sorted((s["cpd"], round(s["coeff"], 6)) for s in sp if s["coeff"] > 0)
    if not react or not prod:
        return False
    if react == prod:
        return False
    return True


def render_equation(species, name_of=None):
    """Human-readable name equation for review: reactants <=> products."""
    react = [sp for sp in species if sp["coeff"] < 0]
    prod = [sp for sp in species if sp["coeff"] > 0]
    def label(sp):
        nm = sp.get("name") or (name_of.get(sp["cpd"]) if name_of else "") or sp["cpd"]
        return f'({format_coeff(abs(sp["coeff"]))}) {nm}'
    return f"{' + '.join(label(s) for s in react)} <=> {' + '.join(label(s) for s in prod)}"


def serialize_stoich(species):
    """Inverse of parse_stoich. Emits the DB's 4-field form coeff:cpd:cmpt:"name"
    unless a community index was present (then the documented 5-field form)."""
    parts = []
    for sp in species:
        comm = sp.get("comm", "")
        if comm not in ("", None):
            parts.append(f'{format_coeff(sp["coeff"])}:{sp["cpd"]}:{sp["cmpt"]}:{comm}:"{sp["name"]}"')
        else:
            parts.append(f'{format_coeff(sp["coeff"])}:{sp["cpd"]}:{sp["cmpt"]}:"{sp["name"]}"')
    return ";".join(parts)


# ---------------------------------------------------------------------------
# Balance computation
# ---------------------------------------------------------------------------
class SpeciesInfo:
    """Formula/charge provider for compound IDs, with an override hook (ChemAxon)."""

    def __init__(self, formula_of, charge_of):
        self.formula_of = formula_of              # cpd -> raw formula string
        self.charge_of = charge_of                # cpd -> int or None
        self._parsed = {}

    @classmethod
    def from_compounds(cls, cpd_df):
        formula_of = dict(zip(cpd_df["id"], cpd_df["formula"]))
        charge_of = {}
        for cid, ch in zip(cpd_df["id"], cpd_df["charge"]):
            try:
                v = int(float(ch))
                charge_of[cid] = None if v in UNKNOWN_SENTINELS else v
            except (ValueError, TypeError):
                charge_of[cid] = None
        return cls(formula_of, charge_of)

    def formula(self, cpd):
        if cpd not in self._parsed:
            self._parsed[cpd] = parse_formula(self.formula_of.get(cpd))
        return self._parsed[cpd]

    def charge(self, cpd):
        return self.charge_of.get(cpd)


def compute_residual(species, info):
    """Return (residual_dict, flags).

    residual maps element-symbol -> net (products - reactants) and 'charge' -> net.
    flags: {'no_formula': bool, 'unknown_charge': bool} for species we couldn't score.
    """
    residual = defaultdict(float)
    no_formula = False
    unknown_charge = False
    for sp in species:
        coeff, cpd = sp["coeff"], sp["cpd"]
        pf = info.formula(cpd)
        if pf is None:
            no_formula = True
        else:
            for el, n in pf.items():
                residual[el] += coeff * n
        ch = info.charge(cpd)
        if ch is None:
            unknown_charge = True
        else:
            residual["charge"] += coeff * ch
    residual = {k: v for k, v in residual.items() if abs(v) > 1e-6}
    return residual, {"no_formula": no_formula, "unknown_charge": unknown_charge}
