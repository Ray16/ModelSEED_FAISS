"""
Balance-closure primitives + chemistry guards, shared by every correction stage.

Before this module these helpers were copy-pasted across cosubstrate_closure,
couple_closure, apply_couples, coverage_audit and menu_expansion, so a guard added in
one place silently failed to protect the others. They now live here once.

Two layers:

  CLOSURES  — the deterministic arithmetic. Given a residual (products - reactants) in
              element/charge space, decide whether a small, physically-meaningful set of
              free species closes it exactly:
                proton_water_close   H+/H2O only               (w = -dO, p = -dq)
                single_closes        one heavy co-substrate  + proton/water
                couple_closes        one donor/acceptor pair + proton/water
              HEAVY (co-substrate menu) and COUPLES (donor/acceptor menu) are the fixed
              vocabularies of "free" species curators routinely omit.

  GUARDS    — refuse balanced-but-wrong fixes that pure arithmetic would accept:
                is_refused_reaction        indefinite-length polymer / generic species
                                           (carries a placeholder formula -> a fixed-
                                           stoichiometry balance is meaningless)
                phosphate_cosubstrate_conflict
                                           ATP->ADP releases orthophosphate (Pi), not PPi;
                                           ATP->AMP releases PPi, not Pi
                ec_compatible              no transferase couple on an oxidoreductase
"""
import re

# --------------------------------------------------------------------------- #
# Free-species vocabularies
# --------------------------------------------------------------------------- #
WATER = ({"H": 2, "O": 1}, 0)
PROTON = ({"H": 1}, 1)
CPD_WATER = "cpd00001"
CPD_PROTON = "cpd00067"

# Commonly-omitted "free" heavy co-substrates: cpd id -> (name, formula dict, charge).
# pH-7 microspecies as stored in ModelSEED.
HEAVY = {
    "cpd00011": ("CO2",     {"C": 1, "O": 2},          0),
    "cpd00009": ("Pi",      {"H": 1, "O": 4, "P": 1}, -2),
    "cpd00012": ("PPi",     {"H": 1, "O": 7, "P": 2}, -3),
    "cpd00013": ("NH4",     {"H": 4, "N": 1},          1),
    "cpd00048": ("Sulfate", {"O": 4, "S": 1},         -2),
}

# EC top-classes whose enzyme genuinely RELEASES/CONSUMES the free co-substrate, so a
# single-species closure is biochemically corroborated (not just arithmetically balanced).
# Deliberately EXCLUDES ATP-dependent kinases (2.7.1/2.7.4): they transfer phosphate FROM
# ATP, so a "missing Pi" there is really a missing ATP->ADP pair -> couple stage.
EC_CORROBORATION = {
    "CO2":     ("4.1.1", "6.4.1"),
    "Pi":      ("3.1.3", "3.6.1", "3.1.4"),
    "PPi":     ("6.", "2.7.7", "3.6.1"),
    "NH4":     ("3.5.1", "3.5.4", "4.3.1"),
    "Sulfate": ("2.8.2", "3.1.6"),
}

# Cofactor couples: name -> (donor_cpd, acceptor_cpd, EC-prefix corroboration tuple).
# donor is the "loaded" carrier (consumed); acceptor is the "unloaded" product.
COUPLES = {
    "methyl (SAM/SAH)":      ("cpd00017", "cpd00019", ("2.1.1",)),
    "glucosyl (UDPglc/UDP)": ("cpd00026", "cpd00014", ("2.4.1",)),
    "mannosyl (GDPman/GDP)": ("cpd00083", "cpd00031", ("2.4.1",)),
    "acetyl (AcCoA/CoA)":    ("cpd00022", "cpd00010", ("2.3.1",)),
    "phospho ATP/ADP":       ("cpd00002", "cpd00008", ("2.7.1", "2.7.2", "2.7.4")),
    "phospho GTP/GDP":       ("cpd00038", "cpd00031", ("2.7.1", "2.7.4")),
    "phospho UTP/UDP":       ("cpd00062", "cpd00014", ("2.7.1", "2.7.4")),
    "adenylyl ATP/AMP":      ("cpd00002", "cpd00018", ("2.7.7", "6.")),
}


# --------------------------------------------------------------------------- #
# Closure arithmetic
# --------------------------------------------------------------------------- #
def proton_water_close(res):
    """Return (w, p) closing residual `res` with water/protons, or None.

    w waters close O, p protons close charge, and H must then close on its own:
        w = -dO,  p = -dq,  require dH + 2w + p == 0.
    Any heavy element (not H/O/charge) remaining -> not closeable here.
    """
    if any(e not in ("H", "O", "charge") for e in res):
        return None
    dH, dO, dq = res.get("H", 0.0), res.get("O", 0.0), res.get("charge", 0.0)
    if any(abs(x - round(x)) > 1e-6 for x in (dH, dO, dq)):
        return None
    w, p = -round(dO), -round(dq)
    if abs(dH + 2 * w + p) > 1e-6:
        return None
    return w, p


# back-compat alias (couple_closure historically called this `pw_close`)
pw_close = proton_water_close


def add_species(res, formula, charge, k):
    """Residual after adding k copies (product side) of a species with `formula`/`charge`."""
    out = dict(res)
    for el, n in formula.items():
        out[el] = out.get(el, 0) + k * n
    out["charge"] = out.get("charge", 0) + k * charge
    return {e: v for e, v in out.items() if abs(v) > 1e-6}


# back-compat aliases (cosubstrate_closure used `residual_after`; couple_closure used `add`)
residual_after = add_species
add = add_species


def single_closes(res0, formula, charge, krange=(-3, -2, -1, 1, 2, 3)):
    """First k in krange for which one heavy species + proton/water closes res0, else None."""
    for k in krange:
        if proton_water_close(add_species(res0, formula, charge, k)) is not None:
            return k
    return None


def couple_closes(res0, donor, acceptor, info):
    """True iff inserting the donor/acceptor couple (+proton/water) closes res0.

    `info` is an mcb.io.SpeciesInfo; donor/acceptor are cpd ids.
    """
    df_, dq_ = info.formula(donor), info.charge(donor)
    af_, aq_ = info.formula(acceptor), info.charge(acceptor)
    if None in (df_, dq_, af_, aq_):
        return False
    for k in (1, -1):
        if pw_close(add_species(add_species(res0, af_, aq_, k), df_, dq_, -k)) is not None:
            return True
    return False


# --------------------------------------------------------------------------- #
# Guards — refuse balanced-but-wrong corrections
# --------------------------------------------------------------------------- #
# Species whose NAME encodes an indefinite repeat count. ModelSEED assigns these a fixed
# placeholder formula (e.g. "(Phosphate)n" -> HO10P3), so parse_formula accepts them, yet a
# fixed-stoichiometry balance against a variable-length polymer is meaningless. Refuse the
# whole reaction. Kept precise so defined compounds ("core oligosaccharide-lipid A", a named
# structure) are NOT swept up: only the (X)n / (X)m forms and the poly/oligo-phosphate names.
_GENERIC_NAME_RE = re.compile(
    r"\)\s*n\b|\(\s*n\s*\)|\)\s*m\b|"        # (X)n , (n), (X)m
    r"\bpoly[- ]?phosphate\b|\boligophosphate\b",
    re.I,
)


def is_generic_species(name):
    """True if a species name marks an indefinite-length polymer / generic residue."""
    return bool(name) and bool(_GENERIC_NAME_RE.search(name))


def is_refused_reaction(species):
    """Refuse a reaction that contains any indefinite-polymer / generic species."""
    return any(is_generic_species(s.get("name", "")) for s in species)


# Phosphate-donor mechanism. A nucleoside-triphosphate -> -diphosphate couple (ATP->ADP,
# GTP->GDP, UTP->UDP) releases orthophosphate (Pi); a -triphosphate -> -monophosphate couple
# (ATP->AMP, GTP->GMP) releases pyrophosphate (PPi). Adding the other one balances by luck.
_TP_LIKE = {"cpd00002", "cpd00038", "cpd00062"}   # ATP, GTP, UTP
_DP_LIKE = {"cpd00008", "cpd00031", "cpd00014"}   # ADP, GDP, UDP
_MP_LIKE = {"cpd00018", "cpd00126"}               # AMP, GMP
CPD_PPI, CPD_PI = "cpd00012", "cpd00009"


def phosphate_cosubstrate_conflict(reaction_cpds, added_cpd):
    """True if injecting `added_cpd` (Pi or PPi) contradicts the nucleotide couple present.

    reaction_cpds: set of cpd ids already in the reaction.
    """
    di = bool(reaction_cpds & _TP_LIKE) and bool(reaction_cpds & _DP_LIKE)   # ..TP -> ..DP
    mono = bool(reaction_cpds & _TP_LIKE) and bool(reaction_cpds & _MP_LIKE)  # ..TP -> ..MP
    if added_cpd == CPD_PPI and di and not mono:
        return True    # ADP present -> orthophosphate expected, not pyrophosphate
    if added_cpd == CPD_PI and mono and not di:
        return True    # AMP present -> pyrophosphate expected, not orthophosphate
    return False


def couple_ec_classes(cname):
    """EC top-classes where this couple's chemistry lives (from its corroboration prefixes)."""
    return {p.split(".")[0] for p in COUPLES[cname][2]}


def ec_compatible(ec, cname):
    """A couple may be offered only if the reaction's EC is null OR shares the couple's
    top-class -- prevents offering a transferase couple (class 2) to an oxidoreductase
    (class 1), the 2-OG-dioxygenase-as-methylation failure KEGG caught."""
    if ec in ("", "null"):
        return True
    home = couple_ec_classes(cname)
    return any(e.strip().split(".")[0] in home for e in ec.split("|"))
