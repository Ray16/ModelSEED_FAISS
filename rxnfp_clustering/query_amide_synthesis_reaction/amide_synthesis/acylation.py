"""Decide whether a reaction N-acylates a free amine, and identify WHICH amine.

Weaker screens fail in both directions:

* Net carboxamide count misses asparagine synthetase (an amide is consumed as
  one is made) and, worse here, misses acetyl-CoA:putrescine
  N-acetyltransferase, because a new acetamide's radius-2 environment is
  identical to CoA's own pantetheine amides.
* "A free amine disappears and some product contains an amide" is too
  permissive. Alanine dehydrogenase (rxn00278) passes it - alanine loses its
  amine while NADH coincidentally supplies an unrelated amide.

The test here asks the question directly: CUT the amide C-N bonds in a product,
reconstitute the parent amine on the N side, and check whether that amine is
itself a reactant. If so, that reactant's nitrogen was acylated. This proves the
linkage and names the acceptor, so the diamine flag describes the molecule that
actually reacted rather than whichever reactant carries the most nitrogens.

Two refinements matter:

* The SPECIFIC nitrogen is validated, not just the molecule. Checking only that
  the acceptor molecule contains some eligible amine would accept acylation of a
  guanidino nitrogen whenever an unrelated aliphatic amine sits elsewhere in the
  same reactant.
* Cut sets up to MAX_CUTS bonds are tried, not one bond at a time. A reaction
  acylating BOTH ends of hexamethylenediamine leaves, under any single cut, a
  mono-acylated fragment that cannot match the free diamine; only cutting both
  amides recovers it.

Comparison uses a protonation-insensitive key, because ModelSEED writes a free
amine as [NH3+] while the same nitrogen is neutral once acylated.
"""
from __future__ import annotations

from itertools import combinations

from rdkit import Chem

AMIDE = Chem.MolFromSmarts('[CX3](=[OX1])[NX3]')
# A nitrogen that could attack an activated acyl group: sp3, H-bearing, bonded to
# sp3 carbon, not aromatic, amide, nitrile, nitro or guanidino.
FREE_AMINE = Chem.MolFromSmarts(
    '[NX3H1,NX3H2,NX4H2+,NX4H3+;$([NX3,NX4+][CX4]);'
    '!$([NX3][CX3]=[OX1,SX1,NX2]);!$([NX3]C#N);!$([NX3+](=O)[O-]);'
    '!$([NX3][CX3](=[NX2])[NX3]);!$([NX3][CX3](=[NX2+])[NX3]);!$(N=*);!a]')

MAX_CUTS = 2          # simultaneous acylations considered (e.g. both ends of HMDA)
MAX_AMIDES = 8        # above this, only single cuts, to bound the combinatorics


def neutral_key(mol) -> str | None:
    """Canonical SMILES with N/O formal charges removed, for matching across pH."""
    if mol is None:
        return None
    edit = Chem.RWMol(mol)
    for atom in edit.GetAtoms():
        if atom.GetSymbol() in ('N', 'O') and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
    try:
        out = edit.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return None


def count_free_amines(mol) -> int:
    return len(mol.GetSubstructMatches(FREE_AMINE)) if mol is not None else 0


def _cuttable_amides(mol):
    """(bond index, nitrogen index, carbonyl carbon index) per non-ring C-N amide."""
    out = []
    for match in mol.GetSubstructMatches(AMIDE):
        carbon, nitrogen = match[0], match[2]
        bond = mol.GetBondBetweenAtoms(carbon, nitrogen)
        if bond is None or bond.IsInRing():   # a lactam was not built from a free amine
            continue
        out.append((bond.GetIdx(), nitrogen, carbon))
    return out


def _carboxylic_acid_from_acyl(piece, carbon):
    """Reconstitute the parent free acid from a cut acyl fragment.

    The amide C-N bond was cut, so the carbonyl carbon is one oxygen short of
    the acid it came from. Adding that hydroxyl recovers the donor's acyl
    moiety as a free carboxylic acid, which is what the substrate-distance
    score compares against adipate.
    """
    try:
        edit = Chem.RWMol(piece)
        atom = edit.GetAtomWithIdx(carbon)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        oxygen = edit.AddAtom(Chem.Atom(8))
        edit.AddBond(carbon, oxygen, Chem.BondType.SINGLE)
        out = edit.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def acylation_cuts(product_mol):
    """Amide disconnections of a product, both halves named.

    Yields (amine_fragment, nitrogen index within it, acyl_fragment_as_acid).
    The acyl half is reconstituted as its free carboxylic acid so that an
    acyl-CoA thioester and a free diacid donating the same acyl group are
    described identically.
    """
    results = []
    if product_mol is None:
        return results
    amides = _cuttable_amides(product_mol)
    if not amides:
        return results

    widths = [1] if len(amides) > MAX_AMIDES else range(1, min(MAX_CUTS, len(amides)) + 1)
    for width in widths:
        for subset in combinations(amides, width):
            bond_ids = [b for b, _, _ in subset]
            try:
                fragmented = Chem.FragmentOnBonds(product_mol, bond_ids,
                                                  addDummies=False)
                membership = []
                pieces = Chem.GetMolFrags(fragmented, asMols=True,
                                          sanitizeFrags=False, frags=membership)
            except Exception:
                continue
            if len(pieces) < 2:
                continue                      # intramolecular; nothing separated
            for _, nitrogen, carbon in subset:
                which = membership[nitrogen]
                # GetMolFrags preserves original atom order within a fragment, so
                # the nitrogen's new index is its rank among that fragment's atoms.
                local = sum(1 for i in range(nitrogen) if membership[i] == which)
                piece = Chem.Mol(pieces[which])
                try:
                    Chem.SanitizeMol(piece)
                except Exception:
                    continue
                acyl = None
                acyl_which = membership[carbon]
                # A doubly-acylated diamine cut at both ends can leave the
                # carbonyl in the same fragment as the nitrogen; that is not a
                # separable donor.
                if acyl_which != which:
                    local_c = sum(1 for i in range(carbon)
                                  if membership[i] == acyl_which)
                    acyl = _carboxylic_acid_from_acyl(
                        Chem.Mol(pieces[acyl_which]), local_c)
                results.append((piece, local, acyl))
    return results


def reconstituted_amines(product_mol):
    """Parent amines recovered by cutting one or several amide C-N bonds.

    Yields (fragment, index of the formerly-acylated nitrogen within it).
    """
    return [(piece, local) for piece, local, _ in acylation_cuts(product_mol)]


def acylation_events(rxn_smiles, max_heavy=60):
    """Every proven N-acylation in a reaction, with BOTH partners named.

    Returns a list of dicts: the amine acceptor that actually reacted, the acyl
    moiety grafted onto it (as the free acid), and whether the two came from the
    same species - the self-condensation signature of an iterative, chain-
    extending polyamide synthetase.

    `acylated_acceptors` is the acceptor-only view of the same analysis and is
    kept as the screen's stable interface.
    """
    lhs, rhs = rxn_smiles.split('>>')
    reactants = {}
    for smiles in lhs.split('.'):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() > max_heavy:
            continue
        amines = count_free_amines(mol)
        if amines < 1:
            continue
        key = neutral_key(mol)
        if key:
            reactants[key] = (smiles, amines)
    if not reactants:
        return []

    product_keys = set()
    for smiles in rhs.split('.'):
        key = neutral_key(Chem.MolFromSmiles(smiles))
        if key:
            product_keys.add(key)

    events = {}
    for smiles in rhs.split('.'):
        product = Chem.MolFromSmiles(smiles)
        if product is None:
            continue
        for piece, nitrogen, acyl in acylation_cuts(product):
            key = neutral_key(piece)
            if key is None or key not in reactants:
                continue
            # An unchanged spectator is not an acceptor.
            if key in product_keys:
                continue
            # Validate THE nitrogen that was acylated, not merely the molecule:
            # otherwise acylation of a guanidino N counts whenever an unrelated
            # aliphatic amine happens to sit elsewhere in the same reactant.
            eligible = {match[0] for match in piece.GetSubstructMatches(FREE_AMINE)}
            if nitrogen not in eligible:
                continue
            acceptor, amines = reactants[key]
            acyl_smiles = Chem.MolToSmiles(acyl) if acyl is not None else None
            slot = events.setdefault(
                (acceptor, acyl_smiles),
                {'acceptor': acceptor, 'acceptor_free_amines': amines,
                 'acyl_acid': acyl_smiles, 'product': smiles})
            slot['acceptor_free_amines'] = max(slot['acceptor_free_amines'], amines)
    return sorted(events.values(), key=lambda e: -e['acceptor_free_amines'])


def acylated_acceptors(rxn_smiles, max_heavy=60):
    """Reactant amines that end up acylated in the products.

    Returns [(acceptor_smiles, n_free_amines_on_acceptor)], richest first.
    """
    best = {}
    for event in acylation_events(rxn_smiles, max_heavy=max_heavy):
        acceptor = event['acceptor']
        best[acceptor] = max(best.get(acceptor, 0), event['acceptor_free_amines'])
    return sorted(best.items(), key=lambda kv: -kv[1])
