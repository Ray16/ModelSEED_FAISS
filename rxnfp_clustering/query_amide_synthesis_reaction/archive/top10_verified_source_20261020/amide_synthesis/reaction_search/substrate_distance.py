"""Score the reacting partners for resemblance to adipate/HMDA chemistry.

The heuristic combines functional architecture, retained reactive ends, spacer
lengths, backbone and scaffold size. The legacy score and the explicit AA/BB
nylon-6,6 architecture score remain separate; neither predicts activity or yield.
"""
from __future__ import annotations

import os

from rdkit import Chem

from amide_synthesis.reaction_search.acylation import FREE_AMINE, acylation_events, count_free_amines, neutral_key

# Nylon-6,6 reference geometry.
# THE TARGET IS THESE FOUR NUMBERS, and nothing else in this module is
# specific to nylon-6,6. They are environment-overridable so the pipeline can
# be aimed at another polyamide without editing code - the screens, the
# routes and the ranking are all target-agnostic once these are set.
#
#   polyamide        donor            acceptor          spacers   heavy atoms
#   nylon-6,6        adipate C6       HMDA C6            4 / 6     10 / 8
#   nylon-6,10       sebacate C10     HMDA C6            8 / 6     14 / 8
#   nylon-6,12       dodecanedioate   HMDA C6           10 / 6     16 / 8
#   nylon-4,6        adipate C6       putrescine C4      4 / 4     10 / 6
#
# Nylon-6 and nylon-12 are self-condensations of one bifunctional monomer, so
# they need the nylon-6 reaction as the SEARCH QUERY rather than different
# constants here; see the README. Defaults are unchanged, and a control pins
# the reference scores so an override cannot silently become the default.
ADIPATE_SPACER = int(os.environ.get('TARGET_ACID_SPACER', 4))
HMDA_SPACER = int(os.environ.get('TARGET_AMINE_SPACER', 6))

CARBOXYLIC_ACID = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1,OX1-]')
N_HYDROXY = Chem.MolFromSmarts('[NX3][OX2H1]')
AMIDE_N = Chem.MolFromSmarts('[NX3][CX3]=[OX1]')

# Component weights; they sum to 1.0 so the score reads as a fraction.
W_POLYCONDENSATION = 0.35
W_PRODUCT_REACTIVE = 0.10
W_SPACER = 0.20
W_BACKBONE = 0.15
W_PARSIMONY = 0.20

# Heavy atoms in the nylon-6,6 monomers; substrates much larger than these are
# mostly scaffold the active site must also accommodate.
ADIPATE_HEAVY = int(os.environ.get('TARGET_ACID_HEAVY_ATOMS', 10))
HMDA_HEAVY = int(os.environ.get('TARGET_AMINE_HEAVY_ATOMS', 8))
# Tolerance in heavy atoms before the excess is charged in full.
PARSIMONY_SLACK = 6.0

# Spacer tolerance: a one-carbon mismatch keeps most of the credit, a
# three-carbon mismatch keeps little.
SPACER_DECAY = 2.0


def _mol(smiles):
    return Chem.MolFromSmiles(smiles) if smiles else None


def _acid_carbons(mol):
    """Indices of carboxyl carbons (protonated or deprotonated)."""
    return sorted({match[0] for match in mol.GetSubstructMatches(CARBOXYLIC_ACID)})


def _amine_nitrogens(mol):
    return sorted({match[0] for match in mol.GetSubstructMatches(FREE_AMINE)})


def _spacer(mol, start, end):
    """Carbons strictly between two anchor atoms along the shortest path.

    Returns (count, path). The count is what adipate and HMDA are measured in:
    adipate's two carboxyl carbons are separated by 4, HMDA's two nitrogens
    by 6.
    """
    if mol is None or start == end:
        return None, ()
    path = Chem.GetShortestPath(mol, start, end)
    if not path:
        return None, ()
    return len(path) - 2, path


def _widest_spacer(mol, anchors):
    """The longest spacer between any two anchors - the monomer's full span.

    A triacid like citrate has three carboxyls; the pair that defines its
    'length' is the most separated one, which is also the pair a condensation
    would use to extend a chain.
    """
    best, best_path = None, ()
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            count, path = _spacer(mol, a, b)
            if count is not None and (best is None or count > best):
                best, best_path = count, path
    return best, best_path


def _parsimony(mol, reference_heavy):
    """1.0 at the nylon monomer's size, decaying as scaffold is added.

    Measures the molecule, not the motif: this is the only term that notices
    a pterin, a UDP-sugar, a CoA or a nucleotide hanging off an otherwise
    perfect amide.
    """
    if mol is None:
        return 0.0
    excess = max(0, mol.GetNumHeavyAtoms() - reference_heavy)
    return float(PARSIMONY_SLACK / (PARSIMONY_SLACK + excess))


def _spacer_credit(observed, reference):
    """1.0 at the nylon spacing, decaying with absolute carbon mismatch."""
    if observed is None:
        return 0.0
    return float(SPACER_DECAY / (SPACER_DECAY + abs(observed - reference)))


def _backbone_penalty(mol, path):
    """Fraction of the spacer that is NOT plain aliphatic carbon.

    Amide nitrogens and their carbonyls are exempt: a growing polyamide chain
    is made of them, so an internal amide is the target motif, not a defect.
    Rings, aromatics, branch substituents and N-hydroxyls are deviations.
    """
    if mol is None or not path:
        return 1.0
    amide_nitrogens = {match[0] for match in mol.GetSubstructMatches(AMIDE_N)}
    interior = path[1:-1]
    if not interior:
        return 0.0
    bad = 0
    for idx in interior:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsAromatic() or atom.IsInRing():
            bad += 1
        elif atom.GetSymbol() == 'C':
            # A carbonyl carbon belonging to a chain amide is part of the motif;
            # any other heteroatom substituent is a deviation.
            hetero = [n for n in atom.GetNeighbors()
                      if n.GetSymbol() not in ('C', 'H') and n.GetIdx() not in path]
            carbonyl_of_amide = any(n.GetIdx() in amide_nitrogens
                                    for n in atom.GetNeighbors())
            if hetero and not carbonyl_of_amide:
                bad += 1
        elif atom.GetIdx() in amide_nitrogens:
            if mol.GetSubstructMatches(N_HYDROXY) and any(
                    n.GetSymbol() == 'O' for n in atom.GetNeighbors()
                    if n.GetIdx() not in path):
                bad += 1          # N-hydroxylated amide: the NIS monomer's gap
        else:
            bad += 1
    return bad / len(interior)


def describe_event(event):
    """Chemical features of one proven acylation. Pure function, no I/O."""
    acceptor = _mol(event.get('acceptor'))
    acyl = _mol(event.get('acyl_acid'))
    product = _mol(event.get('product'))

    out = {
        'acceptor_smiles': event.get('acceptor'),
        'acyl_acid_smiles': event.get('acyl_acid'),
        'acceptor_free_amines': int(event.get('acceptor_free_amines', 0)),
        'acyl_n_acids': 0,
        'self_condensation': False,
        'acceptor_bifunctional': False,
        'aa_bb_pair': False,
        'product_retains_amine': False,
        'product_retains_acid': False,
        'amine_spacer': None,
        'acid_spacer': None,
        'backbone_penalty': 1.0,
        'acceptor_heavy_atoms': None,
        'acyl_heavy_atoms': None,
        'donor_parent_heavy_atoms': None,
        'parsimony': 0.0,
    }
    if acceptor is None:
        return out

    acceptor_acids = _acid_carbons(acceptor)
    acceptor_amines = _amine_nitrogens(acceptor)
    out['acceptor_bifunctional'] = bool(acceptor_acids and acceptor_amines)

    if acyl is not None:
        acyl_acids = _acid_carbons(acyl)
        out['acyl_n_acids'] = len(acyl_acids)
        # The reconstituted acyl acid counts the bond just cut, so a donor that
        # was a diacid shows two carboxyls here and a monoacid shows one.
        out['aa_bb_pair'] = (len(acyl_acids) >= 2
                             and out['acceptor_free_amines'] >= 2)
        # Same species on both sides of the bond: iterative self-condensation.
        key_a, key_b = neutral_key(acceptor), neutral_key(acyl)
        out['self_condensation'] = bool(key_a and key_b and key_a == key_b)
        out['acid_spacer'], acid_path = _widest_spacer(acyl, acyl_acids)
        if out['self_condensation'] and acceptor_acids and acceptor_amines:
            # For an AB monomer the relevant span is amine-to-acid, not acid-to-acid.
            out['acid_spacer'], acid_path = _spacer(
                acceptor, acceptor_amines[0], acceptor_acids[0])
    else:
        acid_path = ()

    if len(acceptor_amines) >= 2:
        out['amine_spacer'], amine_path = _widest_spacer(acceptor, acceptor_amines)
    elif out['self_condensation'] and acceptor_acids and acceptor_amines:
        out['amine_spacer'], amine_path = _spacer(
            acceptor, acceptor_amines[0], acceptor_acids[0])
    else:
        out['amine_spacer'], amine_path = None, ()

    if product is not None:
        out['product_retains_amine'] = count_free_amines(product) > 0
        out['product_retains_acid'] = bool(_acid_carbons(product))

    out['acceptor_heavy_atoms'] = acceptor.GetNumHeavyAtoms()
    out['acyl_heavy_atoms'] = acyl.GetNumHeavyAtoms() if acyl is not None else None
    # Charge for the PARENT donor, which still carries the scaffold.
    donor_heavy = event.get('donor_parent_heavy_atoms')
    if donor_heavy is None and acyl is not None:
        donor_heavy = acyl.GetNumHeavyAtoms()
    out['donor_parent_heavy_atoms'] = donor_heavy
    donor_credit = (PARSIMONY_SLACK
                    / (PARSIMONY_SLACK + max(0, (donor_heavy or 0) - ADIPATE_HEAVY))
                    if donor_heavy else 0.0)
    out['parsimony'] = round(0.5 * (_parsimony(acceptor, HMDA_HEAVY)
                                    + donor_credit), 4)

    penalties = [p for p in (
        _backbone_penalty(acceptor, amine_path) if amine_path else None,
        _backbone_penalty(acyl or acceptor, acid_path) if acid_path else None,
    ) if p is not None]
    out['backbone_penalty'] = sum(penalties) / len(penalties) if penalties else 1.0
    return out


def score_event(features):
    """Combine the features into substrate_distance in [0, 1]. 1 = nylon-like."""
    if features['self_condensation']:
        poly = 1.0                       # iterative: one enzyme extends the chain
    elif features['aa_bb_pair']:
        poly = 0.9                       # the literal nylon-6,6 monomer pairing
    elif features['acceptor_bifunctional']:
        poly = 0.5                       # bifunctional monomer, iteration unproven
    elif features['acceptor_free_amines'] >= 2:
        poly = 0.3                       # diamine acceptor, monoacid donor
    else:
        poly = 0.0                       # terminal amide; cannot polymerise

    reactive = 0.5 * (float(features['product_retains_amine'])
                      + float(features['product_retains_acid']))

    spacer = 0.5 * (_spacer_credit(features['amine_spacer'], HMDA_SPACER)
                    + _spacer_credit(features['acid_spacer'], ADIPATE_SPACER))

    if poly == 0.0:
        # A terminal amide cannot extend a chain, so nothing else about it is
        # relevant. Without this, a small, clean, well-behaved acetamide earns
        # parsimony and backbone credit for being an excellent example of a
        # reaction we do not want.
        return 0.0

    backbone = 1.0 - features['backbone_penalty']

    return round(W_POLYCONDENSATION * poly
                 + W_PRODUCT_REACTIVE * reactive
                 + W_SPACER * spacer
                 + W_BACKBONE * backbone
                 + W_PARSIMONY * features.get('parsimony', 0.0), 4)


def score_reaction(rxn_smiles, max_heavy=60):
    """Best-scoring acylation in a reaction, as (score, features).

    A reaction with no proven acylation scores 0.0 with empty features, so this
    can be mapped over the whole corpus without branching at the call site.
    """
    best, best_features = 0.0, None
    try:
        events = acylation_events(rxn_smiles, max_heavy=max_heavy)
    except Exception:
        events = []
    for event in events:
        features = describe_event(event)
        score = score_event(features)
        if best_features is None or score > best:
            best, best_features = score, features
    return best, best_features


def nylon66_architecture_score(score, features):
    """Retarget the legacy heuristic to an AA/BB nylon-6,6 coupling.

    Preserve the legacy score for comparison. This separate score gives the
    literal diacid/diamine target 1.0 and an AB self-condensation 0.8 of the
    architecture credit. Other components and nonpolymerizing cases are unchanged.
    The weights are chemical judgments, not fitted probabilities of activity.
    """
    if not score or not features:
        return 0.0
    old_poly = (1.0 if features['self_condensation'] else
                0.9 if features['aa_bb_pair'] else
                0.5 if features['acceptor_bifunctional'] else
                0.3 if features['acceptor_free_amines'] >= 2 else 0.0)
    new_poly = (1.0 if features['aa_bb_pair'] else
                0.8 if features['self_condensation'] else old_poly)
    return round(score + W_POLYCONDENSATION * (new_poly - old_poly), 4)
