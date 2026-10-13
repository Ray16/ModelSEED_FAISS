"""Score how close a proven N-acylation is to the nylon-6,6 chemistry itself.

WHY THIS EXISTS. The retrieval score is a cosine between whole-reaction
difference fingerprints, and that quantity is dominated by the largest
compositional change in the reaction - the cofactor. It is why the acyl-CoA
query's nearest EC-labelled neighbours are thioester *hydrolases*, which form no
amide at all. Cosine finds the neighbourhood; it cannot rank within it.

This module ranks within it, using no embedding. It reads the two partners the
acylation screen has already proven - the amine acceptor and the acyl group
grafted onto it - and asks how close that specific pair is to adipate + HMDA.

Four questions, in decreasing order of how much they matter for a polyamide:

1. CAN THE CHEMISTRY POLYMERISE AT ALL? Three amide synthetases that the
   boolean screen cannot tell apart:

     terminal      acid A + amine B -> A-B, and A-B is inert.  (glutamine
                   synthetase, PuuA)  Dead end: one bond, then stop.
     AA/BB pair    a DIACID acylates a DIAMINE. Both partners are
                   bifunctional, so the product still carries a free acid at
                   one end and a free amine at the other. This is nylon-6,6.
     self-condensing  the acyl donor and the amine acceptor are THE SAME
                   species: one bifunctional monomer, acid at one end and
                   amine at the other, joined head-to-tail. The product is
                   again such a monomer, so the same enzyme runs again:
                   monomer -> dimer -> trimer. This is iterative
                   polycondensation (BibC, PubC, DesD), and it is the AB
                   analogue of nylon's AA/BB.

   Only the last two can extend a chain. That distinction is the single most
   informative bit in the whole shortlist and the boolean screen throws it away.

2. DOES THE PRODUCT STAY REACTIVE? A bifunctional monomer that loses its free
   acid on coupling cannot iterate. Checked directly on the product.

3. ARE THE SPACERS THE RIGHT LENGTH? Adipate puts 4 carbons between its two
   carboxyl carbons; HMDA puts 6 between its two nitrogens. The known gap to
   the NIS synthetases is precisely that their native monomers are two carbons
   shorter per arm, so this is the term that quantifies it rather than
   asserting it in prose.

4. IS THE SUBSTRATE MOSTLY THE MOTIF, OR MOSTLY SCAFFOLD? A high score on
   questions 1-3 describes the BOND, not the molecule carrying it, and an
   active site has to hold the whole molecule. Folylpolyglutamate synthetase
   scores 0.883 on bond chemistry - it really does iteratively extend a
   polyamide onto a free amine - but its substrate is
   tetrahydrofolyl-polyglutamate, and the pocket is built to grip a pterin
   ring. MurE/MurF likewise form the right amide onto meso-diaminopimelate
   while recognising UDP-MurNAc-tripeptide. Neither will see a C6 diacid.

   Adipate is 10 heavy atoms and HMDA is 8. A substrate carrying a pterin, a
   UDP-sugar, a CoA or a nucleotide is mostly recognition element, and the
   fraction of it that is NOT the nylon-like motif is charged for here. This
   is the term that separates "forms the right bond" from "accepts the right
   molecule".

5. IS THE BACKBONE A PLAIN ALIPHATIC CHAIN? Nylon monomers are unsubstituted
   methylene chains. Citrate (AsbA's native acid) is branched, hydroxylated and
   a triacid; the NIS monomers are N-hydroxylated. Both are real deviations and
   both are charged for here.

The weights below are deliberately flat and legible, in the spirit of
`candidate_table.priority`: they encode a stated chemical opinion, not a fit to
data, and every component is written to its own column so a reader can reweight
them without rerunning anything.
"""
from __future__ import annotations

from rdkit import Chem

from .acylation import FREE_AMINE, acylation_events, count_free_amines, neutral_key

# Nylon-6,6 reference geometry.
ADIPATE_SPACER = 4      # carbons between adipate's two carboxyl carbons
HMDA_SPACER = 6         # carbons between HMDA's two nitrogens

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
ADIPATE_HEAVY = 10
HMDA_HEAVY = 8
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
