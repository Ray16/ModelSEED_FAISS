"""Deterministic soft chemical selection after DRFP retrieval.

Preferences measure substrate resemblance, never enzyme competence. The policy
and every component are exported; no original reaction IDs affect the score.
"""
from itertools import combinations, product
import json

from rdkit import Chem

from amide_synthesis.infrastructure.paths import PROJECT
from .acylation import neutral_key
from .substrate_distance import _backbone_penalty

POLICY_PATH = PROJECT / 'config/chemical_selection_policy.json'


def load_policy():
    policy = json.loads(POLICY_PATH.read_text())
    if abs(sum(policy['weights'].values()) - 1) > 1e-10:
        raise ValueError('Chemical component weights must sum to one')
    if any(w < 0 for w in policy['weights'].values()):
        raise ValueError('Chemical component weights must be nonnegative')
    return policy


def widest_path(mol, pairs):
    paths = [Chem.GetShortestPath(mol, a, b) for a, b in pairs if a != b]
    if not paths:
        return None, ()
    path = max(paths, key=lambda p: sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in p[1:-1]))
    return sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in path[1:-1]), path


def score_features(row, policy=None):
    from nylon_chemical_review import primary_amines, acid_carbons
    policy = policy or load_policy()
    donor, acceptor = [Chem.MolFromSmiles(row[key]) for key in ['reacting_acid_smiles', 'reacting_acceptor_smiles']]
    target = policy['target']
    acids, amines = acid_carbons(donor), primary_amines(acceptor)
    ab = neutral_key(donor) == neutral_key(acceptor)
    acid_pairs = list(combinations(acids, 2))
    amine_pairs = list(combinations(amines, 2))
    if ab:
        cross = list(product(primary_amines(donor), acids))
        acid_pairs = acid_pairs or cross
        amine_pairs = amine_pairs or list(product(amines, acid_carbons(acceptor)))
    acid_span, acid_path = widest_path(donor, acid_pairs)
    amine_span, amine_path = widest_path(acceptor, amine_pairs)
    category = ('diacid_primary_diamine' if len(acids) >= 2 and len(amines) >= 2 else
                'ab_self_condensation' if ab else
                'aminoacid_acceptor' if acid_carbons(acceptor) else
                'primary_diamine_acceptor' if len(amines) >= 2 else 'other')
    credits = lambda value, reference: (0.0 if value is None else
        policy['spacer_decay'] / (policy['spacer_decay'] + abs(value-reference)))
    penalties = [_backbone_penalty(mol, path) for mol, path in [(donor,acid_path),(acceptor,amine_path)] if path]
    if not penalties:
        # Monofunctional substrates have no two-end spacer. Describe their whole
        # carbon framework rather than equating missing anchors with inactivity.
        atoms = [a for mol in [donor, acceptor] for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        penalties = [sum(a.GetIsAromatic() or a.IsInRing() for a in atoms)/max(1,len(atoms))]
    size_credit = lambda mol, reference: policy['size_slack']/(policy['size_slack']+max(0,mol.GetNumHeavyAtoms()-reference))
    components = {
        'architecture': policy['architecture_credits'][category],
        'retained_ends': .5*(bool(row['product_retains_primary_amine'])+bool(row['product_retains_carboxyl'])),
        'spacers': .5*(credits(acid_span,target['acid_spacer_carbons'])+credits(amine_span,target['amine_spacer_carbons'])),
        'backbone': 1-sum(penalties)/len(penalties),
        'size': .5*(size_credit(donor,target['donor_heavy_atoms'])+size_credit(acceptor,target['acceptor_heavy_atoms'])),
    }
    result = {f'chem_{key}_credit': value for key,value in components.items()}
    result.update({f'chem_{key}_contribution': value*policy['weights'][key] for key,value in components.items()})
    result.update(chemical_resemblance=sum(value*policy['weights'][key] for key,value in components.items()),
                  chem_architecture=category, chem_acid_spacer_carbons=acid_span,
                  chem_amine_spacer_carbons=amine_span,
                  chem_donor_heavy_atoms=donor.GetNumHeavyAtoms(), chem_acceptor_heavy_atoms=acceptor.GetNumHeavyAtoms(),
                  chem_aromatic_atoms=sum(a.GetIsAromatic() for mol in [donor,acceptor] for a in mol.GetAtoms()),
                  chem_ring_atoms=sum(a.IsInRing() for mol in [donor,acceptor] for a in mol.GetAtoms()))
    return result
