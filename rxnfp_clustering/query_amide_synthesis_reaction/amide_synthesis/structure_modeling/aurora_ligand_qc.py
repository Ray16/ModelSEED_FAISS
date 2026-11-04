"""Validate every modeled ligand against the actual target YAML."""
from collections import Counter
from functools import lru_cache

from rdkit import Chem

CCD_SIGNATURES={'ATP':Counter(C=10,N=5,O=13,P=3),
                'ADP':Counter(C=10,N=5,O=10,P=2),'MG':Counter(Mg=1)}


@lru_cache(maxsize=32)
def smiles_signature(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid ligand SMILES')
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()>1)


def validate_ligands(model,config):
    expected={entry.get('protein',entry.get('ligand'))['id'] for entry in config['sequences']}
    if {chain.name for chain in model}!=expected:
        raise ValueError('Output chain set differs from requested composition')
    for entry in config['sequences']:
        ligand=entry.get('ligand')
        if ligand is None:continue
        if 'smiles' in ligand:signature=smiles_signature(ligand['smiles'])
        else:signature=CCD_SIGNATURES[ligand['ccd']]
        actual=Counter(atom.element.name for residue in model[ligand['id']] for atom in residue
                       if atom.element.name not in ('H','D'))
        if actual!=signature:
            raise ValueError('Ligand composition differs for '+ligand['id']+': '+repr(actual)+' versus '+repr(signature))
