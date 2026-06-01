"""pH 7 protonation normalization for ModelSEED compounds.

Uses ChemAxon pKa/pKb values from the ModelSEED database to determine
the correct protonation state at pH 7, then adjusts SMILES using RDKit.
"""

import csv
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "ModelSEEDDatabase", "Libs", "Python"))

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from pka_comparison import predict_charge_from_pka
from structure_compare import compare_inchikeys, compute_formula_charge_from_inchi

logger = logging.getLogger("pubchem_validate")

# ── SMARTS patterns for ionizable groups ──────────────────────────────
# Ordered by typical pKa (strongest acid first) for deprotonation,
# and by typical pKb (strongest base first) for protonation.

ACIDIC_SMARTS = [
    (Chem.MolFromSmarts('[SX4](=O)(=O)[OX2H1]'), 'sulfonic_acid'),
    (Chem.MolFromSmarts('[PX4](=O)[OX2H1]'), 'phosphoric_acid'),
    (Chem.MolFromSmarts('[CX3](=O)[OX2H1]'), 'carboxylic_acid'),
    (Chem.MolFromSmarts('[SX3](=O)[OX2H1]'), 'sulfinic_acid'),
    (Chem.MolFromSmarts('[#16X2H1]'), 'thiol'),
    (Chem.MolFromSmarts('[OX2H1][#15]'), 'phosphate_oh_generic'),
]

BASIC_SMARTS = [
    (Chem.MolFromSmarts('[NX3;H2;!$(NC=O);!$(N=*);!$(n)]'), 'primary_amine'),
    (Chem.MolFromSmarts('[NX3;H1;!$(NC=O);!$(N=*);!$(n)]'), 'secondary_amine'),
    (Chem.MolFromSmarts('[NX3;H0;!$(NC=O);!$(N=*);!$(n)]'), 'tertiary_amine'),
]

_uncharger = rdMolStandardize.Uncharger(canonicalOrder=False)

# SMARTS for strongly acidic groups (pKa << 7).  Used to detect
# molecules that should have ionizable groups but lack pKa data.
_STRONG_ACID_SMARTS = [
    Chem.MolFromSmarts('[PX4](=O)'),     # phosphate / phosphonate
    Chem.MolFromSmarts('[SX4](=O)(=O)'), # sulfonate / sulfate
]


def _should_skip_correction(mol, pka_info, stored_charge, db_charge):
    """Cross-validate db_charge against pKa prediction.

    Returns (True, reason) if the correction should be skipped, else
    (False, None).

    Rule 1 -- pKa cross-validation:  When pKa data exists, compute the
    ionic charge prediction.  If stored_charge is closer to the pKa
    prediction than db_charge, stored is more likely correct.

    Rule 2 -- missing ionization data:  When a compound has NO pKa/pKb
    data but contains strongly acidic groups (phosphate, sulfonate),
    ChemAxon likely didn't compute ionization.  db_charge may be a
    default value and is unreliable.
    """
    pka_vals = pka_info['pka']
    pkb_vals = pka_info['pkb']
    has_pka = bool(pka_vals or pkb_vals)

    if has_pka:
        _, _, pka_ionic = predict_charge_from_pka(pka_vals, pkb_vals)
        dist_stored = abs(stored_charge - pka_ionic)
        dist_db = abs(db_charge - pka_ionic)
        if dist_stored < dist_db:
            return True, (f"pKa cross-validation: pKa_ionic={pka_ionic} "
                          f"closer to stored={stored_charge} than "
                          f"db_charge={db_charge}")
    else:
        for pattern in _STRONG_ACID_SMARTS:
            if mol.HasSubstructMatch(pattern):
                return True, ("no pKa data but molecule has strongly "
                              "acidic groups (phosphate/sulfonate)")

    return False, None


def _get_permanent_charge(mol):
    """Get the permanent (non-ionizable) charge of a molecule.

    Neutralizes the molecule with RDKit's Uncharger to remove ionic
    charges (protonation/deprotonation), leaving only permanent charges
    like quaternary nitrogen (+1) or metal ions.
    """
    try:
        neutral = _uncharger.uncharge(mol)
        return Chem.GetFormalCharge(neutral)
    except Exception:
        return 0


def compute_target_charge(pka_info):
    """Get the ChemAxon-predicted charge at pH 7 from the database.

    The db_charge field in pka_info comes from the compound_*.tsv charge
    column, which ChemAxon computed along with the pKa values.  Using
    this directly is more reliable than recomputing from pKa/pKb because
    it correctly handles permanent charges, borderline pKa values, and
    special cases.

    Returns target charge (int) or None if unavailable.
    """
    db_charge = pka_info.get('db_charge')
    if db_charge is not None:
        return db_charge
    return None


def adjust_smiles_to_target_charge(smiles, target_charge):
    """Adjust protonation of a SMILES to reach the target formal charge.

    Uses RDKit Uncharger to neutralize, then applies SMARTS-based
    deprotonation/protonation to reach the target charge.

    Returns dict {smiles, inchi, inchikey} or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    current_charge = Chem.GetFormalCharge(mol)
    if current_charge == target_charge:
        # Already correct — just compute InChI/InChIKey
        return _mol_to_result(mol)

    # Neutralize first, then re-ionize to target
    try:
        mol = _uncharger.uncharge(mol)
    except Exception:
        return None

    permanent = Chem.GetFormalCharge(mol)
    charge_needed = target_charge - permanent

    if charge_needed < 0:
        # Need to deprotonate (remove H, add negative charge)
        mol = _deprotonate(mol, abs(charge_needed))
    elif charge_needed > 0:
        # Need to protonate (add H, add positive charge)
        mol = _protonate(mol, charge_needed)

    if mol is None:
        return None

    final_charge = Chem.GetFormalCharge(mol)
    if final_charge != target_charge:
        return None

    return _mol_to_result(mol)


def _deprotonate(mol, count):
    """Remove `count` protons from the most acidic groups."""
    emol = Chem.RWMol(mol)
    removed = 0
    for pattern, _name in ACIDIC_SMARTS:
        if removed >= count:
            break
        matches = emol.GetSubstructMatches(pattern)
        for match in matches:
            if removed >= count:
                break
            # The last atom in the match is the -OH hydrogen carrier
            oh_idx = match[-1]
            atom = emol.GetAtomWithIdx(oh_idx)
            total_hs = atom.GetTotalNumHs()
            if total_hs > 0:
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(total_hs - 1)
                atom.SetFormalCharge(atom.GetFormalCharge() - 1)
                removed += 1
    if removed < count:
        return None
    try:
        Chem.SanitizeMol(emol)
        return emol.GetMol()
    except Exception:
        return None


def _protonate(mol, count):
    """Add `count` protons to the most basic groups."""
    emol = Chem.RWMol(mol)
    added = 0
    for pattern, _name in BASIC_SMARTS:
        if added >= count:
            break
        matches = emol.GetSubstructMatches(pattern)
        for match in matches:
            if added >= count:
                break
            n_idx = match[0]
            atom = emol.GetAtomWithIdx(n_idx)
            if atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                added += 1
    if added < count:
        return None
    try:
        Chem.SanitizeMol(emol)
        return emol.GetMol()
    except Exception:
        return None


def _mol_to_result(mol):
    """Convert an RDKit mol to {smiles, inchi, inchikey} dict."""
    smiles = Chem.MolToSmiles(mol)
    inchi = Chem.MolToInchi(mol)
    if not inchi:
        return None
    inchikey = Chem.InchiToInchiKey(inchi)
    if not inchikey:
        return None
    return {'smiles': smiles, 'inchi': inchi, 'inchikey': inchikey}


def normalize_smiles_to_pka_charge(smiles, pka_info):
    """Normalize a SMILES to the charge predicted by ChemAxon.

    Uses the db_charge from pka_info (ChemAxon's predicted charge at pH 7).

    Returns dict {smiles, inchi, inchikey} or None if adjustment fails
    or no pKa data is available.
    """
    target = compute_target_charge(pka_info)
    if target is None:
        return None

    return adjust_smiles_to_target_charge(smiles, target)


def run_phase4_pka_validation(candidates, structures, pka_data,
                              names=None, corrections_file=None):
    """Phase 4: Validate protonation using ChemAxon pKa values.

    For each compound with stored SMILES and pKa data:
    1. Compute target charge from ChemAxon pKa/pKb
    2. Compare to stored SMILES charge
    3. If charges match -> skip (stored protonation is correct)
    4. If charges differ -> adjust SMILES to target charge

    Compounds without pKa data (~40%) are skipped.

    Returns dict of corrections compatible with apply_corrections():
        {cpd_id: {smiles, inchi, inchikey, result_type, strategy, query}}
    """
    if corrections_file is None:
        corrections_file = os.path.join(SCRIPT_DIR,
                                        "pubchem_protonation_corrections.tsv")

    logger.info("Phase 4: ChemAxon pKa protonation validation")

    # Collect compounds with SMILES
    work_items = []
    for cpd_id in candidates:
        stored = structures.get(cpd_id, {})
        smiles = stored.get("smiles", "")
        if smiles:
            work_items.append((cpd_id, smiles))

    logger.info("  Compounds with SMILES: %d", len(work_items))
    if not work_items:
        logger.info("  No compounds to validate.")
        return {}

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    corrections = {}
    stats = {
        'already_correct': 0,
        'pka_corrected': 0,
        'no_pka_data': 0,
        'skipped_crossval': 0,
        'adjustment_failed': 0,
        'parse_failed': 0,
    }
    correction_rows = []
    borderline_warnings = []

    items_iter = work_items
    if _tqdm:
        items_iter = _tqdm(work_items, desc="Phase 4: pKa validation")

    for cpd_id, stored_smiles in items_iter:
        # Look up ChemAxon pKa data
        pka_info = pka_data.get(cpd_id)
        if pka_info is None:
            stats['no_pka_data'] += 1
            continue

        pka_vals = pka_info['pka']
        pkb_vals = pka_info['pkb']

        # Parse stored SMILES
        mol = Chem.MolFromSmiles(stored_smiles)
        if mol is None:
            stats['parse_failed'] += 1
            continue

        stored_charge = Chem.GetFormalCharge(mol)
        target_charge = compute_target_charge(pka_info)
        if target_charge is None:
            stats['parse_failed'] += 1
            continue

        # Check for borderline pKa values
        borderline = [(a, v) for _, a, v in pka_vals if 6.0 <= v <= 8.0]
        if borderline:
            borderline_warnings.append((cpd_id, borderline))

        if stored_charge == target_charge:
            stats['already_correct'] += 1
            continue

        # Cross-validate db_charge against pKa prediction
        skip, skip_reason = _should_skip_correction(
            mol, pka_info, stored_charge, target_charge)
        if skip:
            stats['skipped_crossval'] += 1
            logger.debug("  %s: skipped (%s)", cpd_id, skip_reason)
            continue

        # Charges differ — adjust SMILES to target
        result = adjust_smiles_to_target_charge(stored_smiles, target_charge)
        if result is None:
            stats['adjustment_failed'] += 1
            logger.debug("  %s: could not adjust charge %d -> %d",
                         cpd_id, stored_charge, target_charge)
            continue

        # Verify the adjustment changed only protonation
        stored = structures.get(cpd_id, {})
        stored_ik = stored.get("inchikey", "")
        result_ik = result['inchikey']
        ik_result = compare_inchikeys(stored_ik, result_ik)

        if ik_result not in ("PROTONATION_DIFF", "MATCH"):
            stats['adjustment_failed'] += 1
            logger.debug("  %s: pKa adjustment changed connectivity "
                         "(InChIKey %s vs %s, type=%s)",
                         cpd_id, stored_ik, result_ik, ik_result)
            continue

        formula, charge = compute_formula_charge_from_inchi(result['inchi'])
        corrections[cpd_id] = {
            'smiles': result['smiles'],
            'inchi': result['inchi'],
            'inchikey': result['inchikey'],
            'pubchem_cid': 'pKa',
            'result_type': 'PKA_CORRECTION',
            'strategy': 'chemaxon_pka',
            'query': 'ChemAxon_pKa',
            'validation_reason': (
                f"Protonation adjusted to ChemAxon pKa prediction "
                f"(stored_charge={stored_charge}, "
                f"target_charge={target_charge})"),
        }

        cpd_names = names.get(cpd_id, []) if names else []
        name_str = cpd_names[0] if cpd_names else ""
        bl_str = ""
        if borderline:
            bl_str = "; ".join(f"atom {a} pKa={v:.2f}" for a, v in borderline)
        correction_rows.append({
            'cpd_id': cpd_id,
            'compound_name': name_str,
            'action': 'corrected',
            'stored_inchikey': stored_ik,
            'pka_inchikey': result_ik,
            'stored_smiles': stored_smiles,
            'pka_smiles': result['smiles'],
            'stored_formula': stored.get('formula', ''),
            'pka_formula': formula or '',
            'stored_charge': str(stored_charge),
            'pka_charge': str(target_charge),
            'borderline_pka': bl_str,
        })
        stats['pka_corrected'] += 1

    # Write corrections TSV
    if correction_rows:
        fieldnames = ['cpd_id', 'compound_name', 'action',
                      'stored_inchikey', 'pka_inchikey',
                      'stored_smiles', 'pka_smiles',
                      'stored_formula', 'pka_formula',
                      'stored_charge', 'pka_charge',
                      'borderline_pka']
        with open(corrections_file, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in sorted(correction_rows, key=lambda r: r['cpd_id']):
                writer.writerow(row)
        logger.info("  Protonation corrections log: %s", corrections_file)

    logger.info("  Already correct (ChemAxon agrees): %d",
                stats['already_correct'])
    logger.info("  Protonation corrected (ChemAxon pKa): %d",
                stats['pka_corrected'])
    logger.info("  No pKa data (skipped): %d", stats['no_pka_data'])
    logger.info("  Skipped (cross-validation): %d",
                stats['skipped_crossval'])
    logger.info("  Adjustment failed: %d", stats['adjustment_failed'])
    logger.info("  Parse failed: %d", stats['parse_failed'])

    if borderline_warnings:
        logger.info("  Compounds with borderline pKa (6.0-8.0): %d",
                    len(borderline_warnings))

    return corrections
