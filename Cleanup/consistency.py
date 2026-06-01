"""Phase 0: Self-consistency validation for ModelSEED structures.

Checks that stored SMILES, InChI, InChIKey, formula, and charge are
internally consistent.  Computes missing InChI/InChIKey from SMILES
for non-generic compounds.  Runs before PubChem validation so that
downstream phases work with clean data.
"""

import csv
import logging
import os

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey, MolFromInchi

from structure_compare import compute_formula_charge_from_inchi

logger = logging.getLogger("pubchem_validate")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phase0_consistency(structures, report_file=None):
    """Validate and repair internal consistency of all structures.

    Checks:
    1. SMILES parses correctly via RDKit
    2. InChI/InChIKey can be computed from SMILES (fills missing values)
    3. Stored InChIKey matches InChI
    4. SMILES and InChI encode the same connectivity
    5. Formula/charge matches what InChI implies

    Args:
        structures: dict from load_structures() — modified in-place
        report_file: path for the consistency report TSV

    Returns:
        dict with counts of issues found and fixed.
    """
    if report_file is None:
        report_file = os.path.join(SCRIPT_DIR,
                                   "consistency_report.tsv")

    logger.info("Phase 0: Self-consistency validation")

    stats = {
        'total': 0,
        'smiles_parse_fail': 0,
        'inchi_computed': 0,
        'inchikey_computed': 0,
        'inchikey_mismatch': 0,
        'connectivity_mismatch': 0,
        'formula_fixed': 0,
        'charge_fixed': 0,
        'skipped_rgroup': 0,
    }
    report_rows = []

    for cpd_id in sorted(structures.keys()):
        s = structures[cpd_id]
        stats['total'] += 1
        smiles = s.get('smiles', '')
        inchi = s.get('inchi', '')
        inchikey = s.get('inchikey', '')
        formula = s.get('formula', '')
        charge = s.get('charge', '')

        if not smiles:
            continue

        # Skip R-group/generic compounds — can't compute InChI
        if '*' in smiles:
            stats['skipped_rgroup'] += 1
            continue

        # 1. Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats['smiles_parse_fail'] += 1
            report_rows.append({
                'cpd_id': cpd_id,
                'issue': 'smiles_parse_fail',
                'field': 'SMILES',
                'stored': smiles[:80],
                'computed': '',
                'action': 'flagged',
            })
            continue

        # 2. Compute missing InChI/InChIKey from SMILES
        if not inchi:
            try:
                computed_inchi = MolToInchi(mol)
                if computed_inchi:
                    s['inchi'] = computed_inchi
                    inchi = computed_inchi
                    stats['inchi_computed'] += 1
                    report_rows.append({
                        'cpd_id': cpd_id,
                        'issue': 'inchi_computed',
                        'field': 'InChI',
                        'stored': '',
                        'computed': computed_inchi[:80],
                        'action': 'fixed',
                    })
            except Exception:
                pass

        if inchi and not inchikey:
            try:
                computed_ik = InchiToInchiKey(inchi)
                if computed_ik:
                    s['inchikey'] = computed_ik
                    inchikey = computed_ik
                    stats['inchikey_computed'] += 1
                    report_rows.append({
                        'cpd_id': cpd_id,
                        'issue': 'inchikey_computed',
                        'field': 'InChIKey',
                        'stored': '',
                        'computed': computed_ik,
                        'action': 'fixed',
                    })
            except Exception:
                pass

        if not inchi or not inchikey:
            continue

        # 3. Verify InChIKey matches InChI
        try:
            computed_ik = InchiToInchiKey(inchi)
            if computed_ik and computed_ik != inchikey:
                stats['inchikey_mismatch'] += 1
                report_rows.append({
                    'cpd_id': cpd_id,
                    'issue': 'inchikey_mismatch',
                    'field': 'InChIKey',
                    'stored': inchikey,
                    'computed': computed_ik,
                    'action': 'flagged',
                })
        except Exception:
            pass

        # 4. Cross-check SMILES vs InChI connectivity
        try:
            smi_inchi = MolToInchi(mol)
            if smi_inchi:
                smi_ik = InchiToInchiKey(smi_inchi)
                stored_conn = inchikey.split('-')[0]
                smi_conn = smi_ik.split('-')[0] if smi_ik else ''
                if smi_conn and stored_conn != smi_conn:
                    stats['connectivity_mismatch'] += 1
                    report_rows.append({
                        'cpd_id': cpd_id,
                        'issue': 'smiles_inchi_connectivity_mismatch',
                        'field': 'SMILES/InChI',
                        'stored': f'InChI_conn={stored_conn}',
                        'computed': f'SMILES_conn={smi_conn}',
                        'action': 'flagged',
                    })
        except Exception:
            pass

        # 5. Formula/charge from InChI
        comp_formula, comp_charge = compute_formula_charge_from_inchi(inchi)
        if comp_formula:
            # Normalize element ordering for comparison
            if comp_formula != formula:
                # Check if it's just element ordering (cosmetic)
                from collections import Counter
                import re as _re

                def _parse_formula(f):
                    return Counter(
                        {m[0]: int(m[1]) if m[1] else 1
                         for m in _re.findall(r'([A-Z][a-z]?)(\d*)', f)
                         if m[0]})

                stored_elems = _parse_formula(formula)
                comp_elems = _parse_formula(comp_formula)

                if stored_elems != comp_elems:
                    # Real formula difference
                    old_formula = formula
                    s['formula'] = comp_formula
                    stats['formula_fixed'] += 1
                    report_rows.append({
                        'cpd_id': cpd_id,
                        'issue': 'formula_mismatch',
                        'field': 'Formula',
                        'stored': old_formula,
                        'computed': comp_formula,
                        'action': 'fixed',
                    })

            comp_charge_str = str(comp_charge) if comp_charge is not None else ''
            if comp_charge_str and comp_charge_str != str(charge):
                old_charge = charge
                s['charge'] = comp_charge_str
                stats['charge_fixed'] += 1
                report_rows.append({
                    'cpd_id': cpd_id,
                    'issue': 'charge_mismatch',
                    'field': 'Charge',
                    'stored': str(old_charge),
                    'computed': comp_charge_str,
                    'action': 'fixed',
                })

    # Write report TSV
    if report_rows:
        fieldnames = ['cpd_id', 'issue', 'field', 'stored', 'computed',
                      'action']
        with open(report_file, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in sorted(report_rows, key=lambda r: r['cpd_id']):
                writer.writerow(row)
        logger.info("  Consistency report: %s", report_file)

    logger.info("  Total compounds: %d", stats['total'])
    logger.info("  Skipped (R-groups): %d", stats['skipped_rgroup'])
    logger.info("  SMILES parse failures: %d", stats['smiles_parse_fail'])
    logger.info("  InChI computed from SMILES: %d", stats['inchi_computed'])
    logger.info("  InChIKey computed from InChI: %d",
                stats['inchikey_computed'])
    logger.info("  InChIKey mismatches (flagged): %d",
                stats['inchikey_mismatch'])
    logger.info("  SMILES/InChI connectivity mismatches (flagged): %d",
                stats['connectivity_mismatch'])
    logger.info("  Formula fixed: %d", stats['formula_fixed'])
    logger.info("  Charge fixed: %d", stats['charge_fixed'])

    return stats
