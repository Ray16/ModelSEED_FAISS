import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolFromInchi, MolToInchi, InchiToInchiKey


def mol_from_smiles(smiles):
    """Parse SMILES, return mol or None."""
    if not smiles or smiles == 'null':
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def mol_from_inchi(inchi):
    """Parse InChI, return mol or None."""
    if not inchi or inchi == 'null':
        return None
    try:
        return MolFromInchi(inchi)
    except Exception:
        return None


def process_compound(args):
    """Process a single compound: resolve conflicts, validate with RDKit.

    Runs in a worker process. Returns (cpd_id, result_dict, stats_dict).
    """
    cpd_id, aliases_list, structures_dict, formula, charge = args

    alias_str = ';'.join(aliases_list)
    conflicts = 0
    corrected = 0
    invalid = 0
    warnings = []

    # Resolve each type by majority vote
    resolved = {}
    for typ, values in structures_dict.items():
        counts = Counter(values)
        if len(counts) > 1:
            conflicts += 1
        resolved[typ] = counts.most_common(1)[0][0]

    # Validate consistency with RDKit
    smiles_str = resolved.get('SMILE')
    inchi_str = resolved.get('InChI')
    inchikey_str = resolved.get('InChIKey')

    # Try parsing SMILES
    smi_mol = mol_from_smiles(smiles_str)
    smi_computed_inchikey = None
    smi_computed_inchi = None
    if smi_mol:
        try:
            smi_computed_inchi = MolToInchi(smi_mol)
            smi_computed_inchikey = InchiToInchiKey(smi_computed_inchi) if smi_computed_inchi else None
        except Exception:
            pass
    elif smiles_str and smiles_str != 'null':
        warnings.append(f"  WARNING: Invalid SMILES for {cpd_id}: {smiles_str[:50]}")
        invalid += 1
        smiles_str = None

    # Try parsing InChI
    inchi_mol = mol_from_inchi(inchi_str)
    inchi_computed_inchikey = None
    if inchi_mol:
        try:
            inchi_computed_inchikey = InchiToInchiKey(inchi_str)
        except Exception:
            pass
    elif inchi_str and inchi_str != 'null':
        warnings.append(f"  WARNING: Invalid InChI for {cpd_id}: {inchi_str[:50]}")
        invalid += 1
        inchi_str = None

    # Cross-validate SMILES vs InChI
    final_smiles = smiles_str if smi_mol else None
    final_inchi = inchi_str if inchi_mol else None
    final_inchikey = None

    if smi_computed_inchikey and inchi_computed_inchikey:
        if smi_computed_inchikey == inchi_computed_inchikey:
            final_inchikey = smi_computed_inchikey
        else:
            # Disagree — prefer InChI, recompute SMILES from InChI
            corrected += 1
            final_inchi = inchi_str
            final_inchikey = inchi_computed_inchikey
            try:
                # Assign stereo from InChI mol before converting to SMILES
                Chem.AssignStereochemistry(inchi_mol, cleanIt=True, force=True)
                candidate = Chem.MolToSmiles(inchi_mol, isomericSmiles=True)
                # Verify round-trip: SMILES -> InChIKey should match InChI-derived key
                check_mol = Chem.MolFromSmiles(candidate)
                if check_mol:
                    check_inchi = MolToInchi(check_mol)
                    check_key = InchiToInchiKey(check_inchi) if check_inchi else None
                    if check_key == inchi_computed_inchikey:
                        final_smiles = candidate
                    else:
                        # Round-trip failed — same connectivity check
                        if (check_key and
                                check_key.split('-')[0] == inchi_computed_inchikey.split('-')[0]):
                            # Stereo-only loss, keep best-effort SMILES
                            final_smiles = candidate
                        else:
                            # Can't produce matching SMILES, drop it
                            final_smiles = None
                else:
                    final_smiles = None
            except Exception:
                final_smiles = None
    elif inchi_computed_inchikey:
        final_inchikey = inchi_computed_inchikey
        if not final_smiles and inchi_mol:
            try:
                final_smiles = Chem.MolToSmiles(inchi_mol)
            except Exception:
                pass
    elif smi_computed_inchikey:
        final_inchikey = smi_computed_inchikey
        if not final_inchi and smi_computed_inchi:
            final_inchi = smi_computed_inchi
    else:
        final_inchikey = inchikey_str

    # Check if stored InChIKey differs from computed
    if final_inchikey and inchikey_str and inchikey_str != 'null' and inchikey_str != final_inchikey:
        corrected += 1

    result = {
        'aliases': alias_str,
        'formula': formula,
        'charge': charge,
        'SMILE': final_smiles,
        'InChI': final_inchi,
        'InChIKey': final_inchikey,
    }
    stats = {
        'conflicts': conflicts,
        'corrected': corrected,
        'invalid': invalid,
        'warnings': warnings,
    }
    return cpd_id, result, stats


def cleanup():
    input_path = 'All_ModelSEED_Structures.txt'
    output_path = 'Unique_ModelSEED_Structures_new.txt'
    num_workers = min(cpu_count(), 64)

    print(f"Loading data from {input_path}...")

    # Group by cpd_id: collect aliases, structures by type, formula/charge
    compounds = defaultdict(lambda: {
        'aliases': set(),
        'structures': defaultdict(list),
        'formula': None,
        'charge': None,
    })

    total_rows = 0
    charged_rows = 0

    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 8:
                continue
            total_rows += 1

            cpd_id, type_str, status, ext_id, source, formula, charge, struct = parts

            if status != 'Charged':
                continue
            charged_rows += 1

            cpd = compounds[cpd_id]
            cpd['aliases'].add(ext_id)
            cpd['structures'][type_str].append(struct)

            if cpd['formula'] is None and type_str in ('SMILE', 'InChI'):
                if formula and formula != 'null':
                    cpd['formula'] = formula
                if charge and charge != 'null':
                    cpd['charge'] = charge

    print(f"  Total rows: {total_rows}")
    print(f"  Charged rows: {charged_rows}")
    print(f"  Unique compounds: {len(compounds)}")

    # Prepare work items for parallel processing (must be picklable — no sets/defaultdicts)
    work_items = []
    for cpd_id in sorted(compounds.keys()):
        cpd = compounds[cpd_id]
        work_items.append((
            cpd_id,
            sorted(cpd['aliases']),
            dict(cpd['structures']),  # convert defaultdict to dict
            cpd['formula'] or 'null',
            cpd['charge'] or 'null',
        ))

    # Process compounds in parallel
    print(f"\nValidating structures with RDKit using {num_workers} workers...")
    conflicts_resolved = 0
    structures_corrected = 0
    invalid_skipped = 0

    results = {}
    with Pool(num_workers) as pool:
        for cpd_id, result, stats in pool.imap_unordered(process_compound, work_items, chunksize=256):
            results[cpd_id] = result
            conflicts_resolved += stats['conflicts']
            structures_corrected += stats['corrected']
            invalid_skipped += stats['invalid']
            for w in stats['warnings']:
                print(w)

    # Write output
    print(f"\nWriting output to {output_path}...")
    row_count = 0
    with open(output_path, 'w') as out:
        out.write('ID\tType\tAliases\tFormula\tCharge\tStructure\n')
        for cpd_id in sorted(results.keys()):
            r = results[cpd_id]
            for typ in ('SMILE', 'InChIKey', 'InChI'):
                struct = r.get(typ)
                if struct and struct != 'null':
                    out.write(f"{cpd_id}\t{typ}\t{r['aliases']}\t{r['formula']}\t{r['charge']}\t{struct}\n")
                    row_count += 1

    # Summary
    print(f"\nSummary:")
    print(f"  Total compounds processed: {len(results)}")
    print(f"  Output rows written: {row_count}")
    print(f"  Cross-source conflicts resolved: {conflicts_resolved}")
    print(f"  Structures corrected (SMILES/InChIKey recomputed): {structures_corrected}")
    print(f"  Invalid structures skipped: {invalid_skipped}")
    print(f"  Workers used: {num_workers}")
    print(f"\nDone. Output written to {output_path}")


if __name__ == '__main__':
    cleanup()
