import csv
import logging
import os
import time
from multiprocessing import Pool, cpu_count

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from structure_compare import (compare_inchikeys, count_defined_stereo,
                                _classify_mismatch, classify_mismatch_worker)
from pubchem_api import query_cid_properties

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "ModelSEEDDatabase", "Libs", "Python"))

logger = logging.getLogger("pubchem_validate")


def generate_report(conn, candidates, structures, names,
                    report_file, mismatch_file, protonation_file, stereo_file):
    """Generate TSV reports from cache with enriched columns."""
    logger.info("Generating report...")
    cur = conn.execute("SELECT * FROM cache WHERE cpd_id IN ({})".format(
        ",".join("?" * len(candidates))), candidates)
    rows = cur.fetchall()

    report_header = [
        "cpd_id", "compound_name", "name_queried", "strategy", "result_type",
        "stored_inchikey", "pubchem_inchikey", "pubchem_cid",
        "stored_smiles", "pubchem_smiles",
    ]
    mismatch_header = report_header + ["mismatch_detail"]
    stereo_header = report_header + [
        "stored_stereo_defined", "stored_stereo_potential",
        "pubchem_stereo_defined", "pubchem_stereo_potential",
    ]

    stats = defaultdict(int)
    mismatch_substats = defaultdict(int)
    report_rows = []
    mismatch_work = []  # items to classify in parallel
    protonation_rows = []
    stereo_rows = []

    for row in rows:
        cpd_id = row[0]
        strategy = row[1]
        query = row[2]
        status = row[3]
        pub_cid = row[4] or ""
        pub_inchikey = row[5] or ""
        pub_smiles = row[6] or ""

        stored_inchikey = structures.get(cpd_id, {}).get("inchikey", "")
        stored_smiles = structures.get(cpd_id, {}).get("smiles", "")
        cpd_names = names.get(cpd_id, []) if names else []
        compound_name = cpd_names[0] if cpd_names else ""

        if status == "not_found":
            result_type = "NOT_FOUND"
        elif status == "ambiguous":
            result_type = "AMBIGUOUS"
        elif status == "xref_conflict":
            result_type = "XREF_CONFLICT"
        else:
            result_type = compare_inchikeys(stored_inchikey, pub_inchikey)

        stats[result_type] += 1
        out_row = [
            cpd_id, compound_name, query, strategy, result_type,
            stored_inchikey, pub_inchikey, pub_cid,
            stored_smiles, pub_smiles,
        ]
        report_rows.append(out_row)
        if result_type == "MISMATCH":
            stored_struct = structures.get(cpd_id, {})
            # Derive PubChem formula from SMILES when InChI not available
            pub_formula = ""
            if pub_smiles:
                try:
                    p_mol = Chem.MolFromSmiles(pub_smiles)
                    if p_mol:
                        pub_formula = rdMolDescriptors.CalcMolFormula(p_mol)
                except Exception:
                    pass
            pub_struct = {"smiles": pub_smiles, "inchi": "",
                          "inchikey": pub_inchikey, "formula": pub_formula}
            mismatch_work.append(
                (cpd_id, stored_struct, pub_struct, out_row))
        elif result_type == "PROTONATION_DIFF":
            protonation_rows.append(out_row)
        elif result_type == "STEREO_DIFF":
            s_spec, s_pot = count_defined_stereo(stored_smiles)
            p_spec, p_pot = count_defined_stereo(pub_smiles)
            stereo_rows.append(out_row + [
                s_spec if s_spec is not None else "",
                s_pot if s_pot is not None else "",
                p_spec if p_spec is not None else "",
                p_pot if p_pot is not None else "",
            ])

    # Classify mismatches in parallel
    mismatch_rows = []
    if mismatch_work:
        num_workers = min(cpu_count(), 32)
        logger.info("Classifying %d mismatches with %d workers...",
                    len(mismatch_work), num_workers)
        with Pool(num_workers) as pool:
            for cpd_id, out_row, detail in pool.imap_unordered(
                    classify_mismatch_worker, mismatch_work,
                    chunksize=32):
                subcat = (detail.split("(")[0].strip()
                          if "(" in detail else detail)
                mismatch_substats[subcat] += 1
                mismatch_rows.append(out_row + [detail])

    with open(report_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(report_header)
        writer.writerows(report_rows)

    with open(mismatch_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(mismatch_header)
        writer.writerows(mismatch_rows)

    if protonation_rows:
        with open(protonation_file, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(report_header)
            writer.writerows(protonation_rows)

    if stereo_rows:
        with open(stereo_file, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(stereo_header)
            writer.writerows(stereo_rows)

    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    total = sum(stats.values())
    logger.info("Total compounds checked: %d", total)
    for key in ["MATCH", "STEREO_DIFF", "PROTONATION_DIFF", "MISMATCH",
                "NOT_FOUND", "NO_KEY", "AMBIGUOUS", "XREF_CONFLICT"]:
        if stats[key]:
            pct = 100.0 * stats[key] / total if total else 0
            logger.info("  %-20s: %6d (%5.1f%%)", key, stats[key], pct)
    if mismatch_substats:
        logger.info("  MISMATCH sub-categories:")
        for subcat, cnt in sorted(mismatch_substats.items(),
                                  key=lambda x: -x[1]):
            logger.info("    %-40s: %4d", subcat, cnt)
    logger.info("Full report:       %s", report_file)
    logger.info("Mismatches only:   %s", mismatch_file)
    if protonation_rows:
        logger.info("Protonation diffs: %s", protonation_file)
    if stereo_rows:
        logger.info("Stereo diffs:      %s", stereo_file)


def generate_comparison_images(conn, candidates, structures, images_dir,
                               max_per_type=20, accepted_cpds=None):
    """Generate side-by-side structure images, prioritising accepted corrections.

    Accepted corrections (where PubChem genuinely fixes the ModelSEED
    structure) are generated first.  Remaining slots filled with rejected
    examples.  Each image is labelled CORRECTED or REJECTED.
    """
    from rdkit.Chem import Draw
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(images_dir, exist_ok=True)
    if accepted_cpds is None:
        accepted_cpds = set()

    # Clear old images
    for f in os.listdir(images_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(images_dir, f))

    # Gather compounds by result type from cache
    cur = conn.execute(
        "SELECT cpd_id, strategy, query, status, pubchem_cid, pubchem_inchikey "
        "FROM cache WHERE cpd_id IN ({})".format(
            ",".join("?" * len(candidates))),
        candidates,
    )
    by_type = {"MISMATCH": [], "STEREO_DIFF": [], "PROTONATION_DIFF": []}
    cid_map = {}  # cpd_id -> (cid, strategy, query)
    for row in cur.fetchall():
        cpd_id, strategy, query, status, pub_cid, pub_inchikey = row
        if status != "found" or not pub_cid:
            continue
        stored_ik = structures.get(cpd_id, {}).get("inchikey", "")
        result_type = compare_inchikeys(stored_ik, pub_inchikey)
        if result_type in by_type:
            by_type[result_type].append(cpd_id)
            cid_map[cpd_id] = (pub_cid, strategy, query)

    # Try to get PubChem SMILES from corrections cache first
    all_cpds = []
    for cpds in by_type.values():
        all_cpds.extend(cpds)
    pubchem_smiles = {}
    if all_cpds:
        cur2 = conn.execute(
            "SELECT cpd_id, pubchem_smiles FROM corrections "
            "WHERE cpd_id IN ({})".format(",".join("?" * len(all_cpds))),
            all_cpds,
        )
        for row in cur2.fetchall():
            pubchem_smiles[row[0]] = row[1]

    # For compounds not in corrections cache, fetch SMILES via CID
    missing = [c for c in all_cpds if c not in pubchem_smiles and c in cid_map]
    if missing:
        unique_cids = {}
        for cpd_id in missing:
            cid = cid_map[cpd_id][0]
            unique_cids.setdefault(cid, []).append(cpd_id)
        logger.info("  Fetching SMILES for %d CIDs for images...",
                    len(unique_cids))
        for cid, cpd_ids in unique_cids.items():
            props = query_cid_properties(cid)
            if props and props.get("smiles"):
                for cpd_id in cpd_ids:
                    pubchem_smiles[cpd_id] = props["smiles"]

    # Try to load a monospace font; fall back to default
    try:
        font_large = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_small = font_large

    def _render_image(cpd_id, result_type, tag, filename_prefix):
        stored = structures.get(cpd_id, {})
        stored_smi = stored.get("smiles", "")
        pub_smi = pubchem_smiles.get(cpd_id, "")
        mol_ms = Chem.MolFromSmiles(stored_smi) if stored_smi else None
        mol_pc = Chem.MolFromSmiles(pub_smi) if pub_smi else None
        if mol_ms is None and mol_pc is None:
            return False

        mol_size = (350, 300)
        img_ms = (Draw.MolToImage(mol_ms, size=mol_size) if mol_ms
                  else Image.new("RGB", mol_size, "white"))
        img_pc = (Draw.MolToImage(mol_pc, size=mol_size) if mol_pc
                  else Image.new("RGB", mol_size, "white"))

        margin, header_h, footer_h, gap = 10, 50, 45, 20
        w = mol_size[0] * 2 + gap + margin * 2
        h = mol_size[1] + header_h + footer_h + margin * 2
        canvas = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(canvas)

        # Header: cpd ID, result type, and CORRECTED/REJECTED tag
        tag_color = "green" if tag == "CORRECTED" else "red"
        draw.text((margin, 5),
                  f"{cpd_id}  [{result_type}]", fill="black", font=font_large)
        draw.text((w - margin - len(tag) * 10, 5),
                  tag, fill=tag_color, font=font_large)

        # Column headers
        left_x, right_x = margin, margin + mol_size[0] + gap
        draw.text((left_x + mol_size[0] // 2 - 40, header_h - 18),
                  "ModelSEED", fill="blue", font=font_large)
        draw.text((right_x + mol_size[0] // 2 - 35, header_h - 18),
                  "PubChem", fill="red", font=font_large)
        canvas.paste(img_ms, (left_x, header_h))
        canvas.paste(img_pc, (right_x, header_h))

        div_x = margin + mol_size[0] + gap // 2
        draw.line([(div_x, header_h), (div_x, header_h + mol_size[1])],
                  fill="gray", width=1)

        # Footer
        footer_y = header_h + mol_size[1] + 5
        cid_info = cid_map.get(cpd_id, ("?", "?", "?"))
        formula = stored.get("formula", "")
        draw.text((margin, footer_y),
                  f"Formula: {formula}   CID: {cid_info[0]}   "
                  f"Strategy: {cid_info[1]}   Query: {cid_info[2]}",
                  fill="gray", font=font_small)
        smi_line = (f"MS: {stored_smi[:60]}"
                    f"{'...' if len(stored_smi) > 60 else ''}")
        draw.text((margin, footer_y + 15), smi_line,
                  fill="gray", font=font_small)

        canvas.save(os.path.join(images_dir,
                                 f"{filename_prefix}_{cpd_id}.png"))
        return True

    total_images = 0
    prefix_map = {"MISMATCH": "mismatch", "STEREO_DIFF": "stereo",
                  "PROTONATION_DIFF": "prot"}

    for result_type, cpd_list in by_type.items():
        base = prefix_map[result_type]
        # Prioritise accepted (corrected) compounds
        accepted = sorted(c for c in cpd_list if c in accepted_cpds)
        not_accepted = sorted(c for c in cpd_list if c not in accepted_cpds)

        count = 0
        for cpd_id in accepted:
            if count >= max_per_type:
                break
            if _render_image(cpd_id, result_type, "CORRECTED",
                             f"{base}_accepted"):
                count += 1
                total_images += 1
        # PROTONATION_DIFF is never auto-corrected (by design, not rejected)
        not_accepted_tag = ("NOT_AUTO_CORRECTED" if result_type == "PROTONATION_DIFF"
                            else "REJECTED")
        for cpd_id in not_accepted:
            if count >= max_per_type:
                break
            if _render_image(cpd_id, result_type, not_accepted_tag,
                             f"{base}_rejected"):
                count += 1
                total_images += 1

    logger.info("  Structure images written: %d (in %s/)", total_images,
                images_dir)
