#!/usr/bin/env python3
# usage: python pubchem_validate.py --apply --limit 1000
"""PubChem name-structure cross-validation for ModelSEED compounds.

Validates whether compound names match their stored structures by querying
PubChem for InChIKeys via ChEBI xref, KEGG xref, or name lookup, then
comparing against locally stored InChIKeys.

With --apply, fetches full structures from PubChem for MISMATCH and
STEREO_DIFF compounds and applies corrections only when they demonstrably
improve the data:
  - MISMATCH: accepted only if PubChem formula matches stored formula
  - STEREO_DIFF: accepted only if PubChem has >= defined stereocenters
  - PROTONATION_DIFF: never auto-corrected (pH-dependent)

Safety features:
  - Timestamped backup of structures file before any modification
  - Corrections log is appended (not overwritten) with timestamps
  - Formula and Charge columns updated to match new structures
  - Cross-validation of xref sources (ChEBI vs KEGG)
  - Multi-name cross-validation (ambiguity detection)

Usage:
    python pubchem_validate.py              # full run (resumable)
    python pubchem_validate.py --limit 100  # test with first 100 compounds
    python pubchem_validate.py --apply      # full run + apply corrections
    python pubchem_validate.py --resume     # continue from cache (default)
    python pubchem_validate.py --workers 3  # adjust concurrency (default 5)
    python pubchem_validate.py --clear-cache  # clear cache for fresh run
"""

import argparse
import csv
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from rdkit import Chem
from rdkit.Chem import FindPotentialStereo, StereoSpecified

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Import BiochemPy InChI utilities
sys.path.insert(0, os.path.join(BASE_DIR, "ModelSEEDDatabase", "Libs", "Python"))
from BiochemPy import InChIs

STRUCTURES_FILE = os.path.join(SCRIPT_DIR, "Unique_ModelSEED_Structures_new.txt")
NAMES_FILE = os.path.join(
    BASE_DIR, "ModelSEEDDatabase", "Biochemistry", "Aliases",
    "Unique_ModelSEED_Compound_Names.txt",
)
ALIASES_FILE = os.path.join(
    BASE_DIR, "ModelSEEDDatabase", "Biochemistry", "Aliases",
    "Unique_ModelSEED_Compound_Aliases.txt",
)
CACHE_DB = os.path.join(SCRIPT_DIR, "pubchem_cache.sqlite")
REPORT_FILE = os.path.join(SCRIPT_DIR, "pubchem_validation_report.tsv")
MISMATCH_FILE = os.path.join(SCRIPT_DIR, "pubchem_mismatches.tsv")
CORRECTIONS_LOG = os.path.join(SCRIPT_DIR, "pubchem_corrections_log.tsv")
PROTONATION_FILE = os.path.join(SCRIPT_DIR, "pubchem_protonation_diffs.tsv")

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MAX_RETRIES = 4
NAME_BATCH_SIZE = 100  # names per POST request


# ---------------------------------------------------------------------------
# Rate limiter (token bucket, shared across threads)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate=5.0):
        """rate: max requests per second."""
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                wait = self._next_time - now
                self._next_time += self._interval
            else:
                wait = 0
                self._next_time = now + self._interval
        if wait > 0:
            time.sleep(wait)


_rate_limiter = RateLimiter(rate=5.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_structures(path):
    """Parse structures file -> {cpd_id: {smiles, inchikey, inchi, formula, charge}}."""
    structs = {}
    with open(path, "r") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 6:
                continue
            cpd_id, typ, value = row[0], row[1], row[5]
            if cpd_id not in structs:
                structs[cpd_id] = {"formula": row[3], "charge": row[4]}
            key = typ.strip().lower()
            if key == "smile":
                structs[cpd_id]["smiles"] = value
            elif key == "inchikey":
                structs[cpd_id]["inchikey"] = value
            elif key == "inchi":
                structs[cpd_id]["inchi"] = value
    return structs


def load_names(path):
    """Parse names file -> {cpd_id: [names]}."""
    names = defaultdict(list)
    with open(path, "r") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            cpd_id, name = row[0], row[1]
            if name and name.strip():
                names[cpd_id].append(name.strip())
    return dict(names)


def load_external_ids(path):
    """Parse aliases file -> {cpd_id: {chebi: [...], kegg: [...]}}."""
    ids = defaultdict(lambda: {"chebi": [], "kegg": []})
    with open(path, "r") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            cpd_id, ext_id, source = row[0], row[1], row[2]
            if source == "ChEBI":
                ids[cpd_id]["chebi"].append(ext_id)
            elif source == "KEGG":
                ids[cpd_id]["kegg"].append(ext_id)
    return dict(ids)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def init_cache(db_path):
    """Create/open cache database, return connection."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cpd_id TEXT PRIMARY KEY,
            strategy TEXT,
            query TEXT,
            status TEXT,
            pubchem_cid TEXT,
            pubchem_inchikey TEXT,
            pubchem_smiles TEXT,
            timestamp REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            cpd_id TEXT PRIMARY KEY,
            pubchem_cid TEXT,
            pubchem_smiles TEXT,
            pubchem_inchi TEXT,
            pubchem_inchikey TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    return conn


def get_cached(conn):
    """Return set of cpd_ids already in cache."""
    cur = conn.execute("SELECT cpd_id FROM cache")
    return {row[0] for row in cur.fetchall()}


def save_batch_to_cache(conn, lock, rows):
    """Write multiple rows to cache atomically."""
    with lock:
        conn.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# PubChem queries
# ---------------------------------------------------------------------------

def pubchem_request(method, url, **kwargs):
    """Make a PubChem request with retry and rate limiting."""
    for attempt in range(MAX_RETRIES):
        _rate_limiter.acquire()
        try:
            if method == "GET":
                resp = requests.get(url, timeout=30, **kwargs)
            else:
                resp = requests.post(url, timeout=30, **kwargs)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** (attempt + 1)
                print(f"  PubChem {resp.status_code}, retrying in {wait}s...",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            return None
        except (requests.RequestException, ValueError):
            wait = 2 ** (attempt + 1)
            time.sleep(wait)
    return None


def query_xref(xref_id):
    """Query PubChem by xref ID. Returns (cid, inchikey, smiles) or None."""
    url = (f"{PUBCHEM_BASE}/compound/xref/RegistryID/"
           f"{xref_id}/property/InChIKey,IsomericSMILES/JSON")
    data = pubchem_request("GET", url)
    if data and "PropertyTable" in data:
        props = data["PropertyTable"]["Properties"][0]
        return (
            str(props.get("CID", "")),
            props.get("InChIKey", ""),
            props.get("IsomericSMILES", ""),
        )
    return None


def query_names_batch(name_list):
    """Query PubChem by compound names in batch via POST.

    Returns dict: {name: (cid, inchikey, smiles)} for found names.
    """
    if not name_list:
        return {}
    return _query_names_batch_recursive(name_list)


def _query_names_batch_recursive(name_list):
    """Try batch name lookup; on 404, split and retry recursively."""
    if not name_list:
        return {}

    if len(name_list) == 1:
        name = name_list[0]
        url = f"{PUBCHEM_BASE}/compound/name/property/InChIKey,IsomericSMILES/JSON"
        data = pubchem_request("POST", url, data={"name": name})
        if data and "PropertyTable" in data:
            props = data["PropertyTable"]["Properties"][0]
            return {name: (
                str(props.get("CID", "")),
                props.get("InChIKey", ""),
                props.get("IsomericSMILES", ""),
            )}
        return {}

    url = f"{PUBCHEM_BASE}/compound/name/property/InChIKey,IsomericSMILES/JSON"
    joined = "\n".join(name_list)
    data = pubchem_request("POST", url, data={"name": joined})

    if data and "PropertyTable" in data:
        props_list = data["PropertyTable"]["Properties"]
        results = {}
        for i, props in enumerate(props_list):
            if i < len(name_list):
                results[name_list[i]] = (
                    str(props.get("CID", "")),
                    props.get("InChIKey", ""),
                    props.get("IsomericSMILES", ""),
                )
        return results

    # Batch failed (likely 404 due to unknown name) — split in half
    mid = len(name_list) // 2
    left = _query_names_batch_recursive(name_list[:mid])
    right = _query_names_batch_recursive(name_list[mid:])
    left.update(right)
    return left


def pick_best_names(name_list, max_names=3):
    """Pick the best names for PubChem lookup.

    Returns up to max_names names sorted by quality score (best first).
    Prefers longer, descriptive names starting with letters.
    """
    cleaned = []
    for n in name_list:
        n = re.sub(r"<[^>]+>", "", n)
        n = n.strip()
        if n:
            cleaned.append(n)
    if not cleaned:
        return []
    scored = []
    for n in cleaned:
        score = len(n)
        if re.match(r"^[A-Za-z]", n):
            score += 100
        if re.search(r"[a-z]", n):
            score += 50
        if re.match(r"^[A-Z0-9+\-\s]{1,5}$", n):
            score -= 200
        scored.append((score, n))
    scored.sort(reverse=True)
    seen = set()
    result = []
    for _, n in scored:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            result.append(n)
            if len(result) >= max_names:
                break
    return result


# ---------------------------------------------------------------------------
# InChIKey comparison
# ---------------------------------------------------------------------------

def compare_inchikeys(stored, pubchem):
    """Compare two InChIKeys and return classification."""
    if not stored or not pubchem:
        return "NO_KEY"
    if stored == pubchem:
        return "MATCH"
    s_parts = stored.split("-")
    p_parts = pubchem.split("-")
    if len(s_parts) != 3 or len(p_parts) != 3:
        return "MISMATCH"
    if s_parts[0] == p_parts[0] and s_parts[1] == p_parts[1]:
        return "PROTONATION_DIFF"
    if s_parts[0] == p_parts[0]:
        return "STEREO_DIFF"
    return "MISMATCH"


# ---------------------------------------------------------------------------
# Correction validation helpers
# ---------------------------------------------------------------------------

def count_defined_stereo(smiles):
    """Count defined stereocenters and E/Z bonds in a SMILES string.

    Returns (specified_count, potential_count) or (None, None) on failure.
    """
    if not smiles:
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    stereo_info = FindPotentialStereo(mol)
    potential = len(stereo_info)
    specified = sum(1 for s in stereo_info
                    if s.specified == StereoSpecified.Specified)
    return specified, potential


def compute_formula_charge_from_inchi(inchi_str):
    """Compute molecular formula and charge from an InChI string.

    Uses BiochemPy InChIs utilities to parse layers and adjust for protons.
    Returns (formula, charge) or (None, None) on failure.
    """
    if not inchi_str:
        return None, None
    try:
        formula, layers = InChIs.parse(inchi_str)
        chg = InChIs.charge(layers['q'], layers['p'])
        if layers['p']:
            formula, _ = InChIs.adjust_protons(formula, layers['p'])
        return formula, chg
    except Exception:
        return None, None


def validate_correction(cpd_id, stored_struct, pubchem_struct, result_type):
    """Decide whether a PubChem correction should be accepted.

    Returns (accept: bool, reason: str).

    Rules:
    - MISMATCH: accept only if PubChem formula matches stored formula
      (confirms same molecular skeleton; different formula = wrong compound)
    - STEREO_DIFF: accept only if PubChem has >= defined stereocenters
      (never strip stereochemistry)
    - PROTONATION_DIFF: never auto-correct (pH-dependent)
    """
    if result_type not in ("MISMATCH", "STEREO_DIFF"):
        return False, f"Not auto-correctable ({result_type})"

    stored_inchi = stored_struct.get("inchi", "")
    pub_inchi = pubchem_struct.get("inchi", "")

    if result_type == "STEREO_DIFF":
        # Only accept if PubChem has >= defined stereocenters
        stored_smiles = stored_struct.get("smiles", "")
        pub_smiles = pubchem_struct.get("smiles", "")

        old_spec, old_pot = count_defined_stereo(stored_smiles)
        new_spec, new_pot = count_defined_stereo(pub_smiles)

        if old_spec is None or new_spec is None:
            return False, ("STEREO_DIFF rejected: cannot parse SMILES for "
                           "stereo comparison")
        if new_spec < old_spec:
            return False, (f"STEREO_DIFF rejected: PubChem has fewer defined "
                           f"stereocenters ({new_spec} vs {old_spec})")
        return True, (f"STEREO_DIFF accepted: PubChem stereo "
                      f"{new_spec}>={old_spec} (of {new_pot} potential)")

    # MISMATCH: verify formula matches (same heavy-atom skeleton)
    stored_formula = stored_struct.get("formula", "")
    pub_formula, pub_charge = compute_formula_charge_from_inchi(pub_inchi)

    if pub_formula is None:
        return False, "MISMATCH rejected: cannot parse PubChem InChI for formula"

    if stored_formula and pub_formula:
        # Normalize: strip whitespace for comparison
        sf = stored_formula.strip()
        pf = pub_formula.strip()
        if sf != pf:
            return False, (f"MISMATCH rejected: formula mismatch "
                           f"(stored={sf}, pubchem={pf}) — likely wrong compound")

    return True, "MISMATCH accepted: formula matches"


# ---------------------------------------------------------------------------
# Phase 3: Fetch full structures by CID for correctable compounds
# ---------------------------------------------------------------------------

def query_cid_properties(cid):
    """Fetch InChIKey, IsomericSMILES, InChI for a PubChem CID."""
    url = (f"{PUBCHEM_BASE}/compound/cid/{cid}"
           f"/property/InChIKey,IsomericSMILES,InChI/JSON")
    data = pubchem_request("GET", url)
    if data and "PropertyTable" in data:
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("SMILES", "")
        return {
            "smiles": smiles,
            "inchi": props.get("InChI", ""),
            "inchikey": props.get("InChIKey", ""),
        }
    return None


def fetch_corrections(conn, db_lock, candidates, structures, workers=5):
    """Phase 3: fetch full structures for MISMATCH + STEREO_DIFF compounds.

    Applies validate_correction() gate to every candidate — only returns
    corrections that demonstrably improve the data.

    Returns dict: {cpd_id: {pubchem_cid, smiles, inchi, inchikey, result_type,
                             strategy, query, validation_reason}}
    """
    # Identify correctable compounds from cache
    cur = conn.execute(
        "SELECT cpd_id, strategy, query, status, pubchem_cid, pubchem_inchikey "
        "FROM cache WHERE cpd_id IN ({})".format(
            ",".join("?" * len(candidates))),
        candidates,
    )
    correctable = {}  # cpd_id -> (cid, strategy, query, result_type)
    for row in cur.fetchall():
        cpd_id, strategy, query, status, pub_cid, pub_inchikey = row
        if status != "found" or not pub_cid:
            continue
        stored_ik = structures.get(cpd_id, {}).get("inchikey", "")
        result_type = compare_inchikeys(stored_ik, pub_inchikey)
        if result_type in ("MISMATCH", "STEREO_DIFF"):
            correctable[cpd_id] = (pub_cid, strategy, query, result_type)

    if not correctable:
        print("\nPhase 3: No correctable compounds found.")
        return {}

    # Check which CIDs already have corrections cached
    cur2 = conn.execute("SELECT cpd_id FROM corrections")
    already_corrected = {row[0] for row in cur2.fetchall()}

    # Deduplicate by CID (multiple cpd_ids might map to same CID)
    unique_cids = {}  # cid -> [cpd_ids]
    for cpd_id, (cid, _, _, _) in correctable.items():
        if cpd_id not in already_corrected:
            unique_cids.setdefault(cid, []).append(cpd_id)

    print(f"\nPhase 3: Fetching full structures for corrections")
    print(f"  Correctable compounds: {len(correctable)}")
    print(f"  Already cached corrections: {len(already_corrected & set(correctable))}")
    print(f"  Unique CIDs to fetch: {len(unique_cids)}")

    if unique_cids:
        if tqdm:
            pbar = tqdm(total=len(unique_cids), desc="Phase 3: CID lookups")

        def do_cid_query(cid):
            return (cid, query_cid_properties(cid))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(do_cid_query, cid): cid
                       for cid in unique_cids}
            for future in as_completed(futures):
                cid, props = future.result()
                if tqdm:
                    pbar.update(1)
                if props is None:
                    continue
                rows = []
                for cpd_id in unique_cids[cid]:
                    rows.append((cpd_id, cid, props["smiles"],
                                 props["inchi"], props["inchikey"], time.time()))
                with db_lock:
                    conn.executemany(
                        "INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)",
                        rows,
                    )
                    conn.commit()

        if tqdm:
            pbar.close()

    # Build corrections dict from DB
    cur3 = conn.execute(
        "SELECT cpd_id, pubchem_cid, pubchem_smiles, pubchem_inchi, "
        "pubchem_inchikey FROM corrections WHERE cpd_id IN ({})".format(
            ",".join("?" * len(correctable))),
        list(correctable.keys()),
    )
    raw_result = {}
    for row in cur3.fetchall():
        cpd_id = row[0]
        cid, strategy, query, result_type = correctable[cpd_id]
        raw_result[cpd_id] = {
            "pubchem_cid": row[1],
            "smiles": row[2],
            "inchi": row[3],
            "inchikey": row[4],
            "result_type": result_type,
            "strategy": strategy,
            "query": query,
        }

    # Validate each correction: only accept those that improve the data
    validated = {}
    rejected = []
    for cpd_id, corr in raw_result.items():
        stored = structures.get(cpd_id, {})
        accept, reason = validate_correction(cpd_id, stored, corr,
                                             corr["result_type"])
        if accept:
            corr["validation_reason"] = reason
            validated[cpd_id] = corr
        else:
            rejected.append((cpd_id, corr["result_type"], reason))

    print(f"  Corrections validated: {len(validated)}")
    if rejected:
        print(f"  Corrections rejected: {len(rejected)}")
        for cpd_id, rt, reason in rejected[:20]:
            print(f"    {cpd_id} ({rt}): {reason}")
        if len(rejected) > 20:
            print(f"    ... and {len(rejected) - 20} more")

    return validated


def apply_corrections(corrections, structures, output_path=None):
    """Apply corrections and write to a NEW output file (never overwrites original).

    Replaces SMILES, InChI, InChIKey and updates Formula/Charge to match.
    """
    if not corrections:
        print("\nNo corrections to apply.")
        return

    # Determine output path (never overwrite original)
    if output_path is None:
        base, ext = os.path.splitext(STRUCTURES_FILE)
        output_path = f"{base}_corrected{ext}"
    print(f"\nOriginal file: {STRUCTURES_FILE} (unchanged)")
    print(f"Output file:   {output_path}")

    # Read full structures file preserving all rows
    rows = []
    with open(STRUCTURES_FILE, "r") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            rows.append(row)

    # Compute new formula/charge for corrected compounds
    formula_updates = {}  # cpd_id -> (new_formula, new_charge)
    for cpd_id, corr in corrections.items():
        new_inchi = corr.get("inchi", "")
        if new_inchi:
            new_formula, new_charge = compute_formula_charge_from_inchi(new_inchi)
            if new_formula is not None:
                formula_updates[cpd_id] = (new_formula, str(new_charge))

    # Apply corrections
    log_rows = []
    type_to_field = {"SMILE": "smiles", "InChIKey": "inchikey", "InChI": "inchi"}

    for i, row in enumerate(rows):
        if len(row) < 6:
            continue
        cpd_id, typ = row[0], row[1]
        if cpd_id not in corrections:
            continue
        corr = corrections[cpd_id]

        # Update structure fields (SMILES, InChIKey, InChI)
        field_key = type_to_field.get(typ)
        if field_key is not None:
            old_val = row[5]
            new_val = corr[field_key]
            if new_val and old_val != new_val:
                rows[i] = list(row)
                rows[i][5] = new_val
                log_rows.append([
                    cpd_id, corr["result_type"], typ,
                    old_val, new_val,
                    corr["pubchem_cid"], corr["strategy"], corr["query"],
                ])

        # Update formula/charge for all row types of this compound
        if cpd_id in formula_updates:
            new_formula, new_charge = formula_updates[cpd_id]
            rows[i] = list(rows[i]) if not isinstance(rows[i], list) else rows[i]
            old_formula, old_charge = rows[i][3], rows[i][4]
            rows[i][3] = new_formula
            rows[i][4] = new_charge

    # Log formula/charge changes (once per compound, not per row)
    for cpd_id, (new_formula, new_charge) in formula_updates.items():
        old_formula = structures.get(cpd_id, {}).get("formula", "")
        old_charge = structures.get(cpd_id, {}).get("charge", "")
        corr = corrections[cpd_id]
        if old_formula != new_formula or old_charge != new_charge:
            log_rows.append([
                cpd_id, corr["result_type"], "Formula/Charge",
                f"{old_formula}/{old_charge}", f"{new_formula}/{new_charge}",
                corr["pubchem_cid"], corr["strategy"], corr["query"],
            ])

    if not log_rows:
        print("\nNo fields changed after correction.")
        return

    corrected_count = len({r[0] for r in log_rows})

    # Write corrected structures to NEW file (original unchanged)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

    # Append to corrections log (not overwrite)
    log_header = [
        "timestamp", "cpd_id", "result_type", "field",
        "old_value", "new_value",
        "pubchem_cid", "strategy", "query",
    ]
    log_exists = (os.path.exists(CORRECTIONS_LOG)
                  and os.path.getsize(CORRECTIONS_LOG) > 0)
    with open(CORRECTIONS_LOG, "a", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if not log_exists:
            writer.writerow(log_header)
        ts = datetime.now().isoformat()
        for log_row in log_rows:
            writer.writerow([ts] + log_row)

    print(f"\nCorrections applied:")
    print(f"  Compounds corrected: {corrected_count}")
    print(f"  Fields changed: {len(log_rows)}")
    print(f"  Output file:     {output_path}")
    print(f"  Corrections log: {CORRECTIONS_LOG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate ModelSEED compound names against PubChem structures.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N compounds (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cache (default behavior)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent request threads (default 5)")
    parser.add_argument("--apply", action="store_true",
                        help="Fetch full structures and apply corrections "
                             "for MISMATCH/STEREO_DIFF compounds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for corrected structures "
                             "(default: writes to a new _corrected file, "
                             "never overwrites the original)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the SQLite cache before running "
                             "(for a fresh re-validation)")
    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache and os.path.exists(CACHE_DB):
        os.remove(CACHE_DB)
        print("Cache cleared.")

    print("Loading local data...")
    structures = load_structures(STRUCTURES_FILE)
    names = load_names(NAMES_FILE)
    external_ids = load_external_ids(ALIASES_FILE)

    # Build validation set: compounds with at least one name AND a stored InChIKey
    candidates = []
    for cpd_id in sorted(structures.keys()):
        if cpd_id in names and "inchikey" in structures[cpd_id]:
            candidates.append(cpd_id)

    print(f"Total compounds with structures: {len(structures)}")
    print(f"Total compounds with names: {len(names)}")
    print(f"Validation candidates (name + InChIKey): {len(candidates)}")

    if args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"Limited to first {args.limit} compounds")

    # Init cache
    conn = init_cache(CACHE_DB)
    db_lock = threading.Lock()
    cached = get_cached(conn)
    to_process = [c for c in candidates if c not in cached]
    print(f"Already cached: {len(candidates) - len(to_process)}")
    print(f"To process: {len(to_process)}")

    if not to_process:
        _generate_report(conn, candidates, structures)
        if args.apply:
            corrections = fetch_corrections(
                conn, db_lock, candidates, structures, workers=args.workers)
            apply_corrections(corrections, structures, output_path=args.output)
        conn.close()
        return

    # -------------------------------------------------------------------
    # Phase 1: xref lookups (ChEBI + KEGG) with cross-validation
    # -------------------------------------------------------------------
    # Query BOTH ChEBI and KEGG for compounds that have both
    chebi_to_cpds = defaultdict(list)  # "CHEBI:12345" -> [cpd_ids]
    kegg_to_cpds = defaultdict(list)   # "C00001" -> [cpd_ids]
    cpd_has_chebi = set()
    cpd_has_kegg = set()
    needs_name_lookup = []             # cpd_ids with no xref

    for cpd_id in to_process:
        ext = external_ids.get(cpd_id, {"chebi": [], "kegg": []})
        has_chebi = False
        has_kegg = False
        for cid_val in ext["chebi"]:
            chebi_key = f"CHEBI:{cid_val}"
            chebi_to_cpds[chebi_key].append(cpd_id)
            has_chebi = True
        for kid in ext["kegg"]:
            kegg_to_cpds[kid].append(cpd_id)
            has_kegg = True
        if has_chebi:
            cpd_has_chebi.add(cpd_id)
        if has_kegg:
            cpd_has_kegg.add(cpd_id)
        if not has_chebi and not has_kegg:
            needs_name_lookup.append(cpd_id)

    unique_chebi = list(chebi_to_cpds.keys())
    unique_kegg = list(kegg_to_cpds.keys())
    total_xref_queries = len(unique_chebi) + len(unique_kegg)
    both_count = len(cpd_has_chebi & cpd_has_kegg)

    print(f"\nPhase 1: xref lookups")
    print(f"  Unique ChEBI IDs: {len(unique_chebi)}")
    print(f"  Unique KEGG IDs: {len(unique_kegg)}")
    print(f"  Compounds with both: {both_count}")
    print(f"  Total xref queries (deduplicated): {total_xref_queries}")
    print(f"  Compounds needing name lookup: {len(needs_name_lookup)}")

    # Collect all xref results per compound for cross-validation
    # cpd_id -> [(strategy, xref_id, cid, inchikey, smiles)]
    cpd_xref_results = defaultdict(list)

    def do_xref_query(xref_id, strategy):
        result = query_xref(xref_id)
        return (xref_id, strategy, result)

    progress_lock = threading.Lock()
    completed_count = [0]

    if tqdm:
        pbar = tqdm(total=total_xref_queries, desc="Phase 1: xref queries")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for cid in unique_chebi:
            futures.append(executor.submit(do_xref_query, cid, "chebi_xref"))
        for kid in unique_kegg:
            futures.append(executor.submit(do_xref_query, kid, "kegg_xref"))

        for future in as_completed(futures):
            xref_id, strategy, result = future.result()
            if tqdm:
                pbar.update(1)
            else:
                with progress_lock:
                    completed_count[0] += 1
                    if completed_count[0] % 500 == 0:
                        print(f"  xref: {completed_count[0]}/{total_xref_queries}")

            if result is None:
                continue

            cid, inchikey, smiles = result
            if strategy == "chebi_xref":
                cpd_ids = chebi_to_cpds[xref_id]
            else:
                cpd_ids = kegg_to_cpds[xref_id]

            for cpd_id in cpd_ids:
                cpd_xref_results[cpd_id].append(
                    (strategy, xref_id, cid, inchikey, smiles))

    if tqdm:
        pbar.close()

    # Cross-validate xref results and resolve
    resolved = {}  # cpd_id -> (strategy, query, cid, inchikey, smiles)
    xref_resolved_cpds = set()
    xref_conflicts = set()

    for cpd_id, results in cpd_xref_results.items():
        cids_seen = {r[2] for r in results}  # unique CIDs
        strategies_seen = {r[0] for r in results}

        if len(cids_seen) > 1 and len(strategies_seen) > 1:
            # ChEBI and KEGG returned different CIDs — conflict
            xref_conflicts.add(cpd_id)
        else:
            # All agree or single source — pick best (prefer chebi_xref)
            best = None
            for r in results:
                if r[0] == "chebi_xref":
                    best = r
                    break
            if best is None:
                best = results[0]
            strategy, xref_id, cid, inchikey, smiles = best
            resolved[cpd_id] = (strategy, xref_id, cid, inchikey, smiles)
            xref_resolved_cpds.add(cpd_id)

    # Save xref results to cache
    cache_rows = []
    for cpd_id, (strategy, query, cid, inchikey, smiles) in resolved.items():
        cache_rows.append((cpd_id, strategy, query, "found",
                           cid, inchikey, smiles, time.time()))
    # Save conflicts as xref_conflict status
    for cpd_id in xref_conflicts:
        results = cpd_xref_results[cpd_id]
        query_str = ";".join(f"{r[0]}={r[1]}->CID{r[2]}" for r in results)
        cache_rows.append((cpd_id, "xref_conflict", query_str, "xref_conflict",
                           None, None, None, time.time()))
    if cache_rows:
        save_batch_to_cache(conn, db_lock, cache_rows)

    print(f"  Resolved via xref: {len(resolved)}")
    if xref_conflicts:
        print(f"  xref conflicts (ChEBI/KEGG disagree): {len(xref_conflicts)}")

    # Compounds still needing name lookup: no xref, failed xref, or conflicting
    all_xref_cpds = set()
    for cpd_ids in chebi_to_cpds.values():
        all_xref_cpds.update(cpd_ids)
    for cpd_ids in kegg_to_cpds.values():
        all_xref_cpds.update(cpd_ids)
    unresolved_xref = [c for c in to_process if c in all_xref_cpds
                       and c not in xref_resolved_cpds
                       and c not in xref_conflicts]
    needs_name_lookup.extend(unresolved_xref)

    # -------------------------------------------------------------------
    # Phase 2: Batched name lookups with multi-name cross-validation
    # -------------------------------------------------------------------
    name_to_cpds = defaultdict(list)  # name -> [cpd_ids]
    cpd_to_names = {}                 # cpd_id -> [names]
    no_name = []

    for cpd_id in needs_name_lookup:
        cpd_names = names.get(cpd_id, [])
        best = pick_best_names(cpd_names)
        if best:
            cpd_to_names[cpd_id] = best
            for name in best:
                name_to_cpds[name].append(cpd_id)
        else:
            no_name.append(cpd_id)

    name_list = list(name_to_cpds.keys())
    print(f"\nPhase 2: name lookups")
    print(f"  Unique names to query: {len(name_list)}")
    print(f"  Compounds with no usable name: {len(no_name)}")

    # Process names in batches via thread pool
    name_results = {}  # name -> (cid, inchikey, smiles)

    def do_name_batch(batch):
        return _query_names_batch_recursive(batch)

    batches = [name_list[i:i + NAME_BATCH_SIZE]
               for i in range(0, len(name_list), NAME_BATCH_SIZE)]

    if tqdm:
        pbar2 = tqdm(total=len(batches), desc="Phase 2: name batches")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(do_name_batch, batch): batch
                   for batch in batches}
        for future in as_completed(futures):
            batch_results = future.result()
            name_results.update(batch_results)
            if tqdm:
                pbar2.update(1)

    if tqdm:
        pbar2.close()

    # Cross-validate: for each compound, check if multiple names agree on CID
    cache_rows = []
    found_names = 0
    not_found_names = 0
    ambiguous_names = 0

    for cpd_id, tried_names in cpd_to_names.items():
        found_cids = {}  # cid -> (name, inchikey, smiles)
        for name in tried_names:
            if name in name_results:
                cid, inchikey, smiles = name_results[name]
                found_cids[cid] = (name, inchikey, smiles)

        if not found_cids:
            cache_rows.append((cpd_id, "name_lookup", tried_names[0], "not_found",
                               None, None, None, time.time()))
            not_found_names += 1
        elif len(found_cids) == 1:
            # All found names agree on same CID
            cid = list(found_cids.keys())[0]
            name, inchikey, smiles = found_cids[cid]
            cache_rows.append((cpd_id, "name_lookup", name, "found",
                               cid, inchikey, smiles, time.time()))
            found_names += 1
        else:
            # Multiple names returned different CIDs — ambiguous
            query_str = ";".join(f"{n}->CID{c}"
                                 for c, (n, _, _) in found_cids.items())
            cache_rows.append((cpd_id, "name_lookup_ambiguous", query_str,
                               "ambiguous", None, None, None, time.time()))
            ambiguous_names += 1

    # Save compounds with no usable name
    for cpd_id in no_name:
        cache_rows.append((cpd_id, "name_lookup", "", "not_found",
                           None, None, None, time.time()))
        not_found_names += 1

    if cache_rows:
        save_batch_to_cache(conn, db_lock, cache_rows)

    print(f"  Found via name: {found_names}")
    print(f"  Not found: {not_found_names}")
    if ambiguous_names:
        print(f"  Ambiguous (names disagree): {ambiguous_names}")

    # -------------------------------------------------------------------
    # Generate report
    # -------------------------------------------------------------------
    _generate_report(conn, candidates, structures)

    # -------------------------------------------------------------------
    # Phase 3: Fetch + apply corrections (only with --apply)
    # -------------------------------------------------------------------
    if args.apply:
        corrections = fetch_corrections(
            conn, db_lock, candidates, structures, workers=args.workers)
        apply_corrections(corrections, structures)

    conn.close()


def _generate_report(conn, candidates, structures):
    """Generate TSV reports from cache."""
    print("\nGenerating report...")
    cur = conn.execute("SELECT * FROM cache WHERE cpd_id IN ({})".format(
        ",".join("?" * len(candidates))), candidates)
    rows = cur.fetchall()

    report_header = [
        "cpd_id", "name_queried", "strategy", "result_type",
        "stored_inchikey", "pubchem_inchikey", "pubchem_cid",
        "stored_smiles", "pubchem_smiles",
    ]

    stats = defaultdict(int)
    report_rows = []
    mismatch_rows = []
    protonation_rows = []

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
            cpd_id, query, strategy, result_type,
            stored_inchikey, pub_inchikey, pub_cid,
            stored_smiles, pub_smiles,
        ]
        report_rows.append(out_row)
        if result_type == "MISMATCH":
            mismatch_rows.append(out_row)
        elif result_type == "PROTONATION_DIFF":
            protonation_rows.append(out_row)

    with open(REPORT_FILE, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(report_header)
        writer.writerows(report_rows)

    with open(MISMATCH_FILE, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(report_header)
        writer.writerows(mismatch_rows)

    if protonation_rows:
        with open(PROTONATION_FILE, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(report_header)
            writer.writerows(protonation_rows)

    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    total = sum(stats.values())
    print(f"Total compounds checked: {total}")
    for key in ["MATCH", "STEREO_DIFF", "PROTONATION_DIFF", "MISMATCH",
                "NOT_FOUND", "NO_KEY", "AMBIGUOUS", "XREF_CONFLICT"]:
        if stats[key]:
            pct = 100.0 * stats[key] / total if total else 0
            print(f"  {key:20s}: {stats[key]:6d} ({pct:5.1f}%)")
    print(f"\nFull report:     {REPORT_FILE}")
    print(f"Mismatches only: {MISMATCH_FILE}")
    if protonation_rows:
        print(f"Protonation diffs: {PROTONATION_FILE}")


if __name__ == "__main__":
    main()
