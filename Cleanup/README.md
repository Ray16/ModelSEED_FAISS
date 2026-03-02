# ModelSEED Structure Cleanup
## `cleanup.py`

Generates a clean, deduplicated structure file (`Unique_ModelSEED_Structures_new.txt`) from the raw multi-source ModelSEED compound database (`All_ModelSEED_Structures.txt`).

### Pipeline (3 stages)

1. **Parse & Filter** — Reads the raw `All_ModelSEED_Structures.txt` (~281K rows), keeps only `Charged` status rows (~141K), and groups them by compound ID. For each compound, it collects all external IDs as aliases and all structure values (SMILE/InChI/InChIKey) across sources (KEGG, MetaCyc, ChEBI, etc.).

2. **Validate & Resolve** (parallelized across 64 workers) — For each of the ~37K compounds:
   - **Conflict resolution:** When multiple sources disagree on a structure, picks the majority-vote winner.
   - **Cross-validation:** Independently computes InChIKey from both SMILES and InChI via RDKit. If they agree, structures are consistent. If they disagree, **InChI is trusted** as the canonical standard and SMILES/InChIKey are recomputed from it.
   - **Gap filling:** Derives missing structure types from what's available (e.g., generates InChI from SMILES if InChI is absent).
   - **Invalid removal:** Structures that fail RDKit parsing are dropped with a warning.

3. **Write Output** — Produces a 6-column `Unique_ModelSEED_Structures_new.txt` with header (`ID, Type, Aliases, Formula, Charge, Structure`), writing up to 3 rows per compound (SMILE, InChI, InChIKey).

### Usage

```bash
python cleanup.py
```

**Input:** `All_ModelSEED_Structures.txt` (8-column TSV, no header)
**Output:** `Unique_ModelSEED_Structures_new.txt` (6-column TSV with header)

## `diff_structures.py`

Compares `Unique_ModelSEED_Structures.txt` (old) against `Unique_ModelSEED_Structures_new.txt` (new) and produces `diff_report.csv` detailing every difference.

### Usage

```bash
python diff_structures.py
```

**Output:** `diff_report.csv` with columns: `ID, Type, Change, Field, Old_Value, New_Value`

Change types:
- **added** — row exists only in new
- **removed** — row exists only in old
- **modified** — row exists in both but a field differs (one row per changed field)

Prints both row-level and compound-level summaries.
