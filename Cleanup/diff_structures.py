"""Compare Unique_ModelSEED_Structures.txt (old) vs Unique_ModelSEED_Structures_new.txt (new).

Produces diff_report.csv with columns:
  ID, Type, Change, Field, Old_Value, New_Value

Change types:
  - added:    row exists only in new
  - removed:  row exists only in old
  - modified: row exists in both but a field differs (one row per changed field)
"""

import csv

OLD_PATH = 'Unique_ModelSEED_Structures.txt'
NEW_PATH = 'Unique_ModelSEED_Structures_new.txt'
OUT_PATH = 'diff_report.csv'

FIELDS = ['Aliases', 'Formula', 'Charge', 'Structure']


def load(path):
    """Return dict keyed by (ID, Type) -> {Aliases, Formula, Charge, Structure}."""
    rows = {}
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 6:
                continue
            id_, typ, aliases, formula, charge, structure = parts
            rows[(id_, typ)] = {
                'Aliases': aliases,
                'Formula': formula,
                'Charge': charge,
                'Structure': structure,
            }
    return rows


def main():
    old = load(OLD_PATH)
    new = load(NEW_PATH)

    all_keys = sorted(set(old.keys()) | set(new.keys()))

    diffs = []
    for key in all_keys:
        id_, typ = key
        in_old = key in old
        in_new = key in new

        if in_old and not in_new:
            diffs.append({
                'ID': id_, 'Type': typ, 'Change': 'removed',
                'Field': '', 'Old_Value': old[key]['Structure'], 'New_Value': '',
            })
        elif in_new and not in_old:
            diffs.append({
                'ID': id_, 'Type': typ, 'Change': 'added',
                'Field': '', 'Old_Value': '', 'New_Value': new[key]['Structure'],
            })
        else:
            # Both exist — check each field
            for field in FIELDS:
                ov = old[key][field]
                nv = new[key][field]
                if ov != nv:
                    diffs.append({
                        'ID': id_, 'Type': typ, 'Change': 'modified',
                        'Field': field, 'Old_Value': ov, 'New_Value': nv,
                    })

    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Type', 'Change', 'Field', 'Old_Value', 'New_Value'])
        writer.writeheader()
        writer.writerows(diffs)

    # Summary
    from collections import Counter

    added = sum(1 for d in diffs if d['Change'] == 'added')
    removed = sum(1 for d in diffs if d['Change'] == 'removed')
    modified = sum(1 for d in diffs if d['Change'] == 'modified')

    # Count unique compounds (IDs) affected by each change type
    cpds_added = set(d['ID'] for d in diffs if d['Change'] == 'added')
    cpds_removed = set(d['ID'] for d in diffs if d['Change'] == 'removed')
    cpds_modified = set(d['ID'] for d in diffs if d['Change'] == 'modified')
    cpds_any = cpds_added | cpds_removed | cpds_modified

    print(f"Diff report written to {OUT_PATH}")
    print(f"\n  Row-level summary:")
    print(f"    Rows added   (in new only): {added}")
    print(f"    Rows removed (in old only): {removed}")
    print(f"    Fields modified:            {modified}")
    print(f"    Total diff entries:         {len(diffs)}")
    print(f"\n  Compound-level summary:")
    print(f"    Compounds with additions:     {len(cpds_added)}")
    print(f"    Compounds with removals:      {len(cpds_removed)}")
    print(f"    Compounds with modifications: {len(cpds_modified)}")
    print(f"    Total compounds affected:     {len(cpds_any)}")

    # Breakdown of modified fields
    if modified:
        field_counts = Counter(d['Field'] for d in diffs if d['Change'] == 'modified')
        print(f"\n  Modified field breakdown:")
        for field, count in field_counts.most_common():
            print(f"    {field}: {count}")


if __name__ == '__main__':
    main()
