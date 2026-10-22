"""Verify complete ensembles and calculate the unchanged proximity-weighted angle."""
import hashlib
import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np
import yaml


def summary(distances, angles, include_near_attack=True):
    d, a = np.asarray(distances), np.asarray(angles)
    if d.shape != (50,) or a.shape != (50,) or not np.isfinite(d).all() or not np.isfinite(a).all() or (d <= 0).any():
        raise ValueError('Expected 50 finite positive-distance/angle pairs')
    inv = 1 / d
    result = {'A_deg': float((a * inv).sum() / inv.sum()), 'Dh_A': float(50 / inv.sum()),
            'min_distance_A': float(d.min()), 'median_distance_A': float(np.median(d)),
            'nearby_fraction_4A': float((d < 4).mean()), 'mean_angle_deg': float(a.mean())}
    if include_near_attack:
        result['near_attack_fraction'] = float(((d<4)&(a>150)).mean())
    return result

def inspect(item, out, include_near_attack=True):
    if hashlib.sha256(Path(item['input']).read_bytes()).hexdigest()!=item['input_sha256']:
        raise ValueError('Input identity changed')
    models = sorted(Path(out).glob(f'by_seed/{item["name"]}/s42/{item["name"]}_model_*.pdb'))
    expected = {f'{item["name"]}_model_{i}.pdb' for i in range(50)}
    if {p.name for p in models} != expected:
        raise ValueError(f'Expected all 50 unique models; found {len(models)}')
    distances, angles, poses = [], [], []
    expected_sequence = yaml.safe_load(Path(item['input']).read_text())['sequences'][0]['protein']['sequence']
    for path in models:
        confidence = path.with_name('confidence_' + path.stem + '.json')
        if not confidence.exists():
            raise ValueError('Missing confidence ' + str(confidence))
        json.loads(confidence.read_text())
        model = gemmi.read_structure(str(path))[0]
        if len(model['A']) != item['sequence_length']:
            raise ValueError('Protein residue-count mismatch ' + str(path))
        actual_sequence = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in model['A'])
        if actual_sequence != expected_sequence:
            raise ValueError('Protein sequence mismatch ' + str(path))
        nuc = {a.name: a.pos for r in model['B'] for a in r if a.element.name != 'H'}
        if not {'PA', 'PG', 'O3A', 'O3B'} <= nuc.keys():
            raise ValueError('Incomplete ATP ' + str(path))
        mg = [a for c in model for r in c for a in r if a.element.name == 'Mg']
        if len(mg) != item['mg_count']:
            raise ValueError('Wrong Mg count ' + str(path))
        acid = [a for r in model[item['acid_chain']] for a in r if a.element.name != 'H']
        if len(acid) != 10 or sum(a.element.name == 'C' for a in acid) != 6 or sum(a.element.name == 'O' for a in acid) != 4:
            raise ValueError('Wrong adipate heavy-atom composition ' + str(path))
        p, leaving = nuc[item['phosphate']], nuc[item['leaving_oxygen']]
        oxygens = [a for a in acid if a.element.name == 'O']
        oxygen = min(oxygens, key=lambda a: a.pos.dist(p))
        d = oxygen.pos.dist(p)
        u = np.array([oxygen.pos.x-p.x, oxygen.pos.y-p.y, oxygen.pos.z-p.z])
        v = np.array([leaving.x-p.x, leaving.y-p.y, leaving.z-p.z])
        angle = math.degrees(math.acos(float(np.clip(u @ v / (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))
        distances.append(d); angles.append(angle)
        poses.append({'model': path.name, 'distance_A': d, 'angle_deg': angle})
    return {'accession': item['accession'], 'name': item['name'], 'family': item['mechanism_family'],
            'phosphate': item['phosphate'], 'mg_count': item['mg_count'], 'n_models': 50,
            'assembly_qc': 'protein length, ATP atom names, adipate composition and Mg count passed',
            **summary(distances, angles, include_near_attack), 'poses': poses}

def aggregate(run_dir, *, write=False):
    RUN = Path(run_dir)
    rows = []
    for p in (RUN/'state').glob('*.done.json'):
        row = json.loads(p.read_text())
        rows.append({k: v for k, v in row.items() if k != 'poses'})
    rows.sort(key=lambda r: r['A_deg'], reverse=True)
    if rows and write:
        with (RUN/'geometry_ranking.tsv').open('w') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    return rows

