"""Plot only completed, assembly-checked 50-pose candidate ensembles."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

RUN=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((RUN/'manifest.json').read_text())
    inputs={r['accession']:r for r in manifest['inputs']}
    rows=[json.loads(p.read_text()) for p in (RUN/'state').glob('*.done.json')]
    rows.sort(key=lambda r:r['A_deg'],reverse=True)
    if not rows:raise RuntimeError('No complete ensembles to plot')
    names={'Q2FWC5':'LdmS','P9WJL3':'MurE (Mtb)','Q2FZP6':'MurE (Sa)',
           'Q9WY79':'MurE (Tm)','P22188':'MurE (Ec)','P0AES0':'Gss (Ec)','P90518':'GSP (Cf)'}
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,(ax,dx)=plt.subplots(1,2,figsize=(12,max(4.2,len(rows)*.43+1.6)),sharey=True,
                             gridspec_kw={'width_ratios':[1.85,1]},layout='constrained')
    colors=['#0072B2' if r['phosphate']=='PA' else '#D55E00' for r in rows]
    y=np.arange(len(rows));a=[r['A_deg'] for r in rows];d=[r['Dh_A'] for r in rows]
    ax.barh(y,a,color=colors,height=.65);dx.barh(y,d,color=colors,height=.65)
    labels=[names.get(r['accession'],inputs[r['accession']]['gene'].split()[0])+' / '+r['accession'] for r in rows]
    ax.set_yticks(y,labels);ax.set_ylim(len(rows)-.4,-.6);dx.tick_params(axis='y',left=False,labelleft=False)
    for j,r in enumerate(rows):
        ax.text(r['A_deg']+1.5,j,f'{r["A_deg"]:.1f}',va='center')
        dx.text(r['Dh_A']+.25,j,f'{r["Dh_A"]:.2f}',va='center')
    ax.set_xlim(0,190);dx.set_xlim(0,max(d)*1.3)
    ax.set_xlabel(r'Weighted attack angle $A$ (degrees)')
    dx.set_xlabel(r'Harmonic distance $D_h$ ($\AA$)')
    for panel in [ax,dx]:panel.grid(axis='x',alpha=.15);panel.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor='#0072B2',label=r'$P_\alpha$: AMP-forming'),
                       Patch(facecolor='#D55E00',label=r'$P_\gamma$: ADP-forming')],
              loc='lower right',frameon=False,fontsize=10)
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f'Completed Boltz-2 candidate cofolds: {len(rows)} / 17\n50 poses each; ATP + adipate²⁻ + assigned Mg²⁺',fontsize=15)
    fig.supxlabel(r'$A=\sum(\theta_i/d_i)/\sum(1/d_i)$; larger means more in-line. '+
                  r'$D_h=50/\sum(1/d_i)$ reports absolute proximity.'+'\n'+
                  f'Structural comparison; target activity untested. Snapshot: {stamp}',fontsize=10)
    fig.savefig(RUN/'completed_candidate_geometry.png',dpi=180)
    fig.savefig(RUN/'completed_candidate_geometry.pdf')
    plt.close(fig)
    fields=['accession','family','phosphate','mg_count','n_models','A_deg','Dh_A','min_distance_A','median_distance_A','nearby_fraction_4A','mean_angle_deg']
    with (RUN/'completed_candidate_geometry.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)
    (RUN/'completed_plot_sources.json').write_text(json.dumps({'snapshot':stamp,'completed':len(rows),'requested':17,
        'included_accessions':[r['accession'] for r in rows], 'source':'state/*.done.json',
        'policy':'Only complete 50-pose ensembles that passed assembly checks; unfinished candidates are omitted.'},indent=2))
    print('Plotted',len(rows),'completed candidates')
    for r in rows:print(r['accession'],round(r['A_deg'],2),round(r['Dh_A'],2))

if __name__=='__main__':main()
