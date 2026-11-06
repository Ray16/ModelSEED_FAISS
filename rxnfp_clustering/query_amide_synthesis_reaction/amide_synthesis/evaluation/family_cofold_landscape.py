"""Summarize domain families, seed-defined enzyme groups and within-group ranks."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DOMAIN_FAMILIES = {'PF00120': 'Glutamine synthetase-like', 'PF04183': 'IucA/IucC',
                   'PF27279': 'DdaG-like', 'PF13535': 'ATP-grasp',
                   'PF30763': 'ATP-grasp', 'PF01225': 'MurCDEF'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(data, key):
    rows = []
    for name, group in data.groupby(key):
        best = group.sort_values('two_step_geometry_rank').iloc[0]
        rows.append(dict(group=name, n=len(group), best_accession=best.accession,
                         best_min_fraction=best.two_step_min_fraction,
                         median_min_fraction=group.two_step_min_fraction.median(),
                         q25_min_fraction=group.two_step_min_fraction.quantile(.25),
                         q75_min_fraction=group.two_step_min_fraction.quantile(.75),
                         best_step1_fraction=best.step1_near_attack_fraction,
                         best_step2_fraction=best.step2_near_attack_fraction,
                         both_steps_nonzero=int(group.two_step_min_fraction.gt(0).sum()),
                         reference_accessions=';'.join(sorted(group.reference_seed.unique())),
                         broad_family=best.broad_family))
    result = pd.DataFrame(rows).sort_values(['best_min_fraction', 'median_min_fraction', 'group'],
                                           ascending=[False, False, True]).reset_index(drop=True)
    result['best_geometry_rank'] = result.best_min_fraction.rank(method='min', ascending=False).astype(int)
    return result


def draw(summary, members, key, out, title):
    height = max(6.2, len(summary)*.4+2.9)
    fig, axes = plt.subplots(1, 5, figsize=(18, height), sharey=True,
                             gridspec_kw={'width_ratios':[.75, 1.3, 1, 1, 1.1], 'wspace':.22})
    colors = {name: plt.get_cmap('tab10')(i) for i, name in enumerate(summary.broad_family.unique())}
    rng = np.random.default_rng(42)
    for row, entry in enumerate(summary.to_dict('records')):
        group = members[members[key].eq(entry['group'])]
        color = colors[entry['broad_family']]
        for ax in axes:
            ax.axhspan(row-.45, row+.45, color=color, alpha=.055, zorder=0)
        axes[0].scatter(entry['n'], row, color=color, s=45)
        axes[0].annotate(str(entry['n']), (entry['n'],row), xytext=(6,0), textcoords='offset points', va='center', fontsize=10)
        axes[1].scatter(group.two_step_min_fraction, row+rng.uniform(-.12,.12,len(group)),
                        s=16, color=color, alpha=.35, linewidths=0)
        axes[1].scatter(entry['median_min_fraction'],row,marker='|',s=180,color='#222222',zorder=3)
        axes[1].scatter(entry['best_min_fraction'],row,marker='D',s=38,color=color,edgecolors='white',linewidths=.5,zorder=4)
        for ax, field in [(axes[2],'best_step1_fraction'),(axes[3],'best_step2_fraction')]:
            ax.barh(row,entry[field],height=.45,color=color,alpha=.8)
            ax.annotate(f'{entry[field]:.0%}',(entry[field],row),xytext=(5,0),textcoords='offset points',va='center',fontsize=10)
        axes[4].text(.02,row,f'{entry["best_accession"]}   ({entry["best_min_fraction"]:.0%})',va='center',fontsize=10)
    axes[0].set(xscale='log',xlim=(.8,max(10,summary.n.max()*2.2)),xlabel='Modeled sequences (log)')
    for ax in axes[1:4]:
        ax.set_xlim(-.025,1.16)
        ax.set_xticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
        ax.grid(axis='x',alpha=.15)
    axes[1].set_xlabel('Minimum of the two pose fractions')
    axes[2].set_xlabel('Step 1 near-attack fraction')
    axes[3].set_xlabel('Step 2 near-attack fraction')
    axes[4].set_xlim(0,1);axes[4].set_xticks([])
    titles=['Coverage','All homologs\nline: median; diamond: best','Best representative\nStep 1','Same representative\nStep 2','Representative\n(minimum fraction)']
    for ax,label in zip(axes,titles):
        ax.set_title(label,fontsize=11,weight='bold',pad=13)
        ax.set_ylim(len(summary)-.5,-.5)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    axes[4].spines['bottom'].set_visible(False)
    labels=['\n'.join(textwrap.wrap(name,26)) for name in summary.group]
    axes[0].set_yticks(range(len(summary)),labels)
    for label, name in zip(axes[0].get_yticklabels(), summary.broad_family):label.set_color(colors[name])
    fig.suptitle(title,fontsize=16,weight='bold',y=.97)
    fig.text(.5,.915,'Both stages are evaluated together. Large groups expose more candidates to selection; coverage and full distributions are shown.',ha='center',fontsize=10)
    fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color=color,label=name) for name,color in colors.items()],
               loc='lower center',bbox_to_anchor=(.5,.075),ncol=len(colors),frameon=False,fontsize=9)
    fig.text(.5,.025,'Structural heuristic: min(step 1 near-attack fraction, step 2 near-attack fraction). Best representatives maximize this value; tied scores are not distinct activity ranks.\n'
             'Family labels come from reference-enzyme UniProt/Pfam annotations transferred through the saved homolog mapping. This is not a phylogenetic orthology analysis.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.18,right=.99,top=.82,bottom=.17)
    for extension in ['png','svg','pdf']:fig.savefig(out.with_suffix('.'+extension),dpi=220,facecolor='white')
    plt.close(fig)


def render(campaign, out):
    root=campaign/'lambda_fleet'
    source=root/'analysis/two_step/all_candidates_two_step.tsv'
    mapping_path=campaign/'mapping/fold_candidates.tsv'
    pool_path=campaign/'mapping/candidate_pool.tsv'
    annotation_path=out/'reference_annotations/summary.json'
    annotation=json.loads(annotation_path.read_text())
    annotations={r['accession']:r for r in annotation['references']}
    mapping=pd.read_csv(mapping_path,sep='\t').fillna('')
    pool=pd.read_csv(pool_path,sep='\t').fillna('').drop_duplicates('accession').set_index('accession')
    candidates=pd.read_csv(source,sep='\t')
    assignments=[]
    for member in mapping.to_dict('records'):
        seed=member['reference_accession'] or member['accession']
        reference=pool.loc[seed]
        gene=reference.gene_names.split(';')[0]
        enzyme=gene if gene and '_' not in gene else reference.protein_name
        pfam={v['id'] for v in annotations[seed]['pfam']}
        broad={DOMAIN_FAMILIES[v] for v in pfam if v in DOMAIN_FAMILIES}
        if len(broad)>1:raise ValueError('Ambiguous domain-family assignment: '+seed)
        broad=next(iter(broad)) if broad else 'Unclassified: '+reference.protein_name
        assignments.append(dict(accession=member['accession'],reference_seed=seed,enzyme_group=enzyme,
                                broad_family=broad,reference_gene_names=reference.gene_names,
                                reference_protein_name=reference.protein_name,reference_pfam=';'.join(sorted(pfam)),
                                mapping_reference_identity=member['identity'],mapping_route=member['mapping_route']))
    data=candidates.merge(pd.DataFrame(assignments),on='accession',validate='one_to_one')
    if len(data)!=len(candidates) or set(data.accession)!=set(candidates.accession):raise ValueError('Lost modeled candidates')
    data=data.sort_values('two_step_geometry_rank')
    data['rank_within_enzyme']=data.groupby('enzyme_group').cumcount()+1
    data['rank_within_family']=data.groupby('broad_family').cumcount()+1
    enzymes=summarize(data,'enzyme_group');families=summarize(data,'broad_family');seeds=summarize(data,'reference_seed')
    families['share_of_cohort']=families.n/len(data)
    family_order={name:i for i,name in enumerate(families.group)}
    enzymes['family_order']=enzymes.broad_family.map(family_order)
    grouped=enzymes.sort_values(['family_order','best_min_fraction','median_min_fraction','group'],ascending=[True,False,False,True])
    out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'all_candidates_family_ranked.tsv',sep='\t',index=False)
    enzymes.drop(columns='family_order').to_csv(out/'enzyme_group_ranking.tsv',sep='\t',index=False)
    families.to_csv(out/'broad_family_ranking.tsv',sep='\t',index=False)
    seeds.to_csv(out/'reference_seed_ranking.tsv',sep='\t',index=False)
    (out/'family_definitions.json').write_text(json.dumps(DOMAIN_FAMILIES,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    draw(grouped,data,'enzyme_group',out/'enzyme_landscape','Glutarate + cadaverine: enzyme landscape across both reaction steps')
    draw(families,data,'broad_family',out/'broad_family_landscape','Glutarate + cadaverine: broad protein families and sampling coverage')
    member_columns=['rank_within_enzyme','rank_within_family','accession','reference_seed','family',
                    'step1_plot_weighted_angle_deg','step1_plot_harmonic_distance_A','step1_near_attack_fraction',
                    'step2_plot_weighted_angle_deg','step2_plot_harmonic_distance_A','step2_near_attack_fraction','two_step_min_fraction']
    table=lambda frame:frame.to_html(index=False,escape=True,float_format=lambda value:f'{value:.3f}')
    sections=[]
    for family in families.group:
        section=f'<h2>{html.escape(family)}</h2>'
        for _,entry in grouped[grouped.broad_family.eq(family)].iterrows():
            members=data[data.enzyme_group.eq(entry.group)]
            filename='members_'+hashlib.sha256(entry.group.encode()).hexdigest()[:10]+'.tsv'
            members.to_csv(out/filename,sep='\t',index=False)
            section+=(f'<details><summary>{html.escape(entry.group)} · {len(members)} modeled · '
                      f'best two-step minimum {entry.best_min_fraction:.0%} · representative {entry.best_accession}</summary>'
                      f'<p><a href="{filename}">Download ranked members</a></p>'+table(members[member_columns])+'</details>')
        sections.append(section)
    refs=''.join(f'<li><a href="{html.escape(r["url"])}">{r["accession"]}</a>: '
                 +html.escape('; '.join(r['similarity']) or ', '.join(v['EntryName'] for v in r['pfam']))+'</li>'
                 for r in annotation['references'])
    page=('<!doctype html><meta charset="utf-8"><title>Family landscape: two-step cofolds</title>'
          '<style>body{font:15px/1.5 system-ui;margin:30px;color:#183348}img{width:100%;max-width:1800px}table{border-collapse:collapse;font-size:12px}'
          'td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}summary{cursor:pointer;padding:12px;background:#edf3f6;margin:5px 0}details{overflow:auto}</style>'
          '<h1>Glutarate + cadaverine: the full modeled enzyme landscape</h1>'
          f'<p><b>{len(data)} modeled sequences · {len(enzymes)} reference/gene-defined enzyme groups · {len(families)} broad domain families · both steps complete.</b></p>'
          '<p>The landscape is displayed by protein family and enzyme group so a large homolog pool cannot occupy all visible shortlist rows. '
          'Within each group, sequences are ordered by min(step 1 near-attack fraction, step 2 near-attack fraction), with step 1 fraction, step 2 fraction and accession as tie breakers. '
          'Family/group summary ranks use the best minimum fraction; identical values share a rank. This is structural prioritization, not a calibrated activity prediction. '
          'Best-of-group values favor deeply sampled groups; medians, full distributions and counts are shown alongside them.</p>'
          '<img src="enzyme_landscape.svg" alt="Enzyme groups, coverage, all-homolog score distributions and representative results for both steps">'
          '<p><a href="enzyme_landscape.pdf">Main landscape PDF</a> · <a href="enzyme_landscape.png">PNG</a> · '
          '<a href="enzyme_group_ranking.tsv">Enzyme-group summary</a> · <a href="all_candidates_family_ranked.tsv">All ranked members</a></p>'
          '<h2>Broad protein families</h2><img src="broad_family_landscape.svg" alt="Five broad families and unequal retrieval coverage">'+table(families)
          +'<h2>Enzyme groups, collapsed across homologs</h2>'+table(enzymes.drop(columns='family_order'))
          +'<p>Grouping uses each retained sequence’s saved reference_accession, or its own accession for direct references. '
          'Primary reference gene names join seed lineages into enzyme groups; groups without a named gene retain their reference enzyme name. '
          'Reference Pfam domains define the broader families. These are auditable homolog/annotation groups; strict orthology has not been established phylogenetically. '
          'GlnA2 includes both I3R584- and Q9RDS6-linked lineages (nine sequences); the explicitly named Q9RDS6 is only one of them.</p>'
          '<h2>Reference seed lineages</h2>'+table(seeds)
          +'<h2>Ranked candidates within each family and enzyme group</h2>'+''.join(sections)
          +'<h2>Supporting views and input checks</h2><p><a href="../two_step/report.html">Individual two-step shortlist and original upper-left comparison</a> · '
          '<a href="../../evidence/substrate_pair_recheck_20261105.json">Actual substrate/configuration recheck</a></p>'
          '<p>Both-stage plots use the current C5/C5 campaign. Step 1 near attack: d &lt;4 Å and θ &gt;150°. '
          'Step 2: d &lt;3.5 Å and 90° ≤ θ ≤120°, excluding pocket violations. Step 2 uses forced pocket restraints and assigned activation routes.</p>'
          '<h2>Primary annotation sources</h2><ul>'+refs+'</ul>')
    (out/'report.html').write_text(page)
    source_provenance=json.loads((root/'analysis/two_step/provenance.json').read_text())
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_provenance['source_sha256'],
                 input_hashes={str(p.relative_to(campaign)):digest(p) for p in [source,mapping_path,pool_path,annotation_path]},
                 generator_sha256=digest(Path(__file__)),candidates=len(data),enzyme_groups=len(enzymes),broad_families=len(families),
                 outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'})
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(families[['group','n','best_min_fraction']].to_string(index=False))
    print(enzymes[['best_geometry_rank','group','n','best_accession','best_step1_fraction','best_step2_fraction','best_min_fraction']].to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve()
    render(campaign,campaign/'lambda_fleet/analysis/family_landscape')


if __name__=='__main__':main()
