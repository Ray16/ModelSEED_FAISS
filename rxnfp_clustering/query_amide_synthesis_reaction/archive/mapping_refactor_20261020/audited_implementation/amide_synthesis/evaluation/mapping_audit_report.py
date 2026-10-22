"""Render mapping counts and evidence limitations without changing data."""
from html import escape
from pathlib import Path


def render(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    rows = summary['counts']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1, 1.45]})
    y = np.arange(len(rows))
    labels = [r['rxn_id'] + '\n' + r['reaction'].replace(' ligation', '') for r in rows]
    colors = ['#3568a8', '#63a5d2', '#c18b40', '#8754a1']
    for axis, columns, title in [
        (axes[0], [('saved_direct_accessions', 'Saved mapping'),
                   ('live_reaction_supported_accessions', 'Reaction-supported'),
                   ('trusted_clusters95', '95% representatives')], 'Reaction annotation support'),
        (axes[1], [('expanded_candidates', 'Raw candidate IDs'),
                   ('expanded_exact_sequences', 'Exact sequences'),
                   ('expanded_clusters99', '99% representatives'),
                   ('expanded_clusters95', '95% representatives')], 'Expanded candidates · inferred activity'),
    ]:
        width = 0.75 / len(columns)
        for i, (column, label) in enumerate(columns):
            values = [r[column] for r in rows]
            bars = axis.barh(y + (i - (len(columns)-1)/2)*width, values,
                             height=width*0.88, label=label, color=colors[i])
            axis.bar_label(bars, padding=3, fontsize=8)
        axis.set_yticks(y, labels if axis is axes[0] else ['']*len(rows), fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        axis.set_xlabel('UniProt identifiers / representative sequences')
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='x', alpha=0.15)
        axis.set_axisbelow(True)
        axis.set_xlim(0, max(r[c] for r in rows for c, _ in columns)*1.23+1)
        axis.legend(loc='upper center', bbox_to_anchor=(.5, -.17), frameon=False, fontsize=8, ncol=2)
    fig.suptitle('ModelSEED top five → UniProt: evidence and redundancy', fontsize=13)
    fig.tight_layout(rect=(0, .06, 1, .95))
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(output / ('mapping_counts.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    columns = [('rxn_id','ModelSEED'),('reaction','Reaction'),
               ('saved_direct_accessions','Saved direct'),('live_reaction_supported_accessions','Reaction-supported IDs'),
               ('trusted_clusters95','Supported 95% representatives'),('expanded_candidates','Expanded candidate IDs'),
               ('expanded_exact_sequences','Exact sequences'),('expanded_clusters99','99% representatives'),
               ('expanded_clusters95','95% representatives')]
    table = '<tr>' + ''.join('<th>'+escape(label)+'</th>' for _,label in columns) + '</tr>'
    table += ''.join('<tr>'+''.join('<td>'+escape(str(row[key]))+'</td>' for key,_ in columns)+'</tr>' for row in rows)
    totals = summary['totals']
    limitations = ''.join('<li>'+escape(s)+'</li>' for s in summary['limits'])
    chemistry = ''.join('<tr><td>'+r['rxn_id']+'</td><td>'+escape(r['chemical_status'])+'</td></tr>' for r in rows)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Top-five mapping audit</title>
<style>body{{font:16px/1.5 system-ui;max-width:1400px;margin:35px auto;padding:0 25px;color:#203041}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce3e8;text-align:left}}th{{background:#edf3f8}}img{{width:100%}}.note{{background:#fff4db;padding:18px;border-radius:8px}}a{{color:#215e9c}}details{{margin:15px 0}}</style>
<h1>Top-five reaction-to-enzyme mapping audit</h1>
<p>Retrieved {escape(summary['generated_utc'])}; UniProt release {escape(summary['uniprot_release'])}.</p>
<p><b>{totals['unique_trusted_accessions']} distinct reaction-supported UniProt identifiers</b> across five reactions.
Expanded discovery yields {totals['unique_expanded_accessions']} unique candidate identifiers and
{totals['expanded_clusters95']} representatives at 95% sequence identity.</p>
<p class="note">Reaction-supported means the exact Rhea reaction is present in the protein's own annotation.
Expanded candidates include EC/name/homology inference and must not be reported as experimentally verified reaction mappings.
The ModelSEED citrate products have missing stereochemical detail where explicitly flagged below.</p>
<img src="mapping_counts.svg" alt="Raw and deduplicated UniProt counts for each ModelSEED reaction">
<table>{table}</table>
<h2>What improved</h2><p>The historical merged Rhea/GO mapping is superseded by an evidence-scoped direct mapping.
For rxn47618, the MetaCyc RXN-19820 cross-reference independently leads to RHEA:59124 and SbnE (Q2G1M9).
The local duplicate rxn54616 supplies the Rhea aliases. Full local reaction structures, cofactor turnover,
stoichiometry and specified stereochemistry agree. Missing stereochemistry remains a qualification.</p>
<h2>Redundancy rule</h2><p>Exact sequence duplicates are removed first. CD-HIT then requires 99% or 95%
identity (identical residues divided by the full shorter sequence), with at least 90% alignment coverage
of each protein. These are representative-based sequence groups, not proof of duplicate genes.
Organism/proteome labels alone never merge proteins.</p>
<h2>Chemical audit</h2><table><tr><th>ModelSEED</th><th>Rhea/ChEBI comparison</th></tr>{chemistry}</table>
<h2>Downloads</h2><p><a href="counts.tsv">Counts</a> · <a href="reaction_candidates.tsv">All retained candidates</a> ·
<a href="supported_reaction_links.tsv">Reaction-supported links and evidence</a> ·
<a href="cluster_membership.tsv">Every cluster membership</a> · <a href="rejected_search_hits.tsv">Rejected search hits</a> ·
<a href="chemical_audit.json">Full chemical checks</a> · <a href="manifest.json">Provenance manifest</a> ·
<a href="mapping_counts.png">PNG figure</a> · <a href="mapping_counts.pdf">PDF figure</a></p>
<h2>Interpretation limits</h2><ul>{limitations}</ul></html>'''
    (output / 'report.html').write_text(html)
