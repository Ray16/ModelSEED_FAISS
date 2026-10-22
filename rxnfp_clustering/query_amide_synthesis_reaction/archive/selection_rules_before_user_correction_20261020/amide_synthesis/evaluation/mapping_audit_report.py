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
<p>Computed {escape(summary['generated_utc'])}; UniProt release {escape(summary['uniprot_release'])}.
Mode: {escape(summary.get('execution_mode', 'saved database snapshot'))}.</p>
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


def render_final_review(audit_directory, verification):
    """Publish a current review around the immutable evidence snapshot."""
    import json
    import csv
    audit_directory = Path(audit_directory)
    summary = json.loads((audit_directory / 'summary.json').read_text())
    if verification['status'] != 'passed' or any(
            status != 'current' for status in verification['tracked_stages'].values()):
        raise ValueError('Cannot publish a completed review with stale pipeline artifacts')
    lines = [
        'We have a more reliable mapping method than the historical merged Rhea/GO mapping. '
        'The pipeline now uses evidence-scoped direct Rhea annotations and explicitly approved '
        'full-equation bridges when a ModelSEED entry lacks an alias. Returned annotations, '
        'sequence hashes, complete equations and supporting evidence are checked.',
        '',
        'For the current five chemically selected ModelSEED reactions, the counts are:',
        '',
        'ModelSEED | Reaction-supported UniProt IDs | Expanded candidate IDs | Exact sequences | 99% representatives | 95% representatives',
    ]
    for row in summary['counts']:
        lines.append(' | '.join(str(row[key]) for key in [
            'rxn_id', 'live_reaction_supported_accessions', 'expanded_candidates',
            'expanded_exact_sequences', 'expanded_clusters99', 'expanded_clusters95']))
    totals = summary['totals']
    lines += [
        '',
        f"The primary mapping has {totals['unique_trusted_accessions']} distinct reaction-supported UniProt identifiers. "
        'All eight have reaction-specific experimental annotation evidence (ECO:0000269); '
        'this is evidence for their native reactions, not a new demonstration of nylon activity. '
        'The eight sequences remain distinct after exact, 99% and 95% redundancy filtering.',
        '',
        'The recovered rxn47618 assignment is Q2G1M9 (SbnE). Rhea independently connects '
        'MetaCyc RXN-19820 to RHEA:59124; local rxn54616 supplies a matching complete '
        'ModelSEED equation. The bridge retains its source reaction and evidence. '
        'The rxn42510 and rxn47618 products lack a citrate stereocentre in the ModelSEED '
        'representation, so these are compatible annotations with an explicit stereochemical qualification.',
        '',
        f"The broader discovery pool contains {totals['unique_expanded_accessions']:,} distinct identifiers, "
        f"{totals['expanded_unique_sequences']:,} exact sequences, "
        f"{totals['expanded_clusters99']:,} representatives at 99% identity and "
        f"{totals['expanded_clusters95']:,} representatives at 95% identity. "
        'These extra candidates rely on EC or specific protein-name annotations plus reference-sequence '
        'similarity; they are not verified reaction assignments. Gene-only matches are rejected. '
        'Ten AsbA candidate records are fragments and remain flagged.',
        '',
        'Near-duplicate filtering uses CD-HIT with at least 90% aligned coverage of both proteins. '
        'Identity means identical residues divided by the full shorter-sequence length. '
        'The groups are representative-based sequence clusters; same-genome or same-organism labels '
        'alone never collapse proteins, and a cluster does not establish identical catalytic activity.',
        '',
        f"The implementation is separated into retrieval, reaction chemistry, approved bridges, sequence "
        f"redundancy and reporting. All {len(verification['tracked_stages'])} tracked stages were refreshed and verified current. "
        f"Validation: {verification['tests']}. Superseded scripts, historical slide code, old reports and "
        'obsolete log/build folders were moved into archive/mapping_refactor_20261020/ with path/hash manifests.',
    ]
    output = audit_directory.parent
    candidates = list(csv.DictReader((audit_directory / 'reaction_candidates.tsv').open(), delimiter='\t'))
    memberships = list(csv.DictReader((audit_directory / 'cluster_membership.tsv').open(), delimiter='\t'))
    for threshold in [99, 95]:
        selected = {(r['rxn_id'], r['representative']) for r in memberships
                    if r['pool'] == 'expanded' and float(r['identity']) == threshold / 100}
        representatives = [r for r in candidates if (r['rxn_id'], r['accession']) in selected]
        expected = sum(r['expanded_clusters' + str(threshold)] for r in summary['counts'])
        if len(representatives) != expected or len(selected) != expected:
            raise ValueError('Representative export does not match audited cluster counts')
        with (output / f'representatives{threshold}.tsv').open('w') as handle:
            writer = csv.DictWriter(handle, list(candidates[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(representatives)
        (output / f'representatives{threshold}.fasta').write_text(''.join(
            f">{r['accession']}|{r['rxn_id']}|{r['mapping_route']}\n{r['sequence']}\n"
            for r in representatives))
    draft = '\n'.join(lines) + '\n'
    (output / 'answer_draft.txt').write_text(draft)
    original = (audit_directory / 'report.html').read_text()
    original = original.replace('<style>', '<base href="' + audit_directory.name + '/"><style>', 1)
    marker = '<h1>Top-five reaction-to-enzyme mapping audit</h1>'
    notice = (f'<p class="note"><b>Pipeline integration verified:</b> '
              f'{len(verification["tracked_stages"])} tracked stages are current; '
              f'{escape(verification["tests"])}. '
              '<a href="../answer_draft.txt">Draft answer</a> · '
              '<a href="../pipeline_verification.json">Pipeline verification</a> · '
              '<a href="../representatives95.tsv">Filtered 95% identifier table</a> · '
              '<a href="../representatives95.fasta">Filtered 95% FASTA</a> · '
              '<a href="../representatives99.tsv">99% identifier table</a>.</p>')
    original = original.replace(marker, marker + notice, 1)
    original = original.replace('</html>', '<details><summary>Draft answer to share</summary><pre style="white-space:pre-wrap">'
                                + escape(draft) + '</pre></details></html>')
    (output / 'review.html').write_text(original)
