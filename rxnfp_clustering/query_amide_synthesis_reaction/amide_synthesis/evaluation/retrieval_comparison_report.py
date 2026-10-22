"""Report retrieval decisions through organic reacting-partner chemistry."""
from html import escape


def render(frame, coverage, policy, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from amide_synthesis.evaluation.compare_retrieval import METHODS

    methods = list(METHODS)
    counts = coverage.pivot(index='method', columns='cutoff', values='non_preamidated_analogue_groups').loc[methods, [20, 50, 100]]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout='constrained')
    values = counts.to_numpy()
    im = ax.imshow(values, vmin=0, vmax=5, cmap='RdBu_r', aspect='auto')
    ax.set_yticks(range(len(methods)), [METHODS[m] for m in methods], fontsize=9)
    ax.set_xticks(range(3), ['First 20 retrieved', 'First 50 retrieved', 'First 100 retrieved'])
    for i in range(len(methods)):
        for j in range(3):
            ax.text(j, i, str(values[i, j]) + '/5', ha='center', va='center',
                    color='white' if values[i, j] in [0, 1, 5] else '#202020')
    ax.set_title('Recovery of target-like monomer-coupling groups')
    fig.supxlabel('Same structural criteria for every method; full equations retain cofactors.', fontsize=8)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(output / ('chemical_coverage.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    examples = [
        ('Target adipate + HMDA', 'Straight-chain aliphatic diacid + primary diamine; one new amide; free acid and amine remain.'),
        ('Glutamate + putrescine', 'Same diacid/diamine coupling architecture. Extra amino group, chirality and shorter spacers remain differences.'),
        ('Fumarate + diaminopropionate', 'Same diacid/diamine architecture. Unsaturated short acid and a short chiral, carboxylated diamine.'),
        ('Citrate + spermidine / D-ornithine / diaminopropionate', 'Acyclic polyacid/diamine coupling with ends retained; extra carboxyl/OH groups, branches or a secondary amine reduce the match.'),
        ('Hydroxamate precursor oligomerization', 'Flexible and aliphatic, but starts from pre-amidated amino-acid-type monomers; N–O groups and AB self-condensation differ from free AA + BB monomers. Both methods retrieve these prominently.'),
        ('Glutathione–polyamine intermediates', 'Pre-existing peptide bonds, sulfur, chirality and larger substituted partners make these less literal adipate/HMDA analogues.'),
        ('Aromatic acyl-CoA amide formation', 'Aromatic or long-chain monofunctional partners and activated CoA donors; generally lacks the target free acid/free amine end pattern.'),
    ]
    table = ''.join('<tr><td>' + escape(a) + '</td><td>' + escape(b) + '</td></tr>' for a, b in examples)
    downloads = ' · '.join('<a href="' + m + '_ranking.tsv">' + escape(label) + '</a>' for m, label in METHODS.items())
    html = '''<!doctype html><meta charset="utf-8"><title>Choosing nylon reaction retrieval</title>
<style>body{font:16px/1.5 system-ui;max-width:1150px;margin:35px auto;padding:0 22px;color:#203041}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#edf3f8}img{width:100%}a{color:#215e9c}.note{padding:16px;background:#edf3f8}</style>
<h1>Production choice: DRFP–Tanimoto plus chemical review</h1>
<p class="note">Judge the retrieved organic structures against free adipate + HMDA. All encoded reactions retain their cofactors. The current production top 100 is reproduced exactly; the selected five reaction groups are unchanged.</p>
<h2>What the SMILES show</h2><table><tr><th>Reaction family</th><th>Structural interpretation</th></tr>''' + table + '''</table>
<p>Both DRFP–Tanimoto and the original hybrid-cosine ranking recover the same five relevant monomer-coupling groups within 100 entries. DRFP brings these groups into a smaller retrieval set more effectively under this rubric. Neither method's unfiltered leading hits are a substitute for reacting-group review.</p>
<h2>Did adding ECFP4 to Tanimoto help?</h2>
<p>The signed ECFP4 count-difference block cannot be treated as ordinary binary bits. We tested a valid 6,144-bit concatenation of positive ECFP4 changes, negative ECFP4 changes, and DRFP. It did not improve recovery of the relevant monomer-coupling groups. A second 6,144-bit representation retaining ECFP4 reactant/product context promoted long-chain acyl-glycine and hydroxamate chemistry and missed two of the five groups within 100 entries.</p>
<img src="chemical_coverage.svg" alt="Recovery of the five target-like monomer-coupling groups by each retrieval method">
<p>The heatmap counts structural categories, not enzyme activity. ATP-bookkeeping duplicates are collapsed when counting recovered groups. The rubric is target-specific and is not an independent experimental ground truth.</p>
<h2>Production configuration</h2><p>DRFP–Tanimoto is primary; ECFP4-difference + DRFP cosine breaks ties. Explicit reacting-group chemistry determines the final shortlist. Other configurations remain diagnostic comparisons. A full ECFP4 concatenation is not promoted to the primary ranker without a demonstrated structural benefit.</p>
<h2>Reproducible data</h2><p>''' + downloads + '''</p><p><a href="all_480_reactions.tsv">All 480 reactions, SMILES and chemical features</a> · <a href="chemistry_coverage.tsv">Chemical coverage counts</a> · <a href="production_policy.json">Production decision</a> · <a href="manifest.json">Hashes</a></p>'''
    (output / 'report.html').write_text(html)
