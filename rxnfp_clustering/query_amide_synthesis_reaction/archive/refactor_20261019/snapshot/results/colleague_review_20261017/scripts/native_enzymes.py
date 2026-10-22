"""Sequence-verified literature controls and native-reaction coverage."""
from artifacts import artifact, relative

from pathlib import Path
import sys,csv,json,glob,hashlib
import pandas as pd
from rdkit import Chem
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT))
from amide_synthesis.substrate_distance import score_reaction

FACTS={
 'DdaG':dict(genbank='ADN39487.1',reaction='Fumarate + L-2,3-diaminopropionate + ATP -> N3-fumaroyl-L-2,3-diaminopropionate + AMP + PPi',modelseed='rxn46045',source='https://doi.org/10.1021/bi9013165',similarity_reason='A free diacid acylates a free diamine, retaining acid and amine functionality. Fumarate is C4 and unsaturated versus C6 adipate; DAP has a short, carboxylated amine backbone versus C6 HMDA.',nylon_result='Active; strongest on succinate pairs; lower adipate/HMDA yield.'),
 'SfaB':dict(genbank='ATY72527.1',reaction='Native role: adenylation of 3-isocyanobutanoate (ATP -> acyl-AMP + PPi), supplying the SF2768 pathway. Amidation benchmark: 5-chlorovalerate + cadaverine + ATP -> N-(5-aminopentyl)-5-chloropentanamide + AMP + PPi.',modelseed='',source='https://doi.org/10.1002/anie.202010042',similarity_reason='The tested benchmark acylates an aliphatic C5 diamine, one carbon shorter than HMDA. Its C5 chloro-monoacid is not a diacid, so the product lacks a free acid for nylon-like extension. Native adenylation is an activation reaction, not a free-amine amidation.',nylon_result='Active; adipate/HMDA MA formation confirmed against standards and NMR.'),
 'SuCphA1':dict(genbank='BAA17890.1',reaction='Cyanophycin synthesis alternates ATP-dependent Asp backbone extension and addition of Arg at Asp side-chain carboxyls; ATP -> ADP + Pi at each ligase step.',modelseed='rxn48822;rxn49577;rxn16435;rxn16436;rxn19006;rxn19007',source='https://doi.org/10.1038/s41467-022-31542-7',similarity_reason='A natural polyamide polymerase repeatedly forms amide bonds and uses Asp, a diacid. Arg provides a single free alpha-amine; its guanidinium is not a second primary amine. Polymer recognition, branching, chirality and short spacers differ from adipate/HMDA.',nylon_result='No detectable nylon diad product under the reported conditions.'),
 'McbA':dict(genbank='AGL76720.1',reaction='1-acetyl-9H-beta-carboline-3-carboxylate + beta-phenethylamine (or substituted phenethylamines/tryptamine) + ATP -> marinacarboline amide + AMP + PPi',modelseed='',source='https://doi.org/10.1016/j.tetlet.2014.07.004',similarity_reason='Same ATP-driven acid activation and C(=O)-N bond formation, but the native acid and amine are monofunctional aromatic substrates. It neither starts from a diacid/diamine pair nor leaves both polymerizable ends.',nylon_result='No detectable nylon diad product under the reported conditions.'),
 'PylC':dict(genbank='AAZ69813.1',reaction='(3R)-3-methyl-D-ornithine + L-lysine + ATP -> (3R)-3-methyl-D-ornithyl-N6-L-lysine + ADP + Pi + H+',modelseed='rxn23366',source='https://doi.org/10.1016/j.jmb.2012.09.007',similarity_reason='Forms an isopeptide amide at a terminal lysine epsilon-amine and preserves another amine and acid. Lysine resembles a functionalized C5 diamine; the donor is a methylated amino monoacid, not adipate. Plain ornithine is a surrogate, not the physiological methylornithine substrate.',nylon_result='His-tagged construct insoluble; soluble His-SUMO construct failed native-activity control. No valid negative nylon test.'),
}

def main():
    ranks=pd.read_csv(artifact('all_screened_reactions.tsv'),sep='\t').set_index('rxn_id')
    metadata={}
    for f in sorted((ROOT/'ModelSEEDDatabase/Biochemistry').glob('reaction_*.tsv')):
        for r in csv.DictReader(f.open(),delimiter='\t'):metadata[r['id']]=r
    links={}
    for l in (PROJECT/'uniprot-msd-links.tsv').open():
        a,r,s=l.rstrip('\n').split('\t');links[a]=r
    rows=[];fasta=[]
    for name,info in FACTS.items():
        d=json.loads((RUN/'evidence'/f'{name}_uniprot.json').read_text())['results'][0]
        ncbi=(RUN/'evidence'/f'{name}_{info["genbank"]}.fasta').read_text()
        nseq=''.join(ncbi.splitlines()[1:]);seq=d['sequence']['value'];assert seq==nseq,(name,'sequence mismatch')
        accession=d['primaryAccession'];rxn=info['modelseed'].split(';')[0]
        score=ranks.loc[rxn,'chemistry_score'] if rxn in ranks.index else None
        rank=ranks.loc[rxn,'chemistry_rank'] if rxn in ranks.index else None
        target_score=ranks.loc[rxn,'nylon66_score'] if rxn in ranks.index else None
        target_rank=ranks.loc[rxn,'nylon66_rank'] if rxn in ranks.index else None
        # This is a representative, literature-tested promiscuous reaction, separate from
        # SfaB's native activation. Do not silently label its score the native score.
        benchmark_score=None
        if name=='SfaB':benchmark_score,_=score_reaction('O=C(O)CCCCCl.NCCCCCN>>O=C(CCCCCl)NCCCCCN.O')
        row=dict(enzyme=name,uniprot=accession,genbank=info['genbank'],length=len(seq),exact_sequence_match=True,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),organism=d['organism']['scientificName'],native_reaction=info['reaction'],native_reaction_modelseed_ids=info['modelseed'],reaction_database_status='present' if info['modelseed'] else 'no exact native/free-amidation entry located in this ModelSEED snapshot',native_chemistry_score=score,native_chemistry_rank=rank,sfab_promiscuous_benchmark_score=benchmark_score,original_file_maps_accession=accession in links,original_file_reactions=links.get(accession,''),similarity_reason=info['similarity_reason'],nylon_result=info['nylon_result'],source=info['source'],uniprot_url=f'https://www.uniprot.org/uniprotkb/{accession}/entry',ncbi_url=f'https://www.ncbi.nlm.nih.gov/protein/{info["genbank"]}')
        row.update(native_nylon66_score=target_score,native_nylon66_rank=target_rank)
        rows.append(row);fasta.append(f'>{name}|{accession}|{info["genbank"]}\n{seq}\n')
    pd.DataFrame(rows).to_csv(artifact('five_original_enzymes.tsv'),sep='\t',index=False)
    (artifact('five_original_enzymes.fasta')).write_text(''.join(fasta))
    # A polymer with an unspecified repeat count has no unique molecular fingerprint.
    # Keep that as missing, rather than fabricate a numeric CphA similarity.
    notes={'SuCphA1_accession_caution':'BAA17890.1 matches P73833 (PCC6803/Kazusa) exactly. The cited Sharon study uses a SuCphA1 from UTEX2470/PCC6714. These labels are not interchangeable; the requested original is assigned from the actual sequence accession in Josh\'s Methods.',
           'SfaB_coverage_search':'No free 3-isocyanobutanoate or 5-chlorovalerate native/benchmark entry found by compound/reaction name/alias search. rxn58045 contains an ACP-bound isocyanobutanoyl donor in a downstream reductive assembly; it is not SfaB adenylation or the tested free-amine coupling.',
           'McbA_coverage_search':'No marinacarboline or 1-acetyl-carboxy-beta-carboline amidation entry located. Microcin B17 precursor McbA in rxn58360 is a different protein sharing the name.',
           'DdaG_database_label':'rxn46045 product is labeled N3-fumaramoyl in the definition; Rhea and the actual structure correspond to fumaroyl-DAP. Preserve the raw label and flag it rather than changing ModelSEED silently.',
           'numeric_score_limits':'CphA symbolic polymer lacks a fixed molecular graph, SfaB native adenylation makes no amide, and no exact McbA native reaction is represented. Their native scores are missing, not zero. All scores are heuristic structural resemblance, not a probability or yield.'}
    (artifact('native_reaction_caveats.json')).write_text(json.dumps(notes,indent=2))
    print(pd.DataFrame(rows)[['enzyme','uniprot','length','exact_sequence_match','native_reaction_modelseed_ids','native_chemistry_score']].to_string(index=False))

if __name__=='__main__':main()
