"""Single declaration of scientific phases, stage order and artifact contracts.

The ordered stages retain the established execution order. Phases explain the
scientific decisions; dependency edges can cross phases (e.g. experimental PDB
evidence supports candidate selection before a new cofold is prepared).
"""
from dataclasses import dataclass
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PHASES = {
    'reaction-search': 'Query reaction similarity and check bond chemistry',
    'enzyme-mapping': 'Resolve proteins and retain annotation evidence',
    'candidate-selection': 'Annotate, filter and choose diverse proteins',
    'structure-modeling': 'Prepare and inspect frozen structural campaigns',
    'evaluation': 'Compare evidence, assess controls and report results',
}

@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    description: str
    needs_uniprot: bool = False
    phase: str = ''
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    tracked: bool = False
    source_dependencies: tuple[str, ...] = ()

    @property
    def source(self):
        return PACKAGE.joinpath(*self.module.split('.')).with_suffix('.py')

STAGES = (
    Stage('corpus', 'reaction_search.rebuild_corpus',
          'rebuild reactions + ECFP4/DRFP',
          needs_uniprot=False, phase='reaction-search',
          inputs=(),
          outputs=('../data/modelseed_master.parquet', '../embeddings/ecfp4_drfp_master.npy', 'corpus_manifest.json'), tracked=False),
    Stage('search', 'reaction_search.encode_and_search',
          'encode queries, cosine nearest neighbours',
          needs_uniprot=False, phase='reaction-search',
          inputs=('corpus_manifest.json',),
          outputs=('queries.tsv', 'neighbours_labelled.tsv', 'neighbours_full.tsv', 'ec_vote.tsv', 'ood.json'), tracked=False),
    Stage('screen', 'reaction_search.amide_enzymes',
          'mechanism screens + substrate distance',
          needs_uniprot=False, phase='reaction-search',
          inputs=('neighbours_labelled.tsv', 'neighbours_full.tsv', 'corpus_manifest.json'),
          outputs=('amide_forming_neighbours.tsv', 'enzyme_candidates.tsv', 'screen_summary.json'), tracked=False),
    Stage('enzymes', 'enzyme_mapping.fetch_enzymes',
          'resolve UniProt enzymes (network)',
          needs_uniprot=True, phase='enzyme-mapping',
          inputs=('amide_forming_neighbours.tsv', 'ood.json'),
          outputs=('enzymes_uniprot.tsv', 'enzyme_annotations.tsv', 'enzymes.fasta', 'reactions_without_enzymes.tsv'), tracked=False),
    Stage('table', 'candidate_selection.candidate_table',
          'prioritised candidate table (+ family)',
          needs_uniprot=False, phase='candidate-selection',
          inputs=('enzymes_uniprot.tsv', 'enzyme_annotations.tsv', 'amide_forming_neighbours.tsv', 'enzymes.fasta'),
          outputs=('candidates_ranked.tsv', 'candidates_top.fasta'), tracked=True),
    Stage('families', 'enzyme_mapping.expand_families',
          'expand validated families by domain / clade (network)',
          needs_uniprot=True, phase='enzyme-mapping',
          inputs=('candidates_ranked.tsv', 'enzymes_uniprot.tsv'),
          outputs=('family_expansion.tsv', 'family_expansion_report.tsv', 'family_expansion_summary.json'), tracked=True),
    Stage('links', 'enzyme_mapping.precomputed_links',
          'ingest the external UniProt->ModelSEED link table',
          needs_uniprot=False, phase='enzyme-mapping',
          inputs=('amide_forming_neighbours.tsv',),
          outputs=('precomputed_link_candidates.tsv', 'precomputed_link_summary.json'), tracked=True),
    Stage('refs', 'enzyme_mapping.reference_actives',
          'assayed panel + literature-curated proteins (network)',
          needs_uniprot=True, phase='enzyme-mapping',
          inputs=(),
          outputs=('reference_actives.tsv', 'reference_actives_summary.json'), tracked=False),
    Stage('neighbours', 'enzyme_mapping.chemistry_neighbours',
          'borrow enzymes across chemically equivalent reactions',
          needs_uniprot=False, phase='enzyme-mapping',
          inputs=('candidates_ranked.tsv', 'precomputed_link_candidates.tsv', 'amide_forming_neighbours.tsv'),
          outputs=('chemistry_neighbour_candidates.tsv', 'chemistry_neighbour_summary.json'), tracked=True),
    Stage('names', 'enzyme_mapping.enzyme_names',
          'candidates named for the target function in UniProt (network)',
          needs_uniprot=True, phase='enzyme-mapping',
          inputs=(),
          outputs=('enzyme_name_candidates.tsv', 'enzyme_name_summary.json'), tracked=False),
    Stage('actives', 'enzyme_mapping.active_homologs',
          'candidates reached from the MEASURED actives (network)',
          needs_uniprot=True, phase='enzyme-mapping',
          inputs=(),
          outputs=('active_homolog_candidates.tsv', 'active_homolog_summary.json'), tracked=False),
    Stage('kegg', 'enzyme_mapping.kegg_orthologs',
          'resolve unresolved reactions through KEGG orthology (network)',
          needs_uniprot=True, phase='enzyme-mapping',
          inputs=('reactions_without_enzymes.tsv',),
          outputs=('kegg_ortholog_links.tsv', 'kegg_ortholog_candidates.tsv', 'kegg_ortholog_summary.json'), tracked=False),
    Stage('kinetics', 'candidate_selection.kinetics',
          'measured Km/kcat by EC from SABIO-RK (network)',
          needs_uniprot=True, phase='candidate-selection',
          inputs=(),
          outputs=('sabio_kinetics.tsv', 'kinetics_by_ec.tsv', 'kinetics_per_candidate.tsv', 'kinetics_summary.json'), tracked=False),
    Stage('ecsub', 'candidate_selection.ec_substrate',
          "substrate distance from each protein's own Rhea reaction (network)",
          needs_uniprot=True, phase='candidate-selection',
          inputs=('enzymes_uniprot.tsv', 'precomputed_link_candidates.tsv', 'family_expansion.tsv', 'enzyme_name_candidates.tsv', 'active_homolog_candidates.tsv', 'kegg_ortholog_candidates.tsv', 'reference_actives.tsv'),
          outputs=('ec_substrate_ceiling.tsv', 'ec_substrate_summary.json'), tracked=False, source_dependencies=('infrastructure/table_utils.py',)),
    Stage('master', 'candidate_selection.master_table',
          'merge every route, keeping provenance',
          needs_uniprot=False, phase='candidate-selection',
          inputs=('candidates_ranked.tsv', 'family_expansion.tsv', 'enzymes_uniprot.tsv', 'enzymes.fasta', 'precomputed_link_candidates.tsv', 'reference_actives.tsv', 'chemistry_neighbour_candidates.tsv', 'enzyme_name_candidates.tsv', 'active_homolog_candidates.tsv', 'kegg_ortholog_candidates.tsv', 'ec_substrate_ceiling.tsv'),
          outputs=('master_candidates.tsv', 'master_candidates_summary.json'), tracked=True, source_dependencies=('infrastructure/table_utils.py',)),
    Stage('enrich', 'enzyme_mapping.enrich',
          'fill annotation for bare candidates; UniParc for the panel (network)',
          needs_uniprot=True, phase='candidate-selection',
          inputs=('master_candidates.tsv',),
          outputs=('master_candidates_enriched.tsv', 'enrichment_summary.json'), tracked=True, source_dependencies=('infrastructure/table_utils.py',)),
    Stage('pdb', 'structure_modeling.pdb_evidence',
          'experimental structures per candidate (network)',
          needs_uniprot=True, phase='structure-modeling',
          inputs=('master_candidates_enriched.tsv',),
          outputs=('pdb_entry_facts.tsv', 'pdb_evidence.tsv', 'pdb_evidence_summary.json'), tracked=True),
    Stage('geometry', 'structure_modeling.adenylation_geometry',
          'adenylation geometry in cofactor-bound structures (network)',
          needs_uniprot=True, phase='structure-modeling',
          inputs=('pdb_evidence.tsv',),
          outputs=('adenylation_geometry.tsv', 'adenylation_geometry_summary.json'), tracked=True),
    Stage('ligands', 'structure_modeling.ligand_evidence',
          'classify crystallographically bound ligands (network)',
          needs_uniprot=True, phase='structure-modeling',
          inputs=('pdb_entry_facts.tsv', 'pdb_evidence.tsv', 'master_candidates_enriched.tsv'),
          outputs=('ligand_evidence.tsv', 'ligand_evidence_summary.json', 'pocket_capacity.tsv'), tracked=True),
    Stage('members', 'candidate_selection.member_similarity',
          'discriminate members within a family (tie-break only)',
          needs_uniprot=False, phase='candidate-selection',
          inputs=('candidates_ranked.tsv', 'enzymes_uniprot.tsv', 'enzymes.fasta', 'master_candidates_enriched.tsv'),
          outputs=('member_similarity.tsv', 'member_similarity_summary.json'), tracked=True),
    Stage('famval', 'candidate_selection.family_evaluation',
          "judge each family on its own seeds' chemistry (network)",
          needs_uniprot=True, phase='candidate-selection',
          inputs=('candidates_ranked.tsv', 'enzymes_uniprot.tsv', 'amide_forming_neighbours.tsv', 'master_candidates_enriched.tsv'),
          outputs=('family_evaluation.tsv',), tracked=True),
    Stage('properties', 'candidate_selection.curated_properties',
          'curated kinetics / location / domains from UniProt (network)',
          needs_uniprot=True, phase='candidate-selection',
          inputs=('master_candidates_enriched.tsv',),
          outputs=('curated_properties.tsv', 'curated_properties_summary.json'), tracked=True),
    Stage('shortlist', 'candidate_selection.shortlist',
          'bench-ready diverse shortlist + FASTA',
          needs_uniprot=False, phase='candidate-selection',
          inputs=('master_candidates_enriched.tsv', 'pdb_evidence.tsv', 'member_similarity.tsv', 'family_evaluation.tsv', 'curated_properties.tsv'),
          outputs=('shortlist.tsv', 'shortlist.fasta', 'shortlist_summary.json'), tracked=True, source_dependencies=('infrastructure/table_utils.py',)),
    Stage('models', 'structure_modeling.model_quality',
          'AlphaFold confidence + curated properties on the shortlist (network)',
          needs_uniprot=True, phase='structure-modeling',
          inputs=('shortlist.tsv', 'curated_properties.tsv', 'ec_substrate_ceiling.tsv', 'ligand_evidence.tsv'),
          outputs=('alphafold_quality.tsv', 'shortlist_annotated.tsv', 'alphafold_quality_summary.json'), tracked=True),
    Stage('converged', 'candidate_selection.converged',
          'candidates backed by several INDEPENDENT lines of evidence',
          needs_uniprot=False, phase='candidate-selection',
          inputs=('master_candidates_enriched.tsv', 'family_evaluation.tsv', 'pdb_evidence.tsv', 'ligand_evidence.tsv', 'curated_properties.tsv', 'alphafold_quality.tsv'),
          outputs=('converged_candidates.tsv', 'converged_summary.json', 'evidence_lines.tsv'), tracked=True, source_dependencies=('infrastructure/table_utils.py',)),
    Stage('dossier', 'evaluation.dossier',
          'full dossier per converged candidate (network)',
          needs_uniprot=True, phase='evaluation',
          inputs=('converged_candidates.tsv', 'evidence_lines.tsv', 'master_candidates_enriched.tsv', 'shortlist_annotated.tsv', 'pocket_capacity.tsv', 'alphafold_quality.tsv', 'curated_properties.tsv', 'ligand_evidence.tsv', 'ec_substrate_ceiling.tsv', 'pdb_evidence.tsv', 'adenylation_geometry.tsv', 'kinetics_by_ec.tsv', 'anl_similarity_to_actives.tsv'),
          outputs=('CANDIDATES.md',), tracked=True),
    Stage('native', 'evaluation.native_substrates',
          'native reactions + family grouping',
          needs_uniprot=False, phase='evaluation',
          inputs=('candidates_ranked.tsv', 'enzyme_annotations.tsv'),
          outputs=('native_substrates.tsv', 'enzyme_families.tsv'), tracked=True),
    Stage('deckfigs', 'evaluation.deck_figures',
          'family and substrate-gap figures',
          needs_uniprot=False, phase='evaluation',
          inputs=('enzyme_families.tsv', 'native_substrates.tsv'),
          outputs=('../figures/fig_families.png', '../figures/fig_substrate_gap.png'), tracked=True),
    Stage('figure', 'evaluation.project_tsne',
          't-SNE map with candidates overlaid',
          needs_uniprot=False, phase='evaluation',
          inputs=('candidates_ranked.tsv', 'queries.tsv', 'amide_forming_neighbours.tsv', 'corpus_manifest.json'),
          outputs=('../figures/fig_tsne_nylon_query.png', 'query_tsne_coords.tsv'), tracked=True),
 )
BY_NAME = {stage.name: stage for stage in STAGES}
# Compatibility views are derived here; dependencies are not maintained twice.
RULES = {stage.name: (list(stage.inputs), list(stage.outputs))
         for stage in STAGES if stage.tracked}
MODULES = {stage.name: stage.module for stage in STAGES if stage.tracked}

def producers():
    return {output: stage.name for stage in STAGES for output in stage.outputs}

def dependencies(name):
    owners = producers()
    return tuple(dict.fromkeys(owners[path] for path in BY_NAME[name].inputs
                               if path in owners and owners[path] != name))
