"""Lazy import aliases for pre-refactor module paths; one module identity.

No compatibility implementations are kept. Canonical scientific modules live in
phase packages; this adapter exists for frozen run scripts and older imports.
"""
import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {'active_homologs': 'enzyme_mapping.active_homologs',
 'acylation': 'reaction_search.acylation',
 'adenylation_geometry': 'structure_modeling.adenylation_geometry',
 'amide_enzymes': 'reaction_search.amide_enzymes',
 'candidate_table': 'candidate_selection.candidate_table',
 'chemistry_neighbours': 'enzyme_mapping.chemistry_neighbours',
 'converged': 'candidate_selection.converged',
 'corpus': 'reaction_search.corpus',
 'curated_properties': 'candidate_selection.curated_properties',
 'deck_figures': 'evaluation.deck_figures',
 'dossier': 'evaluation.dossier',
 'ec_substrate': 'candidate_selection.ec_substrate',
 'encode_and_search': 'reaction_search.encode_and_search',
 'enrich': 'enzyme_mapping.enrich',
 'enzyme_names': 'enzyme_mapping.enzyme_names',
 'enzyme_resolution': 'enzyme_mapping.enzyme_resolution',
 'expand_families': 'enzyme_mapping.expand_families',
 'families': 'infrastructure.families',
 'family_evaluation': 'candidate_selection.family_evaluation',
 'fetch_enzymes': 'enzyme_mapping.fetch_enzymes',
 'freshness': 'infrastructure.freshness',
 'http': 'infrastructure.http',
 'kegg_orthologs': 'enzyme_mapping.kegg_orthologs',
 'kinetics': 'candidate_selection.kinetics',
 'ligand_evidence': 'structure_modeling.ligand_evidence',
 'mapping_quality': 'enzyme_mapping.mapping_quality',
 'master_table': 'candidate_selection.master_table',
 'member_similarity': 'candidate_selection.member_similarity',
 'metanetx': 'enzyme_mapping.metanetx',
 'model_quality': 'structure_modeling.model_quality',
 'native_substrates': 'evaluation.native_substrates',
 'normalize': 'reaction_search.normalize',
 'paths': 'infrastructure.paths',
 'pdb_evidence': 'structure_modeling.pdb_evidence',
 'precomputed_links': 'enzyme_mapping.precomputed_links',
 'project_tsne': 'evaluation.project_tsne',
 'rebuild_corpus': 'reaction_search.rebuild_corpus',
 'reference_actives': 'enzyme_mapping.reference_actives',
 'shortlist': 'candidate_selection.shortlist',
 'substrate_distance': 'reaction_search.substrate_distance',
 'uniparc': 'infrastructure.uniparc',
 'uniprot': 'infrastructure.uniprot',
 'workers': 'infrastructure.workers'}

class AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = 'amide_synthesis.' + target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname):
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)

    def is_package(self, fullname):
        return False

class AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix = 'amide_synthesis.'
        if fullname.startswith(prefix):
            name = fullname[len(prefix):]
            if name in ALIASES:
                return importlib.util.spec_from_loader(fullname, AliasLoader(ALIASES[name]))
        return None

def install():
    if not any(isinstance(finder, AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, AliasFinder())
