"""Commands that follow the scientific workflow; default flags run discovery."""
import argparse
import runpy
import sys

EVALUATIONS = {
    'native-audit': 'audit_native_reactions',
    'native-similarity': 'compare_native_reactions',
    'native-draw': 'draw_native_reactions',
    'selected-reactions': 'draw_top5_native_reactions',
    'polymerization-target': 'draw_polymerization_target',
    'audit-mapping': 'audit_mapping_robustness',
    'top5-mapping': 'top5_mapping_audit',
    'top5-cofold-status': 'top5_cofold_status',
    'current-panel': 'current_panel_report',
    'compare-retrieval': 'compare_retrieval',
    'audit-columns': 'audit_columns',
    'audit-coverage': 'audit_coverage',
    'qualified-set': 'qualified_set',
    'acceptor-cohort': 'acceptor_cohort',
    'substrate-walk': 'substrate_walk',
    'signal-granularity': 'signal_granularity',
    'recall-audit': 'recall_audit',
    'annotation-ceiling': 'verify_annotation_ceiling',
    'activity-validation': 'validate_against_activity',
    'yield-validation': 'yield_validation',
    'literature-controls': 'literature_controls',
    'review-figures': 'review_figures',
    'review-report': 'review_report',
    'refresh-summary': 'refresh_summary',
    'probe-unresolved': 'probe_unresolved',
}

COMMAND_MODULES = {
    'rank': 'amide_synthesis.reaction_search.rank_modelseed_nylon',
    'native-go': 'amide_synthesis.enzyme_mapping.fetch_native_enzyme_go',
}


def run_module(module, args):
    """Forward utility arguments without leaking argv to the caller."""
    previous = sys.argv
    try:
        sys.argv = [module] + args
        runpy.run_module(module, run_name='__main__')
    finally:
        sys.argv = previous
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ''
    if command in ('-h', '--help'):
        parser = argparse.ArgumentParser(
            usage='python -m amide_synthesis [COMMAND] [OPTIONS]',
            description='Reaction search, enzyme mapping, candidate selection and model evaluation.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=('Commands:\n'
                    '  discover    Run discovery stages (also the default for bare flags)\n'
                    '  rank        Rank ModelSEED reactions and select the chemical panel\n'
                    '  reproduce   Replay the pinned panel offline\n'
                    '  native-go   Export GO evidence for literature reference enzymes\n'
                    '  model       Prepare, inspect or explicitly launch structural campaigns\n'
                    '  evaluate    Run a named evaluation, audit or reaction drawing\n'
                    '  describe    Describe scientific phases and stage contracts\n'
                    '  report      Report workflow and artifact status\n\n'
                    'Use COMMAND --help for its options.'))
        parser.print_help()
        return 0
    if command in COMMAND_MODULES:
        return run_module(COMMAND_MODULES[command], args[1:])
    if command == 'discover':
        from .pipeline import main as discover
        return discover(args[1:])
    if command == 'reproduce':
        from .reproduce import main as reproduce
        return reproduce(args[1:])
    if command=='describe':
        from .evaluation.workflow_report import describe
        parser = argparse.ArgumentParser(description='Explain scientific phases and canonical stage entry points')
        parser.add_argument('--format', choices=['markdown','mermaid','json'], default='markdown')
        options = parser.parse_args(args[1:])
        print(describe(options.format)); return 0
    if command=='report':
        from .evaluation.workflow_report import main as report
        return report(args[1:])
    if command=='model':
        from .structure_modeling.campaign import main as model
        return model(args[1:])
    if command=='evaluate':
        parser = argparse.ArgumentParser(description='Run a named evaluation or audit from the maintained package')
        parser.add_argument('analysis', choices=sorted(EVALUATIONS))
        # Parse only the analysis name so --help and every utility-specific
        # option reach the selected module instead of this dispatcher.
        options = parser.parse_args(args[1:2])
        remaining = args[2:]
        module = 'amide_synthesis.evaluation.'+EVALUATIONS[options.analysis]
        return run_module(module, remaining)
    from .pipeline import main as discover
    return discover(args)
