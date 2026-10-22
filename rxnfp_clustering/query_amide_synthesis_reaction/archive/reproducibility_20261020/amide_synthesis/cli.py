"""Commands that follow the scientific workflow; default flags run discovery."""
import argparse
import runpy
import sys

EVALUATIONS = {
    'audit-mapping': 'audit_mapping_robustness',
    'top5-mapping': 'top5_mapping_audit',
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


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ''
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
        options, remaining = parser.parse_known_args(args[1:])
        module = 'amide_synthesis.evaluation.'+EVALUATIONS[options.analysis]
        previous = sys.argv
        try:
            sys.argv = [module]+remaining
            runpy.run_module(module, run_name='__main__')
        finally:
            sys.argv = previous
        return 0
    from .pipeline import main as discover
    return discover(args)
