from types import SimpleNamespace
from amide_synthesis.enzyme_mapping.enzyme_resolution import resolve_reaction


def test_online_resolver_uses_validated_equivalent_aliases_without_silent_id_cap():
    calls = []
    def search(query, **kwargs):
        calls.append(query)
        return [], False, False
    hit = SimpleNamespace(rxn_id='rxn47618', ec='nan', definition='')
    aliases = {'rxn47618': {'Rhea_equivalent': {str(n) for n in range(59124, 59132)}}}
    result = resolve_reaction(hit, aliases, {}, search_client=search)
    assert len(calls) == 1
    assert all('rhea:' + str(n) in calls[0] for n in range(59124, 59132))
    assert result.status == 'no_hits'
