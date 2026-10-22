import json
import pytest
from amide_synthesis.enzyme_mapping.uniprot_snapshot import SnapshotClient


def page(accessions, total, next_url=None, release='test_release'):
    return json.dumps({'results': [{'primaryAccession': a} for a in accessions]}), {
        'headers': {'X-Total-Results': str(total), 'X-UniProt-Release': release},
        'next_url': next_url,
    }


def test_snapshot_follows_all_pages_and_checks_total(tmp_path, monkeypatch):
    client = SnapshotClient(tmp_path)
    pages = iter([page(['P3'], 3, 'https://example.org/page2'),
                  page(['P1'], 3, 'https://example.org/page3'), page(['P2'], 3)])
    monkeypatch.setattr(client, 'get', lambda *a, **k: next(pages))
    monkeypatch.setattr('amide_synthesis.enzyme_mapping.uniprot_snapshot.time.sleep', lambda _: None)
    rows, metadata = client.proteins('query', 'rhea:12345')
    assert [r['primaryAccession'] for r in rows] == ['P1', 'P2', 'P3']
    assert metadata['total'] == 3 and metadata['complete']


@pytest.mark.parametrize('pages', [
    [page(['P1'], 2)],
    [page(['P1','P1'], 2)],
    [page(['P1'], 2, 'https://example.org/2'), page(['P2'], 2, release='changed')],
])
def test_partial_duplicate_or_mixed_release_snapshots_fail_closed(tmp_path, monkeypatch, pages):
    client = SnapshotClient(tmp_path)
    iterator = iter(pages)
    monkeypatch.setattr(client, 'get', lambda *a, **k: next(iterator))
    monkeypatch.setattr('amide_synthesis.enzyme_mapping.uniprot_snapshot.time.sleep', lambda _: None)
    with pytest.raises(ValueError):
        client.proteins('query', 'rhea:12345')


def test_successful_empty_snapshot_is_explicit(tmp_path, monkeypatch):
    client = SnapshotClient(tmp_path)
    monkeypatch.setattr(client, 'get', lambda *a, **k: page([], 0))
    rows, metadata = client.proteins('query', 'rhea:12345')
    assert rows == [] and metadata['complete'] and metadata['total'] == 0


def test_completed_audit_cannot_be_overwritten(tmp_path):
    from amide_synthesis.evaluation.top5_mapping_audit import main
    (tmp_path/'manifest.json').write_text('{"status":"complete"}')
    with pytest.raises(FileExistsError, match='immutable'):
        main(['--output', str(tmp_path)])


def test_offline_missing_snapshot_never_uses_network(tmp_path, monkeypatch):
    client = SnapshotClient(tmp_path / 'output', source=tmp_path / 'source', offline=True)
    monkeypatch.setattr(client.session, 'get', lambda *a, **k: pytest.fail('Offline replay used network'))
    with pytest.raises(FileNotFoundError, match='Offline'):
        client.get('missing.json', 'https://example.org')


def test_offline_response_checks_url_and_bytes_then_copies(tmp_path, monkeypatch):
    import hashlib
    source = tmp_path / 'source'
    source.mkdir()
    body = b'{"results":[]}'
    (source / 'response.json').write_bytes(body)
    meta = {'requested_url': 'https://example.org/?q=test', 'sha256': hashlib.sha256(body).hexdigest()}
    (source / 'response.json.meta.json').write_text(json.dumps(meta))
    client = SnapshotClient(tmp_path / 'output', source=source, offline=True)
    monkeypatch.setattr(client.session, 'get', lambda *a, **k: pytest.fail('Offline replay used network'))
    assert client.get('response.json', 'https://example.org/', {'q': 'test'})[0] == body.decode()
    assert (tmp_path / 'output/response.json').read_bytes() == body
    with pytest.raises(ValueError, match='query changed'):
        client.get('response.json', 'https://example.org/', {'q': 'different'})
    (source / 'response.json').write_text('corrupt')
    with pytest.raises(ValueError, match='bytes changed'):
        client.get('response.json', 'https://example.org/', {'q': 'test'})
