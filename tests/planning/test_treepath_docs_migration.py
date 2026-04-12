from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_treepath_docs_migration_files_exist() -> None:
    for rel in [
        'docs/planning/treepath-docs-migration.md',
        'recipes/papers/clss2021.yaml',
        'recipes/baselines/minimal_fred_md.yaml',
    ]:
        assert (ROOT / rel).exists(), rel


def test_readme_mentions_treepath_structure() -> None:
    text = (ROOT / 'README.md').read_text()
    assert 'taxonomy/' in text
    assert 'registries/' in text
    assert 'recipes/' in text
    assert 'runs/' in text


def test_replication_docs_demote_helper_code() -> None:
    text = (ROOT / 'docs/replication/index.md').read_text()
    assert 'migration scaffolding' in text
    assert 'recipes/papers/clss2021.yaml' in text
