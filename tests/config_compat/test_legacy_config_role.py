from pathlib import Path


def test_config_module_declares_compatibility_role() -> None:
    text = Path('macrocast/config.py').read_text()
    assert 'compatibility layer' in text.lower()
    assert 'not the target long-run canonical package grammar' in text.lower()
