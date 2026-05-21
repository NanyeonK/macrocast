"""Skip all wizard tests when the [wizard] extra (solara) is not installed."""
import pytest

pytest.importorskip("solara")
