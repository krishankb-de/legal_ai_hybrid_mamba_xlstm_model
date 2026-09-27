"""While Reference/ exists, regenerating the fixtures from it reproduces the committed ones (P1-X).

Marked ``reference``: skipped, with its reason, once Reference/ is deleted at P1-Z.
"""

import pytest

from tests.conftest import load_script


@pytest.mark.reference
@pytest.mark.slow
def test_fixtures_regenerate_bit_identically():
    assert load_script("dump_reference_parity").main(["--check"]) == 0
