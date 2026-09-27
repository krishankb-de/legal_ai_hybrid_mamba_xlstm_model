"""The built distributions contain the package and nothing else (plan P1-A, CI `package` job).

The CI job proves the wheel imports outside the checkout; this test pins what goes into the
archives, so a stray include (an unanchored `README.md` once pulled in `Reference/**/README.md`)
fails locally before it ever reaches a push.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile

import pytest

from tests.conftest import REPO_ROOT

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH (the environment is uv-managed)"),
]


@pytest.fixture(scope="module")
def dists(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--out-dir", str(out)], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    wheels, sdists = sorted(out.glob("lexhybrid-*.whl")), sorted(out.glob("lexhybrid-*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, f"expected one wheel and one sdist in {out}"
    return wheels[0], sdists[0]


def test_wheel_holds_only_the_package(dists):
    wheel, _ = dists
    names = zipfile.ZipFile(wheel).namelist()
    stray = [n for n in names if not (n.startswith("lexhybrid/") or ".dist-info/" in n)]
    assert not stray, f"wheel contains files outside lexhybrid/: {stray}"
    assert "lexhybrid/models/hybrid_lm.py" in names


def test_sdist_holds_only_the_package_and_its_metadata(dists):
    _, sdist = dists
    with tarfile.open(sdist) as tar:
        names = [m.name.split("/", 1)[1] for m in tar.getmembers() if m.isfile()]
    allowed_top = {"README.md", "pyproject.toml", ".gitignore", "PKG-INFO"}
    stray = [n for n in names if not n.startswith("lexhybrid/") and n not in allowed_top]
    assert not stray, f"sdist contains unexpected files: {stray}"
