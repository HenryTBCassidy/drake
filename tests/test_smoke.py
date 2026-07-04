"""Smoke test — keeps CI green before real code lands.

Delete or absorb into a real test module once the package has substance.
"""

from __future__ import annotations

import drake


def test_package_imports() -> None:
    assert drake.__version__ == "0.1.0"
