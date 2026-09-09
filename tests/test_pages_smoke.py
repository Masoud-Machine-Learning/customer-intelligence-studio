"""Runtime smoke tests for every page.

These execute the actual Streamlit scripts with `AppTest`, which is the only way to
catch errors that live in the page layer - a mistyped column, a chart handed the wrong
frame, a widget key collision - rather than in the analytics functions the other tests
cover.

Each page must render without raising and must produce visible content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "app.py"
PAGES = sorted((PROJECT_ROOT / "pages").glob("*.py"))

# Pages run the full analytics stack against the demo dataset, so allow generous time
# for the first run, which also builds the database.
TIMEOUT_SECONDS = 180


def _run(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT_SECONDS)
    app.run()
    return app


def test_every_page_file_is_discovered():
    assert APP.exists()
    assert len(PAGES) == 8, [p.name for p in PAGES]


def test_landing_page_renders_without_error():
    app = _run(APP)

    assert not app.exception, [str(e) for e in app.exception]
    assert app.markdown, "the landing page rendered no content"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_error(page: Path):
    app = _run(page)

    assert not app.exception, f"{page.name} raised: {[str(e) for e in app.exception]}"
    assert app.markdown or app.dataframe, f"{page.name} rendered no content"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_reports_no_streamlit_error_widget(page: Path):
    """`st.error` output means the page handled a failure rather than working."""
    app = _run(page)

    unexpected = [e.value for e in app.error]
    assert not unexpected, f"{page.name} showed an error: {unexpected}"
