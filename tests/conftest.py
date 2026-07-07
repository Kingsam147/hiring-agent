import importlib.util
from pathlib import Path

import pytest

import evaluator

FAKE_REFLOW_RESUME_PATH = Path(__file__).parent / "fixtures" / "fake_reflow_resume.py"


@pytest.fixture(autouse=True)
def disable_llm_disk_cache(monkeypatch):
    """Prevent unit/e2e tests from reading or writing the real cache/ directory.

    evaluator.py imports DEVELOPMENT_MODE by value at module load time, so it
    must be patched on the evaluator module itself, not on config.
    """
    monkeypatch.setattr(evaluator, "DEVELOPMENT_MODE", False)


@pytest.fixture
def fake_reflow_module():
    """A fictitious stand-in for the real, gitignored reflow_resume.py.

    Same LAYOUT/RENDER ENGINE code as the real generator, fabricated CONTENT.
    Used so tests never depend on a real person's resume being present.
    """
    spec = importlib.util.spec_from_file_location(
        "fake_reflow_resume", FAKE_REFLOW_RESUME_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
