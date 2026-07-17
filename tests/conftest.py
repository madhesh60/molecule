

def pytest_configure(config):
    """Ensure src/ is on the path for all tests."""
    import sys
    from pathlib import Path
    src = Path(__file__).parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
