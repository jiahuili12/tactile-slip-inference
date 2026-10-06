from pathlib import Path


def pytest_sessionstart(session):
    # pytest creates basetemp itself, but not a missing parent directory.
    base = session.config.option.basetemp
    if base:
        Path(base).parent.mkdir(parents=True, exist_ok=True)
