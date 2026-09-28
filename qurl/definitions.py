import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = str(Path(ROOT_DIR).parent)


def results_dir() -> Path:
    """Where runs are saved: $QURL_RESULTS_DIR if set, else ./results.

    Always resolved to an absolute path (relative to the cwd if
    QURL_RESULTS_DIR is itself relative), since Orbax requires an
    absolute checkpoint path at save time.
    """
    return Path(os.environ.get('QURL_RESULTS_DIR', 'results')).expanduser().resolve()
