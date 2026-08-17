"""
Thin wrapper for the Experiment 2 clinical finding extraction engine.

Exposes a single function used by the Streamlit demo. Runs the deterministic
(primary) extraction path — the optional BioBERT semantic fallback stays off,
matching the default CLI behaviour.

The engine patches sys.modules at import time to block tensorflow. That side
effect breaks transformers (>=4.51) lazy BERT imports, so it is isolated here
and undone immediately so the image/text ML branches are unaffected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "text_branch" / "experiment2"))

_tf_sentinel = object()
_tf_before = sys.modules.get("tensorflow", _tf_sentinel)

from evidence_engine import analyze_symptoms  # noqa: E402

if _tf_before is _tf_sentinel:
    sys.modules.pop("tensorflow", None)
else:
    sys.modules["tensorflow"] = _tf_before


def extract_evidence(text: str) -> dict:
    """Run Experiment 2 evidence extraction on symptom text."""
    return analyze_symptoms(text, use_biobert=False)