# evidence_engine.py
"""
Clinical Finding Extraction and Evidence Representation Engine — Experiment 2.

Purpose:
    Accepts free-text clinical descriptions and returns a structured
    representation of extracted clinical findings mapped to the project's
    source-grounded pneumonia evidence knowledge base.

This module does NOT:
    - Estimate pneumonia probability
    - Produce a validated diagnostic score
    - Replace clinical assessment

Pipeline:
    FREE TEXT
        → Text normalisation
        → Clause segmentation
        → Deterministic KB matching (longest-variation-first)
        → Negation detection (window-based, positional)
        → Unmapped finding capture
        → Structured output assembly
    (Optional BioBERT semantic fallback — OFF by default)
"""

import os
import sys
# Fix protobuf gencode/runtime mismatch on this system
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Disable tensorflow to prevent transformers from attempting to import it
sys.modules["tensorflow"] = None

import re
from knowledge_base import KNOWLEDGE_BASE

# ---------------------------------------------------------------------------
# NEGATION TERMS
# ---------------------------------------------------------------------------
NEGATION_TERMS = {
    "no", "not", "denies", "deny", "denied", "without",
    "free of", "negative for", "none", "never", "ruled out", "absent"
}

# ---------------------------------------------------------------------------
# FLAT LIST OF ALL KB GROUPS IN DISPLAY/PROCESSING ORDER
# ---------------------------------------------------------------------------
_ALL_KB_GROUPS = [
    "supportive_findings",
    "concerning_findings",
    "upper_respiratory_findings",
    "influenza_like_findings",
    "nonspecific_findings"
]

def _all_kb_findings() -> list:
    """Returns a flat list of all findings across all KB groups."""
    result = []
    for group in _ALL_KB_GROUPS:
        result.extend(KNOWLEDGE_BASE.get(group, []))
    return result


# ---------------------------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def split_clauses(text: str) -> list:
    """
    Splits text into clauses at sentence/clause boundaries.
    Uses period, semicolon, comma, 'but', 'however', 'and' as delimiters.
    Short conjunctions like 'and' are included so that "cough and no fever"
    is split into ["cough", "no fever"], allowing correct per-clause negation.
    """
    boundaries = r'\.|;|,|\bbut\b|\bhowever\b|\band\b'
    clauses = re.split(boundaries, text)
    return [normalize_text(c) for c in clauses if c.strip()]


# ---------------------------------------------------------------------------
# NEGATION DETECTION
# ---------------------------------------------------------------------------

def check_negation(clause: str, match_start_idx: int) -> bool:
    """
    Returns True if a negation term appears in the 3-word window immediately
    before the matched span in the clause.

    Positional check: only words BEFORE the match are considered, preventing
    "fever, no cough" from incorrectly negating 'fever'.
    """
    pre_match_text = clause[:match_start_idx].strip()
    if not pre_match_text:
        return False

    words = pre_match_text.split()
    window = words[-3:] if len(words) >= 3 else words

    for i in range(len(window)):
        if window[i] in NEGATION_TERMS:
            return True
        # Two-word negation phrases (e.g. "free of", "negative for")
        if i < len(window) - 1:
            phrase = f"{window[i]} {window[i+1]}"
            if phrase in NEGATION_TERMS:
                return True

    return False


# ---------------------------------------------------------------------------
# UNMAPPED FINDING CAPTURE
# ---------------------------------------------------------------------------

# Noun phrases ending in pain/ache/aches that are NOT in the KB get captured
# as unmapped findings. This regex catches common patient pain complaints.
_PAIN_PATTERN = re.compile(
    r'\b(\w+\s+)?(?:pain|ache|aches)\b',
    re.IGNORECASE
)

# Additional common symptom nouns that patients report
_SYMPTOM_NOUNS = re.compile(
    r'\b(nausea|vomiting|diarrhoea|diarrhea|rash|itching|swelling|bloating)\b',
    re.IGNORECASE
)


def extract_unmapped_findings(text: str, mapped_spans: set) -> list:
    """
    Searches the normalised text for pain-complaint phrases and common symptom
    nouns that were NOT already matched by the KB.

    Returns a deduplicated list of unmapped phrase strings.
    No medical judgment is attached to these findings.
    """
    unmapped = []
    norm = normalize_text(text)

    for pattern in (_PAIN_PATTERN, _SYMPTOM_NOUNS):
        for m in pattern.finditer(norm):
            phrase = m.group(0).strip()
            # Skip if already covered by a KB variation match
            if phrase not in mapped_spans:
                # Check: not negated at this position
                if not check_negation(norm, m.start()):
                    unmapped.append(phrase)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for p in unmapped:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# INTERPRETATION BUILDER
# ---------------------------------------------------------------------------

def build_interpretation(
    supportive_count: int,
    concerning_count: int,
    upper_respiratory_count: int,
    influenza_like_count: int,
    nonspecific_count: int
) -> str:
    """
    Produces a qualitative, transparent interpretation string.
    Uses plain count-based language. Does NOT claim diagnostic probability.
    """
    parts = []

    pneumonia_associated = supportive_count + concerning_count

    if pneumonia_associated == 0 and upper_respiratory_count == 0 and influenza_like_count == 0:
        return (
            "No pneumonia-associated findings were identified in the provided text. "
            "Contextual or non-specific findings may still be present."
        )

    if pneumonia_associated > 0:
        parts.append(
            f"{pneumonia_associated} pneumonia-associated finding"
            f"{'s' if pneumonia_associated > 1 else ''} detected"
            f" ({supportive_count} supportive, {concerning_count} concerning)."
        )

    if concerning_count == 1:
        parts.append("A respiratory concern was identified — clinical assessment is required.")
    elif concerning_count > 1:
        parts.append(
            f"{concerning_count} concerning respiratory findings identified — "
            "clinical assessment is required."
        )

    if upper_respiratory_count > 0:
        parts.append(
            f"Upper-respiratory overlap features also detected "
            f"({upper_respiratory_count} finding{'s' if upper_respiratory_count > 1 else ''})."
        )

    if influenza_like_count > 0:
        parts.append(
            f"Influenza-like / overlapping systemic features also present "
            f"({influenza_like_count} finding{'s' if influenza_like_count > 1 else ''})."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# DETERMINISTIC ANALYSIS (PRIMARY PATH)
# ---------------------------------------------------------------------------

def analyze_symptoms_deterministic(text: str) -> dict:
    """
    Primary extraction path.

    Steps:
      1. Normalise and split into clauses.
      2. For each clause, attempt longest-variation-first matching against all KB findings.
      3. Apply positional negation detection.
      4. Capture unmapped symptom phrases (pain complaints, common symptom nouns).
      5. Assemble structured output.
    """
    norm_text = normalize_text(text)
    clauses = split_clauses(norm_text)

    all_findings = _all_kb_findings()

    # Initialise all findings to not_documented
    findings_status = {f["id"]: "not_documented" for f in all_findings}
    matched_phrases = {f["id"]: None for f in all_findings}

    # Track which text spans were matched (for unmapped detection)
    mapped_spans: set = set()

    # --- KB matching ---
    for clause in clauses:
        for finding in all_findings:
            # Longest variation first prevents partial match shadowing
            sorted_vars = sorted(finding["textual_variations"], key=len, reverse=True)
            for var in sorted_vars:
                pattern = rf'\b{re.escape(var)}\b'
                match = re.search(pattern, clause)
                if match:
                    is_negated = check_negation(clause, match.start())
                    status = "absent" if is_negated else "present"

                    # Present is sticky — once confirmed, keep it
                    if findings_status[finding["id"]] != "present":
                        findings_status[finding["id"]] = status
                        matched_phrases[finding["id"]] = var

                    # Record matched span regardless of negation status
                    mapped_spans.add(var)
                    break  # Move to next finding for this clause

    # --- Categorise by group ---
    group_results = {}
    for group in _ALL_KB_GROUPS:
        group_results[group] = [
            f["canonical_name"]
            for f in KNOWLEDGE_BASE.get(group, [])
            if findings_status[f["id"]] == "present"
        ]

    supportive   = group_results["supportive_findings"]
    concerning   = group_results["concerning_findings"]
    upper_resp   = group_results["upper_respiratory_findings"]
    flu_like     = group_results["influenza_like_findings"]
    nonspecific  = group_results["nonspecific_findings"]

    S = len(supportive)
    C = len(concerning)
    U = len(upper_resp)
    I = len(flu_like)
    N = len(nonspecific)

    # --- Unmapped findings ---
    unmapped = extract_unmapped_findings(text, mapped_spans)

    # --- Structured findings dict (all findings with their status) ---
    findings_flat = {f["id"]: findings_status[f["id"]] for f in all_findings}

    interpretation = build_interpretation(S, C, U, I, N)

    return {
        "input_text": text,

        # Per-finding status (present / absent / not_documented)
        "findings": findings_flat,

        # Categorised lists of PRESENT findings by group
        "supportive_findings": supportive,
        "concerning_findings": concerning,
        "upper_respiratory_findings": upper_resp,
        "influenza_like_findings": flu_like,
        "nonspecific_findings": nonspecific,

        # Findings mentioned in the text that did not map to any KB category
        "unmapped_findings": unmapped,

        "summary": {
            "supportive_count": S,
            "concerning_count": C,
            "upper_respiratory_count": U,
            "influenza_like_count": I,
            "nonspecific_count": N
        },

        "interpretation": interpretation,

        "disclaimer": (
            "Structured clinical evidence representation only. "
            "This does not diagnose pneumonia or provide a clinical probability estimate. "
            "Clinical confirmation is required."
        )
    }


# ---------------------------------------------------------------------------
# OPTIONAL BIOBERT SEMANTIC FALLBACK (OFF BY DEFAULT)
# ---------------------------------------------------------------------------

# BioBERT is kept as an optional, strictly constrained fallback.
# It CANNOT override a deterministic result.
# It is NOT the default extraction path.
# Semantic anchor guards prevent cross-symptom false positives.

_tokenizer = None
_model = None
_kb_embeddings: dict = {}


def load_biobert():
    """Lazily loads BioBERT model and tokenizer."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        _model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    return _tokenizer, _model


def get_embedding(text: str):
    """Computes a mean-pooled token embedding for a given text using BioBERT."""
    import torch
    tokenizer, model = load_biobert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask)[0]


def get_cosine_similarity(v1, v2) -> float:
    """Calculates cosine similarity between two PyTorch vectors."""
    import torch.nn.functional as F
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def cache_kb_embeddings():
    """Encodes all textual variations in the KB and caches their embeddings."""
    global _kb_embeddings
    if _kb_embeddings:
        return
    for finding in _all_kb_findings():
        _kb_embeddings[finding["id"]] = []
        for var in finding["textual_variations"]:
            emb = get_embedding(var)
            _kb_embeddings[finding["id"]].append((var, emb))


# Semantic anchors: each finding must have at least one anchor word present
# in the clause before BioBERT similarity is even evaluated.
# This prevents "leg pain" from matching "chest pain" via cosine proximity.
SEMANTIC_ANCHORS = {
    "cough":               ["cough", "phlegm", "mucus", "sputum", "spit", "expectora"],
    "fever":               ["fever", "temp", "chill", "sweat", "rigor", "warm", "hot", "febrile"],
    "chest_pain":          ["chest", "thorax", "pleur", "rib", "lung", "sternum"],
    "sputum":              ["phlegm", "mucus", "sputum", "spit", "expectora", "secretion"],
    "dyspnea":             ["breath", "breathless", "air", "gasp", "wind", "inhale", "exhale", "oxygen"],
    "tachypnea":           ["breath", "respirat", "pant", "rate", "fast", "rapid"],
    "altered_mental_status": ["confus", "delir", "disorient", "letharg", "mental", "alert", "fog"],
    "runny_nose":          ["nose", "nasal", "rhinorr", "drip", "discharge"],
    "nasal_congestion":    ["nose", "nasal", "sinus", "congest", "block", "stuffy"],
    "sneezing":            ["sneez"],
    "sore_throat":         ["throat", "pharyn", "tonsil"],
    "body_aches":          ["ache", "muscle", "myalg", "pain", "body"],
    "headache":            ["head", "migrain", "cephal"],
    "fatigue":             ["tired", "fatigue", "exhaust", "lethar", "energy"],
    "weakness":            ["weak", "malais", "feeble"],
    "loss_of_appetite":    ["appetite", "hungry", "eat", "food", "anorex"]
}


def verify_semantic_anchor(finding_id: str, clause: str) -> bool:
    """
    Returns True only if the clause contains at least one anchor substring
    associated with the finding. Prevents BioBERT from matching unrelated spans.
    """
    anchors = SEMANTIC_ANCHORS.get(finding_id, [])
    if not anchors:
        return True
    words = clause.lower().split()
    for word in words:
        for anchor in anchors:
            if anchor in word:
                return True
    return False


def analyze_symptoms_hybrid(text: str, similarity_threshold: float = 0.85) -> dict:
    """
    Hybrid pipeline (optional, BioBERT-assisted).

    1. Run deterministic matching (primary, authoritative).
    2. For findings still marked 'not_documented', attempt BioBERT cosine
       similarity against cached KB variation embeddings.
    3. Semantic anchor guard MUST pass before any BioBERT check.
    4. BioBERT CANNOT override an existing deterministic result.
    5. Negation is re-checked on the clause before marking BioBERT matches.
    """
    res = analyze_symptoms_deterministic(text)

    load_biobert()
    cache_kb_embeddings()

    norm_text = normalize_text(text)
    clauses = split_clauses(norm_text)

    all_findings = _all_kb_findings()
    updated = False

    for finding in all_findings:
        fid = finding["id"]
        # Deterministic result is authoritative — skip if already resolved
        if res["findings"][fid] in ("present", "absent"):
            continue

        for clause in clauses:
            if not clause.strip() or len(clause.split()) < 2:
                continue

            # Semantic anchor guard — must pass before incurring BioBERT cost
            if not verify_semantic_anchor(fid, clause):
                continue

            clause_emb = get_embedding(clause)

            best_sim = -1.0
            best_var = None
            for var_text, var_emb in _kb_embeddings.get(fid, []):
                sim = get_cosine_similarity(clause_emb, var_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_var = var_text

            if best_sim >= similarity_threshold:
                # Positional negation: check if negation appears before the
                # best-matching span position (approximate: check start of clause)
                is_negated = False
                words = clause.split()
                for i, w in enumerate(words):
                    if w in NEGATION_TERMS:
                        is_negated = True
                        break

                status = "absent" if is_negated else "present"
                res["findings"][fid] = status
                updated = True
                break  # One clause match per finding is sufficient

    if updated:
        # Rebuild group lists and summary from updated findings dict
        for group in _ALL_KB_GROUPS:
            key = group  # e.g. "supportive_findings"
            res[key] = [
                f["canonical_name"]
                for f in KNOWLEDGE_BASE.get(group, [])
                if res["findings"].get(f["id"]) == "present"
            ]

        S = len(res["supportive_findings"])
        C = len(res["concerning_findings"])
        U = len(res["upper_respiratory_findings"])
        I = len(res["influenza_like_findings"])
        N = len(res["nonspecific_findings"])

        res["summary"] = {
            "supportive_count": S,
            "concerning_count": C,
            "upper_respiratory_count": U,
            "influenza_like_count": I,
            "nonspecific_count": N
        }
        res["interpretation"] = build_interpretation(S, C, U, I, N)

    return res


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def analyze_symptoms(text: str, use_biobert: bool = False) -> dict:
    """
    Main entry point for Experiment 2 clinical finding extraction.

    Args:
        text: Free-text patient or clinician description.
        use_biobert: If True, runs the optional BioBERT semantic fallback
                     after the primary deterministic pass.

    Returns:
        Structured dictionary of extracted clinical findings.
        See module docstring for output schema.
    """
    if use_biobert:
        return analyze_symptoms_hybrid(text)
    return analyze_symptoms_deterministic(text)
