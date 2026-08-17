# tests.py
"""
Test suite for Experiment 2 — Clinical Finding Extraction and Evidence Representation.

Tests the deterministic pipeline against all 16 required clinical cases.
Each test prints what was found and asserts correctness. Failures are reported
without hiding them — the goal is an honest evaluation.

Test cases cover:
  - Basic pneumonia-associated symptom detection
  - Negation handling (present / absent distinction)
  - Unknown/unmapped symptoms (must not become false positives)
  - Upper-respiratory overlap detection
  - Influenza-like overlap detection
  - Sputum/secretion detection
  - Critical phrase coverage ("difficulty to breathe" -> Dyspnea)
  - Completely unrelated text (no false positives)
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from evidence_engine import analyze_symptoms

# ---------------------------------------------------------------------------
# Test case definitions
# Each entry: (input_text, description, assertion_function)
# ---------------------------------------------------------------------------

def _assert(condition: bool, msg: str, failures: list):
    if not condition:
        failures.append(msg)


TEST_CASES = [

    # 1. Basic supportive symptoms
    (
        "I have fever, cough and chills.",
        "Clear pneumonia-supportive symptoms",
        lambda res, f: [
            _assert(res["findings"]["cough"] == "present",       "cough should be present", f),
            _assert(res["findings"]["fever"] == "present",       "fever should be present (chills maps to fever)", f),
            _assert(res["summary"]["supportive_count"] == 2,     "supportive_count should be 2", f),
            _assert(res["summary"]["concerning_count"] == 0,     "concerning_count should be 0", f),
        ]
    ),

    # 2. Supportive + concerning
    (
        "I have cough, fever and severe shortness of breath.",
        "Supportive + concerning finding",
        lambda res, f: [
            _assert(res["findings"]["cough"] == "present",     "cough should be present", f),
            _assert(res["findings"]["fever"] == "present",     "fever should be present", f),
            _assert(res["findings"]["dyspnea"] == "present",   "dyspnea should be present", f),
            _assert(res["summary"]["supportive_count"] == 2,   "supportive_count should be 2", f),
            _assert(res["summary"]["concerning_count"] == 1,   "concerning_count should be 1", f),
        ]
    ),

    # 3. Unmapped symptoms must NOT become false positives
    (
        "I have fever, cough, leg pain and nausea.",
        "Unmapped symptoms mixed with pneumonia symptoms",
        lambda res, f: [
            _assert(res["findings"]["cough"] == "present",     "cough should be present", f),
            _assert(res["findings"]["fever"] == "present",     "fever should be present", f),
            _assert(res["summary"]["supportive_count"] == 2,   "supportive_count should be 2", f),
            _assert(res["summary"]["concerning_count"] == 0,   "concerning_count should be 0", f),
            _assert(
                any("leg pain" in u for u in res["unmapped_findings"]),
                "leg pain should appear in unmapped_findings", f
            ),
            _assert(
                any("nausea" in u for u in res["unmapped_findings"]),
                "nausea should appear in unmapped_findings", f
            ),
        ]
    ),

    # 4. No pneumonia evidence — only unmapped pain
    (
        "I have back pain and stomach pain.",
        "No pneumonia-associated findings",
        lambda res, f: [
            _assert(res["summary"]["supportive_count"] == 0,   "supportive_count should be 0", f),
            _assert(res["summary"]["concerning_count"] == 0,   "concerning_count should be 0", f),
            _assert(
                any("back pain" in u or "stomach pain" in u for u in res["unmapped_findings"]),
                "back pain / stomach pain should appear in unmapped_findings", f
            ),
        ]
    ),

    # 5. Negation — cough present, fever absent
    (
        "I have cough but no fever.",
        "Negation: cough present, fever absent",
        lambda res, f: [
            _assert(res["findings"]["cough"] == "present",   "cough should be present", f),
            _assert(res["findings"]["fever"] == "absent",    "fever should be absent (negated)", f),
            _assert(res["summary"]["supportive_count"] == 1, "only cough counts", f),
        ]
    ),

    # 6. Multiple negations
    (
        "No cough and no shortness of breath.",
        "Multiple negations",
        lambda res, f: [
            _assert(res["findings"]["cough"] == "absent",    "cough should be absent", f),
            _assert(res["findings"]["dyspnea"] == "absent",  "dyspnea should be absent", f),
            _assert(res["summary"]["supportive_count"] == 0, "supportive_count should be 0", f),
            _assert(res["summary"]["concerning_count"] == 0, "concerning_count should be 0", f),
        ]
    ),

    # 7. Upper-respiratory findings only
    (
        "I have a sore throat and runny nose.",
        "Upper-respiratory findings",
        lambda res, f: [
            _assert(res["findings"]["sore_throat"] == "present",  "sore_throat should be present", f),
            _assert(res["findings"]["runny_nose"] == "present",   "runny_nose should be present", f),
            _assert(res["summary"]["upper_respiratory_count"] == 2, "upper_respiratory_count should be 2", f),
            _assert(res["summary"]["supportive_count"] == 0,      "no supportive findings", f),
            _assert(res["summary"]["concerning_count"] == 0,      "no concerning findings", f),
        ]
    ),

    # 8. Concerning findings — altered mental status + tachypnea
    (
        "The patient is confused and breathing rapidly.",
        "Concerning: altered mental status + tachypnea",
        lambda res, f: [
            _assert(res["findings"]["altered_mental_status"] == "present", "AMS should be present", f),
            _assert(res["findings"]["tachypnea"] == "present",             "tachypnea should be present ('breathing rapidly')", f),
            _assert(res["summary"]["concerning_count"] == 2,               "concerning_count should be 2", f),
        ]
    ),

    # 9. Mixed: cough present, fever absent, dyspnea present, leg pain unmapped
    (
        "The patient has cough, no fever, shortness of breath and leg pain.",
        "Mixed: negation + unmapped",
        lambda res, f: [
            _assert(res["findings"]["cough"] == "present",    "cough should be present", f),
            _assert(res["findings"]["fever"] == "absent",     "fever should be absent", f),
            _assert(res["findings"]["dyspnea"] == "present",  "dyspnea should be present", f),
            _assert(
                any("leg pain" in u for u in res["unmapped_findings"]),
                "leg pain should appear in unmapped_findings", f
            ),
        ]
    ),

    # 10. CRITICAL: "difficulty to breathe" must map to Dyspnea
    (
        "The patient is having difficulty to breathe.",
        "CRITICAL: 'difficulty to breathe' -> Dyspnea",
        lambda res, f: [
            _assert(res["findings"]["dyspnea"] == "present",  "dyspnea should be present", f),
            _assert(res["summary"]["concerning_count"] == 1,  "concerning_count should be 1", f),
        ]
    ),

    # 11. "can't catch my breath" -> Dyspnea; "exhausted" -> Fatigue
    (
        "I can't catch my breath and I'm feeling exhausted.",
        "Dyspnea + fatigue idioms",
        lambda res, f: [
            _assert(res["findings"]["dyspnea"] == "present",  "dyspnea should be present", f),
            _assert(res["findings"]["fatigue"] == "present",  "fatigue should be present ('exhausted')", f),
        ]
    ),

    # 12. Nasal congestion + sneezing
    (
        "My nose is blocked and I keep sneezing.",
        "Upper-respiratory: nasal congestion + sneezing",
        lambda res, f: [
            _assert(res["findings"]["nasal_congestion"] == "present", "nasal_congestion should be present", f),
            _assert(res["findings"]["sneezing"] == "present",         "sneezing should be present", f),
            _assert(res["summary"]["upper_respiratory_count"] == 2,   "upper_respiratory_count should be 2", f),
        ]
    ),

    # 13. Body aches + headache + chills -> overlapping contextual
    (
        "I have body aches, headache and chills.",
        "Influenza-like overlap + fever/chills",
        lambda res, f: [
            _assert(res["findings"]["body_aches"] == "present",  "body_aches should be present", f),
            _assert(res["findings"]["headache"] == "present",    "headache should be present", f),
            _assert(res["findings"]["fever"] == "present",       "fever should be present (chills)", f),
            _assert(res["summary"]["influenza_like_count"] == 2, "influenza_like_count should be 2", f),
            _assert(res["summary"]["supportive_count"] >= 1,     "fever/chills counts as supportive", f),
        ]
    ),

    # 14. Sputum / coughing up phlegm
    (
        "I am coughing up a lot of phlegm.",
        "Sputum detection + cough",
        lambda res, f: [
            _assert(res["findings"]["sputum"] == "present",  "sputum should be present", f),
            # 'coughing up phlegm' also contains 'coughing' — cough should be present
            _assert(res["findings"]["cough"] == "present",   "cough should be present ('coughing')", f),
        ]
    ),

    # 15. Negation: dyspnea absent, cough present
    (
        "I have no difficulty breathing but I have a cough.",
        "Negation: dyspnea absent, cough present",
        lambda res, f: [
            _assert(res["findings"]["dyspnea"] == "absent",  "dyspnea should be absent (negated)", f),
            _assert(res["findings"]["cough"] == "present",   "cough should be present", f),
        ]
    ),

    # 16. Completely unrelated text — no false positives
    (
        "The weather is very sunny today and I need to buy groceries.",
        "Completely unrelated text — no false positives",
        lambda res, f: [
            _assert(res["summary"]["supportive_count"] == 0,          "no supportive findings", f),
            _assert(res["summary"]["concerning_count"] == 0,          "no concerning findings", f),
            _assert(res["summary"]["upper_respiratory_count"] == 0,   "no upper-resp findings", f),
            _assert(res["summary"]["influenza_like_count"] == 0,      "no flu-like findings", f),
        ]
    ),
]


def run_tests(use_biobert: bool = False):
    W = 72
    print("=" * W)
    print(f"EXPERIMENT 2 — TEST SUITE  (biobert={'on' if use_biobert else 'off'})")
    print("=" * W)

    passed = 0
    failed = 0
    all_failures = []

    for idx, (text, desc, assertion_fn) in enumerate(TEST_CASES, 1):
        res = analyze_symptoms(text, use_biobert=use_biobert)
        failures = []
        assertion_fn(res, failures)

        status = "PASS" if not failures else "FAIL"
        if not failures:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {idx:02d} [{status}]: {desc}")
        print(f"  Input : \"{text}\"")
        print(f"  Supportive     : {res['supportive_findings']}")
        print(f"  Concerning     : {res['concerning_findings']}")
        print(f"  Upper-resp     : {res['upper_respiratory_findings']}")
        print(f"  Flu-like       : {res['influenza_like_findings']}")
        print(f"  Unmapped       : {res['unmapped_findings']}")
        print(f"  Interpretation : {res['interpretation']}")

        if failures:
            for msg in failures:
                print(f"  FAIL: ASSERTION FAILED: {msg}")
            all_failures.append((idx, desc, failures))

    print("\n" + "=" * W)
    print(f"RESULTS: {passed}/{len(TEST_CASES)} passed,  {failed} failed")
    if all_failures:
        print("\nFAILED TESTS:")
        for idx, desc, msgs in all_failures:
            print(f"  Test {idx:02d} - {desc}")
            for m in msgs:
                print(f"    FAIL: {m}")
    else:
        print("All tests passed.")
    print("=" * W)

    return failed == 0


if __name__ == "__main__":
    run_tests(use_biobert=False)
