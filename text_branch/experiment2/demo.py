# demo.py
"""
CLI demonstration for the Experiment 2 clinical finding extraction pipeline.

Accepts a free-text patient or clinician description and prints a structured
human-readable evidence representation.

Usage:
    python demo.py
    python demo.py "I have cough, fever and difficulty to breathe."
    python demo.py "..." --biobert
"""

import sys
from evidence_engine import analyze_symptoms


def print_result(res: dict):
    W = 62
    print("=" * W)
    print("   CLINICAL FINDING EXTRACTION — EXPERIMENT 2 OUTPUT   ")
    print("=" * W)
    print(f"Input: \"{res['input_text'].strip()}\"\n")

    # --- Pneumonia-associated findings ---
    print("PNEUMONIA-ASSOCIATED FINDINGS")
    print("  Supportive:")
    if res["supportive_findings"]:
        for item in res["supportive_findings"]:
            print(f"    [+] {item}")
    else:
        print("      None detected")

    print("  Concerning (respiratory severity indicators):")
    if res["concerning_findings"]:
        for item in res["concerning_findings"]:
            print(f"    [!] {item}")
    else:
        print("      None detected")

    # --- Upper-respiratory overlap ---
    print("\nUPPER-RESPIRATORY FINDINGS (overlap / contextual):")
    if res["upper_respiratory_findings"]:
        for item in res["upper_respiratory_findings"]:
            print(f"    [-] {item}")
    else:
        print("    None detected")

    # --- Influenza-like overlap ---
    print("\nINFLUENZA-LIKE / SYSTEMIC OVERLAP FINDINGS:")
    if res["influenza_like_findings"]:
        for item in res["influenza_like_findings"]:
            print(f"    [-] {item}")
    else:
        print("    None detected")

    # --- Non-specific ---
    if res["nonspecific_findings"]:
        print("\nNON-SPECIFIC / CONTEXTUAL FINDINGS:")
        for item in res["nonspecific_findings"]:
            print(f"    [-] {item}")

    # --- Unmapped ---
    print("\nOTHER / UNMAPPED FINDINGS:")
    if res["unmapped_findings"]:
        for item in res["unmapped_findings"]:
            print(f"    [?] {item}  (not mapped to KB -- no pneumonia significance assigned)")
    else:
        print("    None")

    # --- Summary ---
    s = res["summary"]
    print("\nSUMMARY:")
    print(f"    Supportive findings      : {s['supportive_count']}")
    print(f"    Concerning findings      : {s['concerning_count']}")
    print(f"    Upper-respiratory        : {s['upper_respiratory_count']}")
    print(f"    Influenza-like / overlap : {s['influenza_like_count']}")

    # --- Interpretation ---
    print(f"\nINTERPRETATION:\n    {res['interpretation']}")

    # --- Disclaimer ---
    print("-" * W)
    print("IMPORTANT:")
    print(f"    {res['disclaimer']}")
    print("=" * W)


def main():
    use_biobert = "--biobert" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    print(f"Clinical Finding Extraction Engine  (biobert={'on' if use_biobert else 'off'})\n")

    if args:
        text = args[0]
    else:
        text = "The patient has cough, fever, leg pain and difficulty to breathe."
        print(f"No input provided. Running default example:\n  \"{text}\"\n")

    res = analyze_symptoms(text, use_biobert=use_biobert)
    print_result(res)


if __name__ == "__main__":
    main()
