# Experiment 2: Clinical Finding Extraction and Evidence Representation

**A source-grounded clinical finding extraction and evidence representation layer
for the pneumonia NLP branch.**

This module accepts free-text clinical descriptions (patient self-reports or
clinician notes) and returns a structured representation of the clinical findings
it detects, categorised against a small, source-grounded evidence knowledge base.

> **The module represents documented clinical findings and their relationship to a
> source-grounded pneumonia evidence profile. It does not estimate diagnostic
> probability and does not replace clinical assessment.**

---

## 1. Purpose

The pipeline performs the following steps:

```
FREE TEXT
    |
    v
Text normalisation
    |
    v
Clause segmentation
    |
    v
Deterministic KB matching (negation-aware)
    |
    v
Unmapped finding capture
    |
    v
Structured clinical evidence output
```

It does **NOT**:
- Estimate pneumonia probability
- Produce a validated clinical severity score
- Make or support a diagnostic claim
- Replace physical examination, laboratory investigation, or chest imaging

---

## 2. Clinical Sources

All KB entries are grounded in one or more of the following sources:

| Abbreviation | Full reference |
|---|---|
| CDC | Centers for Disease Control and Prevention. *Pneumonia symptoms and diagnosis.* 2024. |
| NHLBI | National Heart, Lung, and Blood Institute. *Pneumonia — Symptoms, Causes, Treatment.* |
| WHO | World Health Organization. *Pneumonia fact sheet.* 2019. |
| IDSA/ATS | Mandell LA et al. (2007). *Infectious Diseases Society of America/American Thoracic Society consensus guidelines for CAP.* Clin Infect Dis 44(Suppl 2):S27–72. |
| CURB-65 | Lim WS et al. (2003). *Defining community acquired pneumonia severity on presentation to hospital.* Thorax 58:377–382. |
| Metlay 2019 | Metlay JP et al. (2019). *Diagnosis and treatment of adults with community-acquired pneumonia.* Am J Respir Crit Care Med 200(7):e45–e67. |
| Eccles 2005 | Eccles R. (2005). *Understanding the symptoms of the common cold and influenza.* Lancet Infect Dis 5(11):718–725. |
| Monto 2000 | Monto AS et al. (2000). *Clinical signs and symptoms predicting influenza infection.* Arch Intern Med 160(21):3243–3247. |

---

## 3. Knowledge Base Structure

### 3.1 Pneumonia-Associated Supportive Findings

These are symptoms documented across CDC, NHLBI, WHO, IDSA/ATS, and Metlay et al. (2019)
as common clinical presentations of community-acquired pneumonia.

| Finding | Canonical Name | Source |
|---|---|---|
| `cough` | Acute Cough | CDC / NHLBI / WHO / Metlay 2019 |
| `fever` | Fever / Chills | CDC / NHLBI / IDSA/ATS |
| `chest_pain` | Pleuritic Chest Pain | CDC / NHLBI / IDSA/ATS |
| `sputum` | Sputum / Respiratory Secretions | CDC / NHLBI / Metlay 2019 |

**Rationale for each:**
- **Cough**: Alveolar inflammation and exudate accumulation trigger the cough reflex.
  Productive cough (yellow/green sputum) reflects bacterial consolidation.
- **Fever / Chills**: Systemic host immune response mediated by pyrogenic cytokines
  (IL-1, IL-6, TNF-α). Rigors/shaking chills reflect bacteraemia (IDSA/ATS).
- **Pleuritic Chest Pain**: Inflammation spreading from alveoli to visceral and parietal
  pleura causes friction during respiratory expansion. Worsens with inspiration.
- **Sputum**: Purulent sputum reflects alveolar exudate. Documented in NHLBI and Metlay 2019.
  *Not diagnostic alone.*

### 3.2 Concerning / Respiratory Severity Findings

These represent physiological signs of respiratory compromise or systemic severity,
documented in WHO severity criteria, CURB-65, and IDSA/ATS severe CAP criteria.

| Finding | Canonical Name | Source |
|---|---|---|
| `dyspnea` | Dyspnea | NHLBI / WHO / IDSA/ATS / Metlay 2019 |
| `tachypnea` | Tachypnea | WHO / IDSA/ATS / CURB-65 |
| `altered_mental_status` | Altered Mental Status | NHLBI / CURB-65 / IDSA/ATS |

**Rationale for each:**
- **Dyspnea**: V/Q mismatch from fluid-filled alveoli causes subjective breathlessness.
  A key severity indicator in WHO, NHLBI, and IDSA/ATS guidelines.
- **Tachypnea**: Compensatory reflex to hypoxemia. RR ≥ 30 is a CURB-65 severity
  criterion. A key WHO severity marker.
- **Altered Mental Status**: CURB-65 criterion (1 point). Reflects systemic sepsis or
  cerebral hypoxia. Particularly important in elderly patients who may lack typical fever.

### 3.3 Upper-Respiratory Findings (Contextual/Overlap)

Clinically recognised features of upper respiratory tract infections (rhinitis,
pharyngitis). Their purpose is to distinguish a purely pneumonia-associated cluster
from one that also contains URTI features.

**They are NOT negative evidence against pneumonia.**

| Finding | Source |
|---|---|
| Runny Nose | Eccles (2005) |
| Nasal Congestion | Eccles (2005) |
| Sneezing | Eccles (2005) |
| Sore Throat | Eccles (2005) / General guidelines |

### 3.4 Influenza-Like / Overlapping Systemic Findings

Systemic features common across influenza, severe viral URTI, and early pneumonia.
They are **not** pneumonia-specific. Their presence contextualises the symptom cluster.

**They do NOT create a "flu probability."**

| Finding | Source |
|---|---|
| Body Aches / Myalgia | Eccles (2005) / Monto et al. (2000) |
| Headache | Eccles (2005) / Monto et al. (2000) |
| Marked Fatigue | Eccles (2005) / Monto et al. (2000) |

### 3.5 Non-Specific / Contextual Findings

General constitutional symptoms with no discriminatory value for pneumonia.
Included to capture commonly reported patient language without assigning
pneumonia-specific significance.

| Finding | Source |
|---|---|
| Weakness / Malaise | General clinical practice |
| Loss of Appetite | General clinical practice |

---

## 4. Extraction Approach

### 4.1 Deterministic Matching (Primary Path)

The primary extraction path is fully deterministic:

1. Text is lowercased and whitespace-normalised.
2. Text is split into clauses at sentence/clause boundaries (`.`, `,`, `;`,
   `but`, `however`, `and`).
3. For each clause, each KB finding's textual variations are tested against
   the clause using regex word-boundary matching. **Longest variation first**
   prevents partial-match shadowing.
4. Negation is checked positionally (see §5).
5. Unmapped symptom phrases (pain complaints, common symptom nouns not in KB)
   are captured separately.

### 4.2 Optional BioBERT Semantic Fallback

BioBERT (`dmis-lab/biobert-base-cased-v1.1`) is available as an optional,
strictly constrained semantic similarity fallback (`--biobert` flag).

**Constraints:**
- It **cannot** override a deterministic result. The deterministic path is authoritative.
- It only activates for findings still marked `not_documented` after the deterministic pass.
- A **semantic anchor guard** must pass before any cosine similarity is computed —
  the clause must contain at least one root word semantically associated with the
  target finding. This prevents `"leg pain"` from matching `"chest pain"` via the
  shared word *pain*.
- Default similarity threshold: 0.85.

After expanding the deterministic KB with natural-language variations, the BioBERT
path adds limited marginal value and is OFF by default.

---

## 5. Negation Handling

The engine distinguishes three states for each finding:

| State | Meaning |
|---|---|
| `present` | The finding was explicitly mentioned as present |
| `absent` | The finding was explicitly negated |
| `not_documented` | The finding was not mentioned |

`not_documented` ≠ `absent`. An absent finding requires explicit negation.

**Negation detection** is positional: the engine checks the 3-word window
immediately *before* the matched span within the clause. Negation terms include:
`no`, `not`, `denies`, `without`, `free of`, `negative for`, `absent`,
`ruled out`, `never`, `none`.

**Examples:**

| Input | Result |
|---|---|
| "I have cough but no fever." | cough=present, fever=absent |
| "No cough and no shortness of breath." | cough=absent, dyspnea=absent |
| "She denies fever." | fever=absent |
| "Without difficulty breathing." | dyspnea=absent |
| "I have no difficulty breathing but I have a cough." | dyspnea=absent, cough=present |

---

## 6. Normalisation

Different phrases referring to the same finding map to one canonical concept
before evidence categorisation:

| Patient phrase | Canonical finding |
|---|---|
| "difficulty to breathe" | Dyspnea |
| "can't catch my breath" | Dyspnea |
| "breathing rapidly" | Tachypnea |
| "dry hacking cough" | Acute Cough |
| "feeling feverish" | Fever / Chills |
| "shaking chills" | Fever / Chills |
| "phlegm" | Sputum / Respiratory Secretions |
| "coughing up mucus" | Sputum / Respiratory Secretions |
| "blocked nose" | Nasal Congestion |
| "throat hurts" | Sore Throat |
| "body aches" | Body Aches / Myalgia |
| "exhausted" | Marked Fatigue |

---

## 7. Structured Output

The engine returns a Python dictionary with the following schema:

```python
{
    "input_text": str,             # Original input text

    "findings": {                  # Per-finding status
        "cough": "present" | "absent" | "not_documented",
        "fever": ...,
        "chest_pain": ...,
        "sputum": ...,
        "dyspnea": ...,
        "tachypnea": ...,
        "altered_mental_status": ...,
        "runny_nose": ...,
        "nasal_congestion": ...,
        "sneezing": ...,
        "sore_throat": ...,
        "body_aches": ...,
        "headache": ...,
        "fatigue": ...,
        "weakness": ...,
        "loss_of_appetite": ...
    },

    "supportive_findings": list,           # Present supportive findings
    "concerning_findings": list,           # Present concerning findings
    "upper_respiratory_findings": list,    # Present upper-resp findings
    "influenza_like_findings": list,       # Present flu-like findings
    "nonspecific_findings": list,          # Present nonspecific findings
    "unmapped_findings": list,             # Phrases not mapped to any KB category

    "summary": {
        "supportive_count": int,
        "concerning_count": int,
        "upper_respiratory_count": int,
        "influenza_like_count": int,
        "nonspecific_count": int
    },

    "interpretation": str,    # Qualitative summary — transparent count-based language
    "disclaimer": str         # Mandatory disclaimer
}
```

---

## 8. Why Arbitrary Probability Scoring Is NOT Used

The previous implementation used thresholds such as:
- `S >= 2 AND C >= 1 → HIGH PNEUMONIA EVIDENCE`
- `C >= 1 → URGENT/SEVERE`

These thresholds were **not derived from any cited clinical scoring system**.
CURB-65, PSI, and similar validated tools use structured clinical variables
(respiratory rate measured in breaths/min, BUN level, blood pressure, age, etc.)
and were validated on large patient populations.

A text-extraction module that counts symptom mentions cannot replicate this
without those structured variables. Presenting an uncalibrated count threshold
as a clinically meaningful severity band would be methodologically indefensible.

**This module instead uses transparent counts and qualitative language:**
```
"2 pneumonia-associated findings detected (2 supportive, 0 concerning)."
```

Actual clinical risk scoring using structured variables will be addressed
separately in Module 3.

---

## 9. Unmapped Findings

If a phrase is mentioned in the text but does not match any KB category, it is
placed in `unmapped_findings`. **No medical judgment is attached.**

The engine does **not** claim that unmapped findings are "irrelevant to pneumonia"
and does **not** treat them as negative evidence.

Example:
```
Input:  "I have cough, fever and leg pain."
Output:
    supportive_findings: [Acute Cough, Fever / Chills]
    unmapped_findings:   ["leg pain"]
```

---

## 10. Limitations

- **Text-only**: Cannot detect quantitative severity (SpO2, RR in bpm, temperature).
- **No WSD**: Word-sense disambiguation is not implemented; rare homonyms may cause errors.
- **Clause-boundary negation only**: Long-range discourse negation ("she was not
  experiencing any of the above") is not handled.
- **No co-reference resolution**: "She denies it" is not handled if "it" refers
  to a symptom.
- **Not validated on clinical datasets**: The KB and extractor have not been evaluated
  against a labelled clinical NLP corpus. Sensitivity and specificity are unknown.
- **Symptom alone cannot diagnose pneumonia**: Clinical diagnosis requires physical
  examination, chest imaging, and laboratory investigation.

---

## 11. Future Work

- Integration with a dedicated clinical NER model (e.g., fine-tuned BioBERT on
  clinical notes or i2b2/n2c2 annotated corpora) as a replacement for the
  deterministic extractor.
- Long-range negation handling using a dependency parser.
- Evaluation against a labelled clinical NLP test set to quantify precision/recall.
- Module 3: structured clinical variable intake and validated risk scoring (CURB-65/PSI).

---

## 12. How to Run

### Install dependencies
```bash
pip install torch transformers
```

### CLI demo (deterministic mode)
```bash
python demo.py
python demo.py "The patient has cough, fever and difficulty to breathe."
```

### CLI demo (with optional BioBERT fallback)
```bash
python demo.py "The patient has cough, fever and difficulty to breathe." --biobert
```

### Run the test suite
```bash
python tests.py
```

---

*Experiment 2 — Pneumonia NLP Branch. Master's thesis research implementation.*
