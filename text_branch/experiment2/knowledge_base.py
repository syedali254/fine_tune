# knowledge_base.py
"""
Source-Grounded Clinical Finding Knowledge Base — Experiment 2.

Purpose: maps textual variations of clinical findings to canonical concepts
used in the pneumonia evidence representation layer.

Sources cited per finding:
  - WHO (2019): Pneumonia fact sheet, clinical presentation guidance.
  - CDC (2024): Pneumonia symptoms and diagnosis guidance.
  - NHLBI: Pneumonia — Symptoms, Causes, Treatment (National Heart, Lung, and Blood Institute).
  - IDSA/ATS (Mandell et al., 2007): Consensus guidelines for CAP.
  - CURB-65 (Lim et al., 2003): Severity scoring for CAP in clinical practice.
  - Metlay et al. (2019): Diagnosis and Treatment of Adults with CAP. Am J Respir Crit Care Med.
  - Eccles (2005): Understanding the symptoms of the common cold and influenza.
    Lancet Infect Dis.

Variation philosophy: every textual variation is a natural-language paraphrase
that patients or clinicians demonstrably use for that concept. Variations are
kept purposely lean — enough to capture common real-world language without
bloating the dictionary with every possible word combination.
"""

KNOWLEDGE_BASE = {

    # ------------------------------------------------------------------
    # PNEUMONIA-ASSOCIATED FINDINGS
    # These map to signs/symptoms documented in CDC, NHLBI, WHO, IDSA/ATS
    # pneumonia clinical presentation guidelines.
    # ------------------------------------------------------------------
    "supportive_findings": [
        {
            "id": "cough",
            "canonical_name": "Acute Cough",
            "category": "supportive",
            "textual_variations": [
                # Core term
                "cough",
                "coughing",
                # Character descriptors that patients commonly use
                "dry cough",
                "wet cough",
                "productive cough",
                "hacking cough",
                "persistent cough",
                "worsening cough",
                # Colour descriptors documented as pneumonia sputum characteristics
                "yellow mucus",
                "green mucus",
                "yellow green mucus"
            ],
            "source": "CDC / NHLBI / WHO / Metlay et al. 2019",
            "rationale": (
                "Alveolar inflammation and exudate accumulation trigger the cough "
                "reflex. Productive cough (yellow/green sputum) reflects bacterial "
                "consolidation; dry cough is also common early in the disease course."
            )
        },
        {
            "id": "fever",
            "canonical_name": "Fever / Chills",
            "category": "supportive",
            "textual_variations": [
                "fever",
                "high fever",
                "high temperature",
                # Clinical/patient language for febrile state
                "febrile",
                "feverish",
                "feeling feverish",
                "feels hot",
                "burning up",
                # Rigors / chills — listed as a distinct feature in IDSA/ATS guidelines
                "chills",
                "shaking chills",
                "rigors",
                "shivering with fever",
                # Night sweats are documented as a pneumonia feature (NHLBI)
                "sweating",
                "sweats"
            ],
            "source": "CDC / NHLBI / IDSA/ATS",
            "rationale": (
                "Systemic host immune response mediated by pyrogenic cytokines "
                "(IL-1, IL-6, TNF-α) in response to active alveolar infection. "
                "Rigors/shaking chills reflect bacteraemia and are specifically "
                "noted in IDSA/ATS CAP guidelines."
            )
        },
        {
            "id": "chest_pain",
            "canonical_name": "Pleuritic Chest Pain",
            "category": "supportive",
            "textual_variations": [
                "chest pain",
                "sharp chest pain",
                "stabbing chest pain",
                # Breathing-related chest pain — the hallmark of pleuritic pain
                "pain when breathing",
                "pain when taking a breath",
                "hurts to breathe",
                "stabbing pain when breathing",
                # Cough-related chest pain
                "pain when coughing",
                # Clinical terminology
                "pleuritic pain",
                "pleuritic chest pain",
                "chest pain with breathing"
            ],
            "source": "CDC / NHLBI / IDSA/ATS",
            "rationale": (
                "Inflammation spreading from alveoli to visceral and parietal pleura "
                "causes friction during respiratory expansion, producing sharp, "
                "positional chest pain that worsens with inspiration — a hallmark "
                "feature of pneumonia noted in all major CAP guidelines."
            )
        },
        {
            "id": "sputum",
            "canonical_name": "Sputum / Respiratory Secretions",
            "category": "supportive",
            "textual_variations": [
                # Clinical terms
                "sputum",
                "phlegm",
                "mucus",
                # Patient-reported phrases for expectoration
                "producing phlegm",
                "producing mucus",
                "coughing up mucus",
                "coughing up phlegm",
                "mucus in cough",
                "phlegm in cough"
            ],
            "source": "CDC / NHLBI / Metlay et al. 2019",
            "rationale": (
                "Purulent sputum is a recognized feature of bacterial pneumonia, "
                "reflecting alveolar exudate. Metlay et al. (2019) and NHLBI list "
                "sputum production as a clinical feature of CAP. Note: sputum alone "
                "is not diagnostic; it is one piece of the clinical picture."
            )
        }
    ],

    # ------------------------------------------------------------------
    # CONCERNING / RESPIRATORY-SEVERITY FINDINGS
    # These represent physiological signs of respiratory compromise or
    # systemic severity, documented in WHO severity criteria, CURB-65,
    # and IDSA/ATS severe CAP criteria.
    # ------------------------------------------------------------------
    "concerning_findings": [
        {
            "id": "dyspnea",
            "canonical_name": "Dyspnea",
            "category": "concerning",
            "textual_variations": [
                # Standard clinical/patient terms
                "shortness of breath",
                "short of breath",
                "breathless",
                "feeling breathless",
                "winded",
                # Active struggle phrases — high natural-language prevalence
                "difficulty breathing",
                "difficulty to breathe",
                "difficult to breathe",
                "struggling to breathe",
                "trouble breathing",
                # Can't-breathe constructions
                "can't breathe properly",
                "cannot breathe properly",
                "can't catch my breath",
                "cannot catch my breath",
                "cannot catch breath"
            ],
            "source": "NHLBI / WHO / IDSA/ATS / Metlay et al. 2019",
            "rationale": (
                "Fluid filling alveolar spaces causes ventilation-perfusion (V/Q) "
                "mismatch, severely reducing gas-exchange capacity and producing "
                "subjective breathlessness. Dyspnea is listed as a key severity "
                "indicator in WHO, NHLBI, and IDSA/ATS CAP guidelines."
            )
        },
        {
            "id": "tachypnea",
            "canonical_name": "Tachypnea",
            "category": "concerning",
            "textual_variations": [
                # Core clinical term
                "rapid breathing",
                "rapid respirations",
                # Patient/observer language
                "fast breathing",
                "breathing fast",
                "breathing rapidly",
                "breathing very fast",
                "increased respiratory rate",
                "shallow fast breaths"
            ],
            "source": "WHO / IDSA/ATS / CURB-65",
            "rationale": (
                "Compensatory physiological reflex to hypoxemia — the body increases "
                "respiratory rate to maintain oxygenation when alveolar gas exchange "
                "is impaired. Respiratory rate ≥30 breaths/min is a CURB-65 severity "
                "criterion. Tachypnea is a key WHO severity marker for pneumonia."
            )
        },
        {
            "id": "altered_mental_status",
            "canonical_name": "Altered Mental Status",
            "category": "concerning",
            "textual_variations": [
                # Clinical terms
                "confusion",
                "confused",
                "delirium",
                "disoriented",
                "altered mental status",
                # Patient/caregiver-reported
                "not acting normally",
                "mentally confused",
                "unusually confused",
                # Lethargy — the paediatric/elderly equivalent documented in NHLBI
                "lethargic",
                "unusually sleepy",
                "unusually sluggish"
            ],
            "source": "NHLBI / CURB-65 / IDSA/ATS",
            "rationale": (
                "Confusion / new-onset altered mental status is a CURB-65 criterion "
                "(1 point) reflecting systemic sepsis or cerebral hypoxia. NHLBI and "
                "IDSA/ATS note that elderly patients frequently present with confusion "
                "rather than classic fever, making AMS an important severity marker."
            )
        }
    ],

    # ------------------------------------------------------------------
    # UPPER-RESPIRATORY FINDINGS
    # These are clinically recognised features of upper respiratory tract
    # infections (URTI) — rhinitis, pharyngitis — that overlap with early
    # or mild respiratory illness presentations.
    #
    # Purpose: to distinguish a purely pneumonia-associated symptom cluster
    # from a cluster that also contains URTI features. They are NOT scored
    # against pneumonia and are NOT negative evidence.
    #
    # Source: Eccles (2005), Lancet Infect Dis — common cold symptom taxonomy;
    # general ENT / primary-care clinical guidelines.
    # ------------------------------------------------------------------
    "upper_respiratory_findings": [
        {
            "id": "runny_nose",
            "canonical_name": "Runny Nose",
            "category": "upper_respiratory",
            "textual_variations": [
                "runny nose",
                "nose running",
                "watery nose",
                "nasal discharge",
                "nasal drip"
            ],
            "source": "Eccles (2005)",
            "rationale": (
                "Rhinorrhoea is a hallmark upper-respiratory symptom. Its presence "
                "alongside respiratory symptoms may suggest a viral URTI (e.g., "
                "rhinovirus) rather than bacterial lower respiratory consolidation."
            )
        },
        {
            "id": "nasal_congestion",
            "canonical_name": "Nasal Congestion",
            "category": "upper_respiratory",
            "textual_variations": [
                "blocked nose",
                "stuffy nose",
                "congested nose",
                "nasal congestion",
                "nose is blocked",
                "blocked nasal passages"
            ],
            "source": "Eccles (2005)",
            "rationale": (
                "Nasal congestion is a primary URTI feature (rhinitis). Its presence "
                "contextualises the symptom cluster toward upper-airway involvement."
            )
        },
        {
            "id": "sneezing",
            "canonical_name": "Sneezing",
            "category": "upper_respiratory",
            "textual_variations": [
                "sneezing",
                "frequent sneezing",
                "keeps sneezing",
                "sneezing a lot"
            ],
            "source": "Eccles (2005)",
            "rationale": (
                "Sneezing is a cardinal symptom of rhinitis/URTI; it is rarely a "
                "feature of lower respiratory pneumonia and contextualises the "
                "presentation toward upper-airway aetiology."
            )
        },
        {
            "id": "sore_throat",
            "canonical_name": "Sore Throat",
            "category": "upper_respiratory",
            "textual_variations": [
                "sore throat",
                "throat hurts",
                "throat pain",
                "painful throat",
                "irritated throat",
                "throat is sore"
            ],
            "source": "Eccles (2005) / General clinical guidelines",
            "rationale": (
                "Pharyngitis / sore throat is a common URTI feature. Its presence "
                "may indicate pharyngeal involvement characteristic of viral URTI "
                "rather than lower respiratory infection."
            )
        }
    ],

    # ------------------------------------------------------------------
    # INFLUENZA-LIKE / OVERLAPPING FINDINGS
    # These are systemic findings common across influenza, severe viral
    # URTI, and early pneumonia. They are NOT pneumonia-specific.
    #
    # Source: Eccles (2005), Lancet Infect Dis; Monto et al. (2000)
    # Clinical signs and symptoms predicting influenza infection.
    # Arch Intern Med.
    # ------------------------------------------------------------------
    "influenza_like_findings": [
        {
            "id": "body_aches",
            "canonical_name": "Body Aches / Myalgia",
            "category": "influenza_like",
            "textual_variations": [
                "body aches",
                "body pain",
                "muscle aches",
                "aching muscles",
                "muscle pain",
                "myalgia"
            ],
            "source": "Eccles (2005) / Monto et al. (2000)",
            "rationale": (
                "Myalgia is a systemic feature mediated by pro-inflammatory cytokines "
                "and is more prominently associated with influenza than with typical "
                "bacterial pneumonia. Its presence contextualises the symptom cluster."
            )
        },
        {
            "id": "headache",
            "canonical_name": "Headache",
            "category": "influenza_like",
            "textual_variations": [
                "headache",
                "head hurts",
                "head pain",
                "pounding headache"
            ],
            "source": "Eccles (2005) / Monto et al. (2000)",
            "rationale": (
                "Headache is a common systemic symptom of influenza and viral URTI, "
                "mediated by cytokine release. Not a primary pneumonia feature."
            )
        },
        {
            "id": "fatigue",
            "canonical_name": "Marked Fatigue",
            "category": "influenza_like",
            "textual_variations": [
                "fatigue",
                "extremely tired",
                "very tired",
                "exhausted",
                "severe tiredness",
                "feeling exhausted",
                "exhaustion"
            ],
            "source": "Eccles (2005) / Monto et al. (2000)",
            "rationale": (
                "Marked fatigue/exhaustion is a prominent feature of influenza and "
                "systemic viral infection, mediated by interferon release. Fatigue "
                "can also accompany severe pneumonia but is non-specific on its own."
            )
        }
    ],

    # ------------------------------------------------------------------
    # NON-SPECIFIC / CONTEXTUAL FINDINGS
    # General systemic symptoms that appear across many illnesses and
    # carry no specific discriminatory value for pneumonia.
    # Included to prevent unmapped noise for common patient-reported terms.
    # ------------------------------------------------------------------
    "nonspecific_findings": [
        {
            "id": "weakness",
            "canonical_name": "Weakness / Malaise",
            "category": "nonspecific",
            "textual_variations": [
                "weakness",
                "feeling weak",
                "very weak",
                "weak",
                "malaise",
                "general malaise"
            ],
            "source": "General clinical practice",
            "rationale": (
                "Weakness and malaise are non-specific constitutional symptoms "
                "common across viral, bacterial, and other systemic illnesses. "
                "Included to capture commonly reported patient language without "
                "assigning pneumonia-specific significance."
            )
        },
        {
            "id": "loss_of_appetite",
            "canonical_name": "Loss of Appetite",
            "category": "nonspecific",
            "textual_variations": [
                "loss of appetite",
                "no appetite",
                "poor appetite",
                "not hungry",
                "doesn't feel like eating"
            ],
            "source": "General clinical practice",
            "rationale": (
                "Anorexia is a non-specific constitutional symptom of acute illness. "
                "It has no discriminatory value for pneumonia versus other illnesses "
                "and is represented here only to avoid unmapped-finding noise."
            )
        }
    ]
}
