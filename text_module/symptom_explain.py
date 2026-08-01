SYMPTOM_ALIASES = [
    {"name": "Fever", "base": "fever", "aliases": ("fever", "febrile", "high temperature")},
    {"name": "Chills", "base": "chills", "aliases": ("chills", "shivering", "rigors")},
    {"name": "Persistent cough", "base": "cough", "aliases": ("persistent cough", "chronic cough", "persistent coughing")},
    {"name": "Productive cough", "base": "cough", "aliases": ("productive cough", "cough with phlegm", "cough with sputum", "coughing up phlegm")},
    {"name": "Dry cough", "base": "cough", "aliases": ("dry cough",)},
    {"name": "Cough", "base": "cough", "aliases": ("cough", "coughing")},
    {"name": "Shortness of breath", "base": "sob", "aliases": ("shortness of breath", "breathlessness", "dyspnea", "difficulty breathing", "trouble breathing")},
    {"name": "Chest pain", "base": "chestpain", "aliases": ("chest pain", "chest discomfort", "chest tightness")},
    {"name": "Wheezing", "base": "wheezing", "aliases": ("wheezing", "wheeze")},
    {"name": "Fatigue", "base": "fatigue", "aliases": ("fatigue", "tiredness", "lethargy", "weakness")},
    {"name": "Weight loss", "base": "weightloss", "aliases": ("weight loss", "losing weight")},
    {"name": "Night sweats", "base": "nightsweats", "aliases": ("night sweats", "night sweating")},
    {"name": "Hemoptysis", "base": "hemoptysis", "aliases": ("hemoptysis", "coughing up blood", "coughing blood")},
    {"name": "Dizziness", "base": "dizziness", "aliases": ("dizziness", "lightheaded", "lightheadedness")},
    {"name": "Swelling", "base": "swelling", "aliases": ("swelling", "edema", "oedema")},
    {"name": "Acid reflux", "base": "reflux", "aliases": ("acid reflux", "heartburn", "gastroesophageal reflux")},
    {"name": "Difficulty swallowing", "base": "dysphagia", "aliases": ("difficulty swallowing", "dysphagia", "trouble swallowing")},
]

DISEASE_SYMPTOMS = {
    "Atelectasis": {"Cough", "Shortness of breath", "Chest pain"},
    "Emphysema": {"Persistent cough", "Cough", "Shortness of breath", "Wheezing", "Fatigue"},
    "Hiatal Hernia": {"Chest pain", "Acid reflux", "Difficulty swallowing"},
    "Pleural Effusion": {"Shortness of breath", "Chest pain", "Cough"},
    "Pneumonia": {"Fever", "Chills", "Persistent cough", "Productive cough", "Cough", "Shortness of breath", "Chest pain", "Fatigue", "Night sweats"},
    "Pneumothorax": {"Chest pain", "Shortness of breath", "Dizziness"},
    "Pulmonary Congestion": {"Cough", "Shortness of breath", "Wheezing", "Fatigue", "Swelling"},
    "Pulmonary Fibrosis": {"Persistent cough", "Cough", "Shortness of breath", "Fatigue", "Weight loss"},
}


def extract_symptoms(text: str) -> list:
    text = text.lower()
    found = []
    covered = set()
    for symptom in SYMPTOM_ALIASES:
        if symptom["base"] in covered:
            continue
        if any(alias in text for alias in symptom["aliases"]):
            found.append(symptom["name"])
            covered.add(symptom["base"])
    return found


def explain_symptoms(text: str, predicted: str) -> dict:
    detected = extract_symptoms(text)
    associated = [s for s in detected if s in DISEASE_SYMPTOMS.get(predicted, set())]
    return {
        "predicted": predicted,
        "detected": detected,
        "associated": associated,
    }
