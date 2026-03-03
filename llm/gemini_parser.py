import spacy
import json

# Load small English model
nlp = spacy.load("en_core_web_sm")

# Simple predefined skill categories
TECH_SKILLS = [
    "python", "java", "sql", "aws", "cloud", "excel",
    "data analysis", "machine learning", "html", "css"
]

BEHAVIORAL_SKILLS = [
    "teamwork", "collaboration", "communication",
    "leadership", "problem solving", "adaptability"
]

def extract_skills(query):
    doc = nlp(query.lower())

    technical = []
    behavioral = []

    for token in doc:
        if token.text in TECH_SKILLS:
            technical.append(token.text)

        if token.text in BEHAVIORAL_SKILLS:
            behavioral.append(token.text)

    return json.dumps({
        "technical_skills": list(set(technical)),
        "behavioral_skills": list(set(behavioral))
    })