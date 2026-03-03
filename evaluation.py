import pandas as pd
import json
from llm.gemini_parser import extract_skills
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("shl_assessments_clean.csv")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["assessment_name"])

def recommend(query):
    skills_json = extract_skills(query)
    skills = json.loads(skills_json)

    combined_query = " ".join(
        skills["technical_skills"] + skills["behavioral_skills"]
    )

    query_vector = vectorizer.transform([combined_query])
    similarity = cosine_similarity(query_vector, tfidf_matrix)
    indices = similarity.argsort()[0][::-1][:10]

    return indices

# Example evaluation queries
test_queries = [
    "Java developer with teamwork",
    "Data analyst with Excel skills",
    "Python developer strong communication"
]

recall_count = 0
total = len(test_queries)

for query in test_queries:
    indices = recommend(query)
    if len(indices) > 0:
        recall_count += 1

recall_at_10 = recall_count / total

print("Recall@10:", recall_at_10)