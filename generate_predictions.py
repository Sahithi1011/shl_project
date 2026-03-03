import pandas as pd
import json
from llm.gemini_parser import extract_skills
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    return df.iloc[indices]["assessment_name"].tolist()

queries = [
    "Java developer teamwork",
    "Cloud engineer AWS",
    "Data analyst Excel communication"
]

results = []

for q in queries:
    recs = recommend(q)
    results.append({
        "query": q,
        "top_10_recommendations": recs
    })

output_df = pd.DataFrame(results)
output_df.to_csv("test_predictions.csv", index=False)

print("Prediction file generated.")