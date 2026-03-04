import pandas as pd
import json
from llm.gemini_parser import extract_skills 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load 427-row dataset
df = pd.read_csv("shl_assessments_clean.csv")

df['assessment_name'] = df['assessment_name'].fillna('Unknown Assessment')
df['description'] = df['description'].fillna('Assessment for screening skills') # Default description

# 2. Vectorizer - robust configuration
vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
tfidf_matrix = vectorizer.fit_transform(df['assessment_name'])

def recommend(query):
    try:
        skills_json = extract_skills(query)
        skills = json.loads(skills_json)
        combined_query = " ".join(skills.get("technical_skills", []) + skills.get("behavioral_skills", []))
        
       
        if not combined_query.strip():
            combined_query = query

        query_vector = vectorizer.transform([combined_query])
        similarity = cosine_similarity(query_vector, tfidf_matrix)
        indices = similarity.argsort()[0][::-1][:10]
        return df.iloc[indices]
    except Exception as e:
        return pd.DataFrame()

# 3. Test Queries (Appendix 1) [cite: 141-146]
test_queries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.",
    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests.",
    "Need a Java developer who is good in collaborating with external teams and stakeholders."
]

submission_rows = []

for q in test_queries:
    recs = recommend(q)
    for _, row in recs.iterrows():
        # REQUIRED FORMAT: Query and Assessment_url [cite: 212-213]
        submission_rows.append({
            "Query": q,
            "Assessment_url": row["url"] 
        })

# 4. Save final file with the name asked
output_df = pd.DataFrame(submission_rows)
output_df.to_csv("Hema sahithi_Vasamsetti_predictions.csv", index=False)

print("SUCCESS: Hema Sahithi_Vasamsetti_predictions.Csv!")