from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
from llm.gemini_parser import extract_skills
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load dataset
df = pd.read_csv("shl_assessments_clean.csv")

# Load embedding model (lightweight & powerful)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute assessment embeddings
assessment_embeddings = embedding_model.encode(
    df["assessment_name"].tolist(),
    convert_to_tensor=False
)

@app.get("/health")
def health():
    return {"status": "OK"}

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend(request: QueryRequest):

    # Step 1: Extract skills
    skills_json = extract_skills(request.query)
    skills = json.loads(skills_json)

    combined_query = " ".join(
        skills["technical_skills"] + skills["behavioral_skills"]
    )

    # Step 2: Embed query
    query_embedding = embedding_model.encode([combined_query])

    # Step 3: Compute similarity
    similarity = cosine_similarity(query_embedding, assessment_embeddings)
    indices = similarity.argsort()[0][::-1][:10]

    results = []
    for i in indices:
        results.append({
            "assessment_name": df.iloc[i]["assessment_name"],
            "url": df.iloc[i]["url"]
        })

    return {
        "extracted_skills": skills,
        "recommendations": results
    }