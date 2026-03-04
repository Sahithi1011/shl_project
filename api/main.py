from flask import Flask, request, jsonify
import pandas as pd
import json
import os
from llm.gemini_parser import extract_skills 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Load Scraped Data (427 URLs)
df = pd.read_csv("shl_assessments_clean.csv")
df = df.fillna('unknown')

# Vectorizer - Metadata base chesi logic set cheyadam
df['metadata'] = df['assessment_name'] + " " + df['description'] + " " + df['test_type']
vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
tfidf_matrix = vectorizer.fit_transform(df['metadata'])

# ENDPOINTS

# 1. Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# 2. Recommendation Endpoint (Balanced Logic Update)
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # LLM based skill extraction
        skills_json = extract_skills(query)
        skills = json.loads(skills_json)
        
        combined_query = " ".join(skills.get("technical_skills", []) + skills.get("behavioral_skills", []))
        if not combined_query.strip():
            combined_query = query

        # Cosine Similarity search
        query_vector = vectorizer.transform([combined_query])
        similarity = cosine_similarity(query_vector, tfidf_matrix)
        indices = similarity.argsort()[0][::-1] # Total listed score based on sort
        
        #  BALANCING LOGIC START 
        top_tech = []
        top_behavioral = []
        
        for idx in indices:
            row = df.iloc[idx]
            test_type = str(row["test_type"]).upper()
            
            # 'P' unte Personality/Behavioral (Soft Skills)
            if "P" in test_type:
                if len(top_behavioral) < 4: # Max 4 behavioral results
                    top_behavioral.append(row)
            else:
                if len(top_tech) < 6: # Max 6 technical results
                    top_tech.append(row)
            
            # Total 10 rows select ayithe loop aapu
            if len(top_tech) + len(top_behavioral) >= 10:
                break
        
        final_results = top_tech + top_behavioral
        # --- BALANCING LOGIC END ---

        recommendations = []
        for row in final_results:
            recommendations.append({
                "url": row["url"],
                "name": row["assessment_name"],
                "adaptive_support": "No",
                "description": row["description"][:200],
                "duration": 15,
                "remote_support": "Yes",
                "test_type": [row["test_type"]]
            })
            
        return jsonify({"recommended_assessments": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)