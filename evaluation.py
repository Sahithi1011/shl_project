import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm.gemini_parser import extract_skills 

# 1. Load Data
df = pd.read_csv("shl_assessments_clean.csv")
df = df.fillna('unknown')
df['metadata'] = df['assessment_name'] + " " + df['description'] + " " + df['test_type']

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df['metadata'])

def get_recommendations(query):
    
    try:
        skills_json = extract_skills(query)
        skills = json.loads(skills_json)
        combined_query = " ".join(skills.get("technical_skills", []) + skills.get("behavioral_skills", []))
        
        query_vector = vectorizer.transform([combined_query])
        similarity = cosine_similarity(query_vector, tfidf_matrix)
        indices = similarity.argsort()[0][::-1][:10] # Top 10 results
        
        return df.iloc[indices]['url'].tolist()
    except:
        return []

def evaluate():
    # 2. Labeled data load (E.g., train.csv)
    try:
        test_df = pd.read_csv("labeled_train_set.csv") 
    except FileNotFoundError:
        print("Error: labeled_train_set.csv not Found .")
        return

    hits = 0
    total = len(test_df)

    print(f"Starting Evaluation on {total} queries...")

    for _, row in test_df.iterrows():
        query = row['Query']
        actual_url = str(row['Assessment_url']).strip()
        
        recommended_urls = get_recommendations(query)
        
        # Check if correct URL is in Top 10
        if any(actual_url in url for url in recommended_urls):
            hits += 1

    recall_at_10 = (hits / total) * 100
    print(f"\n--- Evaluation Result ---")
    print(f"Mean Recall@10: {recall_at_10:.2f}%")
    print(f"--------------------------")

if __name__ == "__main__":
    evaluate()