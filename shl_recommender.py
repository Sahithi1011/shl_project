import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load cleaned data
df = pd.read_csv("shl_assessments_clean.csv")

df["combined"] = df["assessment_name"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

def recommend(name, top_n=5):

    # Fuzzy match input
    match, score, _ = process.extractOne(name, df["assessment_name"])

    if score < 50:
        print("No close match found.")
        return

    print(f"\nUsing closest match: {match}")

    index = df[df["assessment_name"] == match].index[0]

    similarity = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    indices = similarity.argsort()[::-1][1:top_n+1]

    print("\nTop 5 Recommendations:")
    for i in indices:
        print("-", df.iloc[i]["assessment_name"])


if __name__ == "__main__":
    user_input = input("Enter an assessment name: ")
    recommend(user_input)