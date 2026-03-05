import google.generativeai as genai
import os

def extract_skills(query_text):
    """skill extraction using Gemini AI directly, no Spacy needed."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return query_text
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"Extract only the key technical skills and job roles from this text as a comma-separated list: {query_text}"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return query_text