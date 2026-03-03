from llm.gemini_parser import extract_skills

query = "Looking for a Java developer who collaborates with business teams"

result = extract_skills(query)

print(result)