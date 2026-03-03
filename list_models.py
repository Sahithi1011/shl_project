import google.generativeai as genai

genai.configure(api_key="AIzaSyCOScetuD2AUTrpTmg-okEJ43sNxHPbHG4")

for m in genai.list_models():
    print(m.name)