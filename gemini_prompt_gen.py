import google.generativeai as genai

genai.configure(api_key="AIzaSyDxfzAG0F6GbD_0i3lQf9iqkMMKp3VhcrM")
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Explain LLM works in Jupyter Notebook")
print(response.text)