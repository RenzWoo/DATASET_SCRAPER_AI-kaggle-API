import google.generativeai as genai
import os

# usage of the key found in the notebook
KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBI1_q_i9XgsGeu3eaxVnGJ6ZFy4lZObn8")

print(f"Testing key: {KEY[:5]}...{KEY[-5:]}")

try:
    genai.configure(api_key=KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Hello, are you working?")
    print("✅ Success!")
    print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")
