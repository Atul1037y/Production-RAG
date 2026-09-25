from google import genai
from src.config import API_KEY

def test_connection():
    try:
        client = genai.Client(api_key=API_KEY)
        
        print("Testing Gemini API connection...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Respond with a simple "Hello, world!"'
        )
        print("Success! Gemini response:", response.text)
    except Exception as e:
        print(f"Failed to connect to Gemini API: {e}")

if __name__ == "__main__":
    test_connection()