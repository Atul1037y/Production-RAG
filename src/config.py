import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    return api_key

API_KEY = get_api_key()