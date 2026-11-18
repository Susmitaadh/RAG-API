
from dotenv import load_dotenv
from google import genai
import time

load_dotenv()

client = genai.Client()

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Explain RAG in simple words."
        )

        print("Gemini response:")
        print(response.text)
        break

    except Exception as error:
        print(f"Attempt {attempt + 1} failed: {error}")

        if attempt < 2:
            print("Retrying in 10 seconds...")
            time.sleep(10)
