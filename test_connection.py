import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Step 1: Securely load the API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Step 2: Validate the key exists in the environment
if not api_key:
    print("CRITICAL ERROR: Could not find GEMINI_API_KEY in the .env file.")
    exit()
else:
    print("SYSTEM: API Key loaded successfully. Establishing connection...")

# Step 3: Initialize the Gemini 3.5 Flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    google_api_key=api_key
)
# Step 4: Send the SBR-aligned test prompt
print("SYSTEM: Sending test prompt to Gemini...")
test_prompt = "Define metacognition in one short sentence as if explaining it to a software developer."

# Step 5: Execute and catch any errors
try:
    response = llm.invoke(test_prompt)
    print("\n--- CONNECTION SUCCESSFUL ---")
    print("GEMINI RESPONSE:", response.content)
    print("-----------------------------\n")
except Exception as e:
    print("\n--- CONNECTION FAILED ---")
    print("ERROR:", e)