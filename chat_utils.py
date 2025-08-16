# chat_utils.py
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY

MODEL = "gpt-4o-mini"   # you can switch to "gpt-4o" or "gpt-3.5-turbo"
TEMPERATURE = 0.7

def get_chat_model(api_key: str = None):
    """Initialize the OpenAI chat model"""
    return ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        api_key=api_key or OPENAI_API_KEY
    )

def ask_chat_model(chat_model, prompt: str):
    """Send a prompt to the model and return the response text"""
    response = chat_model.invoke(prompt)
    return response.content


