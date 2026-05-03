"""
LangChain Simple Chain Tutorial
===============================
This file demonstrates how to create a basic LangChain pipeline (or "chain") 
using the LangChain Expression Language (LCEL).

A chain typically consists of three main components:
1. Prompt Template: Formats the user input.
2. LLM / Chat Model: Generates a response based on the prompt.
3. Output Parser: Extracts the desired output format from the model's response.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
# Load environment variables from a .env file into the system environment.
# This is crucial for securely loading API keys, such as your Hugging Face token.
load_dotenv() 

# ---------------------------------------------------------
# STEP 2: Configure the Base Language Model (LLM)
# ---------------------------------------------------------
# Here we connect to a Hugging Face model via its API endpoint.
# Note: HuggingFaceEndpoint automatically looks for the 'HUGGINGFACEHUB_API_TOKEN'
# in your environment variables (loaded via dotenv above).
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", # The Hugging Face model repository ID
    task="text-generation",                # The type of task the model will perform
    max_new_tokens=512,                    # Limits the length of the generated response
    temperature=0.7                        # Controls the creativity/randomness (0.0 to 1.0)
)

# ---------------------------------------------------------
# STEP 3: Create a Chat Model Wrapper
# ---------------------------------------------------------
# Many modern models operate best in a conversational (chat) format.
# ChatHuggingFace wraps our base LLM to handle chat message formats 
# (e.g., system messages, user messages).
chat_model = ChatHuggingFace(llm=llm)

# ---------------------------------------------------------
# STEP 4: Define the Prompt Template
# ---------------------------------------------------------
# ChatPromptTemplate formats inputs into a list of chat messages.
# Variables enclosed in curly braces, like {topic}, will be replaced 
# dynamically when we run the chain.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports journalist."), # Sets the AI's persona
    ("user", "Generate a report on {topic}")                 # The user's request
])

# ---------------------------------------------------------
# STEP 5: Add an Output Parser
# ---------------------------------------------------------
# StrOutputParser converts the complex chat message object returned 
# by the model into a simple, readable string.
parser = StrOutputParser()

# ---------------------------------------------------------
# STEP 6: Construct the Chain using LCEL
# ---------------------------------------------------------
# LangChain Expression Language (LCEL) uses the pipe operator '|' 
# to chain components together. Data flows from left to right.
# Flow: Input Dict -> Prompt Template -> Chat Model -> Output Parser
chain = prompt | chat_model | parser

# ---------------------------------------------------------
# STEP 7: Execute the Chain
# ---------------------------------------------------------
try:
    # We use `.invoke()` to pass data into the chain. 
    # Here, we provide the value for the '{topic}' variable defined in our prompt.
    response = chain.invoke({'topic': 'Mohan Bagan FC'})
    print("--- AI Response ---")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    print("Tip: Check if your token has 'Read' access and your .env file is in the same folder.")

# ---------------------------------------------------------
# EXTRA: Visualize the Chain
# ---------------------------------------------------------
# This prints a helpful ASCII diagram showing how data flows through your chain.
print("\n--- Chain Architecture ---")
chain.get_graph().print_ascii()