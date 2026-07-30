"""
LangChain Sequential Chain Tutorial
===================================
This file demonstrates how to create a sequential pipeline (or "chain") 
using the LangChain Expression Language (LCEL).

A sequential chain links multiple chains together, where the output 
of one step becomes the input to the next.

In this example:
Step 1: Generate a detailed sports report about a specific topic.
Step 2: Take that report and summarize it into a 5-point presentation format.
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
load_dotenv() 

# ---------------------------------------------------------
# STEP 2: Configure the Base Language Model (LLM)
# ---------------------------------------------------------
# Connect to a Hugging Face model via its API endpoint.
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# ---------------------------------------------------------
# STEP 3: Create a Chat Model Wrapper
# ---------------------------------------------------------
# Wrap our base LLM to handle conversational chat message formats.
chat_model = ChatHuggingFace(llm=llm)

# ---------------------------------------------------------
# STEP 4: Define Prompt Templates for Each Stage
# ---------------------------------------------------------
# Prompt 1: For the initial report generation.
# This takes the starting '{topic}' as input.
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports journalist."),
    ("user", "Generate a report on {topic}")
])

# Prompt 2: For the summary/PPT generation.
# This will take the '{text}' output from the previous step as its input.
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports PPT expart."),
    ("user", "Generate a 5 points PPT/summary on {text}")
])

# ---------------------------------------------------------
# STEP 5: Add an Output Parser
# ---------------------------------------------------------
# Converts the model's message object output back into a standard string 
# so it can be cleanly fed into the next prompt.
parser = StrOutputParser()

# ---------------------------------------------------------
# STEP 6: Construct the Sequential Chain using LCEL
# ---------------------------------------------------------
# We pipe the output of the first sequence directly into the second sequence.
# Flow: Input Dict -> Prompt 1 -> Chat Model -> String Parser (Report Generated)
#       -> Prompt 2 -> Chat Model -> String Parser (PPT Summary Generated)
chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

# ---------------------------------------------------------
# STEP 7: Execute the Sequential Chain
# ---------------------------------------------------------
try:
    # We invoke the entire chain with the initial topic.
    # The chain automatically handles passing data between the steps.
    response = chain.invoke({'topic': 'Mohan Bagan FC'})
    
    print("--- Final AI Response (5-Point Summary) ---")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    print("Tip: Check if your token has 'Read' access and your .env file is in the same folder.")

# ---------------------------------------------------------
# EXTRA: Visualize the Chain
# ---------------------------------------------------------
# Prints an ASCII representation showing the sequential flow of data.
print("\n--- Sequential Chain Architecture ---")
chain.get_graph().print_ascii()