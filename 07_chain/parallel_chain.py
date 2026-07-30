"""
LangChain Parallel Chain Tutorial
=================================
This file demonstrates how to create a parallel pipeline (or "chain") 
using the LangChain Expression Language (LCEL) and RunnableParallel.

A parallel chain allows you to run multiple independent chains at the 
same time, which is much faster than running them sequentially.

In this example:
Phase 1: Generate an initial sports report (Sequential).
Phase 2 (Parallel): Simultaneously generate two things from that report:
    - Notes (using Llama 3.1)
    - Quiz (using Qwen3)
Phase 3: Merge the notes and quiz into one final document.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
# Load environment variables from a .env file into the system environment.
load_dotenv() 

# ---------------------------------------------------------
# STEP 2: Configure the Models (LLMs)
# ---------------------------------------------------------
# We will use three different models for the three different tasks!

# Model 1: For the initial research/report generation
llm_research = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)
chat_model_research = ChatHuggingFace(llm=llm_research)

# Model 2: For generating notes
llm_notes = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)
chat_model_notes = ChatHuggingFace(llm=llm_notes)

# Model 3: For generating a quiz (and merging at the end)
llm_quiz = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)
chat_model_quiz = ChatHuggingFace(llm=llm_quiz)

# ---------------------------------------------------------
# STEP 3: Define Output Parser & Prompt Templates
# ---------------------------------------------------------
parser = StrOutputParser()

# Prompt for the initial report
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports journalist."),
    ("user", "Generate a report on {topic}")
])

# Prompts for the parallel branches
prompt_notes = ChatPromptTemplate.from_messages([
    ("system", "You are a professional note generater fron detail report."),
    ("user", "Generate notes on {text}")
])

prompt_quiz = ChatPromptTemplate.from_messages([
    ("system", "You are a professional quiz master, and you have a detail report."),
    ("user", "Generate quiz on {text}")
])

# Prompt for combining the results
prompt_combine = ChatPromptTemplate.from_messages([
    ("system", "You are a professional modular/ combiner. You have Notes and Quiz"),
    ("user", "Combine these into one document:\n\nNOTES:\n{notes}\n\nQUIZ:\n{quiz}")
])

# ---------------------------------------------------------
# STEP 4: Construct the Chains using LCEL
# ---------------------------------------------------------

# Phase 1: Initial Report Chain
# Notice the lambda function: it wraps the string output into a dictionary `{"text": "..."}` 
# so the next chains (which expect {text} as input) can use it.
initial_report_chain = prompt | chat_model_research | parser | (lambda x: {"text": x})

# Phase 2: Parallel Chain
# RunnableParallel runs all its dictionary values at the same time.
# Here, both the notes chain and quiz chain receive the output dictionary from Phase 1.
parallel_chain = RunnableParallel({
    "notes": prompt_notes | chat_model_notes | parser,
    "quiz": prompt_quiz | chat_model_quiz | parser,
})

# Phase 3: Merge Chain
# Takes the {"notes": "...", "quiz": "..."} output from Phase 2 and combines them.
merge_chain = prompt_combine | chat_model_quiz | parser

# ---------------------------------------------------------
# STEP 5: Combine Everything into the Full Chain
# ---------------------------------------------------------
# Flow: Topic -> Initial Report -> (Notes || Quiz) -> Merge
full_chain = initial_report_chain | parallel_chain | merge_chain

# ---------------------------------------------------------
# STEP 6: Execute the Full Parallel Chain
# ---------------------------------------------------------
try:
    print("Generating report, notes, and quiz... please wait.")
    # Invoking the full chain kicks off the entire sequence.
    response = full_chain.invoke({'topic': 'Mohun Bagan FC'})
    
    print("\n--- FINAL COMBINED DOCUMENT ---")
    print(response)
    
    # EXTRA: Visualize the Chain
    # Prints an ASCII representation showing the parallel flow.
    print("\n--- CHAIN STRUCTURE ---")
    full_chain.get_graph().print_ascii()
    
except Exception as e:
    print(f"Error: {e}")