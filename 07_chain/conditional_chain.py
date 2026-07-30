"""
LangChain Conditional Chain Tutorial
====================================
This file demonstrates how to create a conditional pipeline (or "branching chain") 
using the LangChain Expression Language (LCEL) and RunnableLambda.

A conditional chain allows your pipeline to make decisions and take different 
paths based on the output of previous steps.

In this example:
Phase 1: Generate a random piece of customer feedback for a product.
Phase 2: Analyze the sentiment of that feedback (Positive or Negative).
Phase 3 (Conditional): 
    - IF Positive -> Route to a Customer Success Manager prompt to write a Thank You note.
    - IF Negative -> Route to a Support Agent prompt to write an Apology and offer a refund.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
# Load environment variables from a .env file into the system environment.
load_dotenv() 

# ---------------------------------------------------------
# STEP 2: Configure the Base Model (LLM)
# ---------------------------------------------------------
# We'll use one base model for all the tasks in this chain to keep it simple.
llm_feedback_generator = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# Wrap for Chat capability
chat_model = ChatHuggingFace(llm=llm_feedback_generator)

# ---------------------------------------------------------
# STEP 3: Setup the Initial Feedback Generator
# ---------------------------------------------------------
# This prompt asks the AI to act as a user and generate random feedback.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a usser of a product."),
    ("user", "Generate a feedback randomly on {product}")
])

parser = StrOutputParser()

# The initial chain generates the feedback text and packages it into a dictionary.
initial_feedback_chain = prompt | chat_model | parser | (lambda x: {"text": x})

# ---------------------------------------------------------
# STEP 4: Setup the Sentiment Analyzer (Structured Output)
# ---------------------------------------------------------

# We define a Pydantic model to force the LLM to output exactly what we want.
# We restrict the output to ONLY "positive" or "negative".
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description = "Give the sentiment of the feedback")

# PydanticOutputParser helps parse the LLM's raw text into our Python Pydantic object.
parser_sentiment = PydanticOutputParser(pydantic_object = Feedback)

prompt_sentiment = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sentiment analysis agent."),
    ("user", "Classify the sentiment of this feedback:\n{text}\n\n{format_instructions}")
])

# We use .partial() to inject the formatting instructions required by Pydantic 
# into our prompt template before we run the chain.
prompt_sentiment_with_format = prompt_sentiment.partial(
    format_instructions=parser_sentiment.get_format_instructions()
)

# ---------------------------------------------------------
# STEP 5: Define the Branching Logic (The "Condition")
# ---------------------------------------------------------

# Define the Prompt for Positive Feedback
prompt_positive_reply = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly customer success manager."),
    ("user", "Write a thank you note for this positive feedback: {text}")
])

# Define the Prompt for Negative Feedback
prompt_negative_reply = ChatPromptTemplate.from_messages([
    ("system", "You are a professional support agent."),
    ("user", "Write a polite apology and offer a refund for this negative feedback: {text}")
])

# This function looks at the Pydantic object from the previous step to determine the route.
# It dynamically returns a new chain based on the sentiment!
def route_feedback(input_dict):
    if input_dict["sentiment_obj"].sentiment == "positive":
        return prompt_positive_reply | chat_model | StrOutputParser()
    else:
        return prompt_negative_reply | chat_model | StrOutputParser()

# ---------------------------------------------------------
# STEP 6: Build the Full Conditional Chain
# ---------------------------------------------------------

# Phase 2: Classification (Using Parallel)
# We use RunnableParallel to run the sentiment classification AND pass through the 
# original text at the same time. The routing function needs both!
classification_parallel = RunnableParallel({
    "sentiment_obj": prompt_sentiment_with_format | chat_model | parser_sentiment,
    "text": lambda x: x["text"] # Keep the original feedback text flowing through
})

# Phase 3: The Full Pipeline
full_chain = (
    initial_feedback_chain 
    | classification_parallel 
    | RunnableLambda(route_feedback) # RunnableLambda turns our Python function into a Chain step
)

# ---------------------------------------------------------
# STEP 7: Execute the Conditional Chain
# ---------------------------------------------------------
try:
    print("Processing feedback and generating appropriate response...")
    # This single call kicks off the generation -> classification -> routing -> final reply!
    final_reply = full_chain.invoke({'product': 'Coffee Machine'})
    
    print("\n--- FINAL AGENT REPLY ---")
    print(final_reply)
except Exception as e:
    print(f"Error: {e}")