import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch

# 1. Load variables from .env into the system environment
load_dotenv() 

# 2. Configure the model via the API Endpoint
# Note: It automatically looks for HUGGINGFACEHUB_API_TOKEN in os.environ
llm_feedback_generator = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# 3. Wrap for Chat capability
chat_model = ChatHuggingFace(llm=llm_feedback_generator)

# 4. Prompt
# Use ChatPromptTemplate for better results with ChatHuggingFace
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a usser of a product."),
    ("user", "Generate a feedback randomly on {product}")
])

#5. Parser
parser = StrOutputParser()

# 5. Chain
initial_feedback_chain = prompt | chat_model | parser | (lambda x: {"text": x})

# topic : conditional chain

class Feedback(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(description = "Give the sentiment of the feedback")

parser_sentiment = PydanticOutputParser(pydantic_object = Feedback)

prompt_sentiment = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sentiment analysis agent."),
    ("user", "Classify the sentiment of this feedback:\n{text}\n\n{format_instructions}")
])

# 1. Generate Feedback
# Use partial to inject format instructions into the sentiment prompt
prompt_sentiment_with_format = prompt_sentiment.partial(
    format_instructions=parser_sentiment.get_format_instructions()
)

# 2. Classify Sentiment
# classification_chain = prompt_sentiment_with_format | chat_model | parser_sentiment

# 1. Define the Response Prompts
prompt_positive_reply = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly customer success manager."),
    ("user", "Write a thank you note for this positive feedback: {text}")
])

prompt_negative_reply = ChatPromptTemplate.from_messages([
    ("system", "You are a professional support agent."),
    ("user", "Write a polite apology and offer a refund for this negative feedback: {text}")
])

# 2. Define the Branching Logic
# This function looks at the 'sentiment' key from the previous step
def route_feedback(input_dict):
    if input_dict["sentiment_obj"].sentiment == "positive":
        return prompt_positive_reply | chat_model | StrOutputParser()
    else:
        return prompt_negative_reply | chat_model | StrOutputParser()

# 3. Build the Full Chain
# Step A: Generate Feedback
# Step B: Classify (We use a Parallel to keep the original text for the reply)
classification_parallel = RunnableParallel({
    "sentiment_obj": prompt_sentiment_with_format | chat_model | parser_sentiment,
    "text": lambda x: x["text"] # Keep the original feedback text
})

# Step C: Route to the correct reply
full_chain = (
    initial_feedback_chain 
    | classification_parallel 
    | RunnableLambda(route_feedback)
)

# 4. Execute
try:
    print("Processing feedback and generating appropriate response...")
    final_reply = full_chain.invoke({'product': 'Coffee Machine'})
    print("\n--- FINAL AGENT REPLY ---")
    print(final_reply)
except Exception as e:
    print(f"Error: {e}")