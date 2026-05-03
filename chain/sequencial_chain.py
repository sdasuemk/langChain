import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load variables from .env into the system environment
load_dotenv() 

# 2. Configure the model via the API Endpoint
# Note: It automatically looks for HUGGINGFACEHUB_API_TOKEN in os.environ
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# 3. Wrap for Chat capability 
chat_model = ChatHuggingFace(llm=llm)

# 4. Prompt
# Use ChatPromptTemplate for better results with ChatHuggingFace
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports journalist."),
    ("user", "Generate a report on {topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports PPT expart."),
    ("user", "Generate a 5 points PPT/summary on {text}")
])

#5. Parser
parser = StrOutputParser()

# 5. Chain
chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

try:
    response = chain.invoke({'topic': 'Mohan Bagan FC'})
    print("--- AI Response ---")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    print("Tip: Check if your token has 'Read' access and your .env file is in the same folder.")

# Print the visual representation of the chain
chain.get_graph().print_ascii()