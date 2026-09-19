"""
LangChain Document Loading & Summarization Pipeline Tutorial
=============================================================
This file demonstrates how to build an end-to-end pipeline that loads 
a custom text document using TextLoader, and summarizes it using 
the LangChain Expression Language (LCEL), a Hugging Face API model, 
and advanced prompting techniques.

Key Concepts:
- TextLoader: Loads and extracts text content from local files.
- Hugging Face Inference Endpoint: Accesses models remotely via API endpoints.
- Best-Practice Prompting: Uses role-based system instructions, clear delimiters 
  (<tag></tag>) to segment input data, and explicit structural constraints.
- LCEL: Chains the prompt, model, and output parser using the pipe (|) operator.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
# Load environment variables from the .env file.
# This initializes 'HUGGINGFACEHUB_API_TOKEN' for API requests.
load_dotenv()

# ---------------------------------------------------------
# STEP 2: Load the Document (Poem)
# ---------------------------------------------------------
# We use TextLoader to load the custom LangChain poem.
# This reads the text and packages it inside a Document object.
loader = TextLoader("asset/langchain_poem.txt", encoding="utf-8")
docs = loader.load()

# Extract the page content (the actual text) from the document list
poem_content = docs[0].page_content

# ---------------------------------------------------------
# STEP 3: Configure the Hugging Face API Model
# ---------------------------------------------------------
# Set up the base model and wrap it in ChatHuggingFace to support
# structured, chat-based system and user messages.
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", # Hugging Face hosted model
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3                        # Low temperature for focused, analytical output
)
chat_model = ChatHuggingFace(llm=llm)

# ---------------------------------------------------------
# STEP 4: Define the Advanced Prompt Template
# ---------------------------------------------------------
# Here we employ the latest prompting best practices:
# 1. System Role: Gives the model a specific persona (Literary Critic).
# 2. Context Separation: Wraps the dynamic input in XML-style tags (<poem>...) 
#    to define boundaries and protect against prompt injection.
# 3. Output Schema: Enforces a strict, well-formatted markdown structure.
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert literary critic and analytical assistant.\n"
        "Your task is to analyze and summarize the input poem.\n"
        "Follow these structural guidelines to construct your output:\n"
        "1. **Core Theme**: A one-sentence high-level summary of what the poem is about.\n"
        "2. **Key Concepts**: A 3-point bulleted list detailing the specific technologies or steps referenced in the poem.\n"
        "3. **Takeaway**: A single, impactful sentence highlighting the poem's conclusion.\n"
        "Ensure your output is professional, well-formatted, and uses markdown."
    )),
    ("user", "Here is the poem to summarize:\n\n<poem>\n{poem_content}\n</poem>")
])

# ---------------------------------------------------------
# STEP 5: Construct the Chain using LCEL
# ---------------------------------------------------------
# Use the pipe operator (|) to construct the pipeline:
# Input Dict -> Prompt Template -> Chat Model -> String Parser
parser = StrOutputParser()
chain = prompt | chat_model | parser

# ---------------------------------------------------------
# STEP 6: Execute the Summarization Chain
# ---------------------------------------------------------
# We invoke the chain, passing the extracted poem text.
try:
    print("Sending poem to Hugging Face API for summarization...")
    response = chain.invoke({"poem_content": poem_content})
    print("\n--- Summary Output ---")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    print("Tip: Check if your token is valid and set in the .env file.")
