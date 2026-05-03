import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 1. Load variables from .env into the system environment
load_dotenv() 

# 2. Configure the model via the API Endpoint
# Note: It automatically looks for HUGGINGFACEHUB_API_TOKEN in os.environ
llm_research = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# 3. Wrap for Chat capability
chat_model_research = ChatHuggingFace(llm=llm_research)

# 4. Prompt
# Use ChatPromptTemplate for better results with ChatHuggingFace
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sports journalist."),
    ("user", "Generate a report on {topic}")
])

#5. Parser
parser = StrOutputParser()

# 5. Chain
initial_report_chain = prompt | chat_model_research | parser | (lambda x: {"text": x})


# above is simple text generated. Now I will generate 2 model. model -1 for Notes and Model-2 for quiz parallal way.
# then combine them and generate one document.

llm_notes = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# 3. Wrap for Chat capability
chat_model_notes = ChatHuggingFace(llm=llm_notes)

llm_quiz = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# 3. Wrap for Chat capability
chat_model_quiz = ChatHuggingFace(llm=llm_quiz)

prompt_notes = ChatPromptTemplate.from_messages([
    ("system", "You are a professional note generater fron detail report."),
    ("user", "Generate notes on {text}")
])
prompt_quiz = ChatPromptTemplate.from_messages([
    ("system", "You are a professional quiz master, and you have a detail report."),
    ("user", "Generate quiz on {text}")
])

prompt_combine = ChatPromptTemplate.from_messages([
    ("system", "You are a professional modular/ combiner. You have Notes and Quiz"),
    ("user", "Combine these into one document:\n\nNOTES:\n{notes}\n\nQUIZ:\n{quiz}")
])

parallel_chain = RunnableParallel({
"notes": prompt_notes | chat_model_notes | parser,
"quiz": prompt_quiz | chat_model_quiz | parser,
})

merge_chain = prompt_combine | chat_model_quiz | parser

full_chain = initial_report_chain | parallel_chain | merge_chain

try:
    print("Generating report, notes, and quiz... please wait.")
    response = full_chain.invoke({'topic': 'Mohun Bagan FC'})
    print("\n--- FINAL COMBINED DOCUMENT ---")
    print(response)
    
    print("\n--- CHAIN STRUCTURE ---")
    full_chain.get_graph().print_ascii()
    
except Exception as e:
    print(f"Error: {e}")

# Print the visual representation of the chain
# chain.get_graph().print_ascii()