# YouTube Retrieval-Augmented Generation (RAG) Tutorial Note

This folder contains a step-by-step implementation of a **Retrieval-Augmented Generation (RAG)** pipeline designed to answer user questions based on the content of YouTube videos.

---

## Tutorial Steps Outline

The application is built incrementally through the following stages:
1. **Step 1: Fetching transcripts from YouTube videos using direct API calls.** (Completed)
2. **Step 2: Chunking & Creating Embeddings in FAISS vector store.** (Completed)
3. **Section 2: Retrieval via Retriever Interface.** (Completed)
4. **Step 4: Prompt Augmentation.** (Completed)
5. **Step 5: Generation (RAG QA).** (Completed)

---

## 1. Dependencies and Environment Setup

To support downloading transcripts and setting up the RAG pipeline, we configure our Python virtual environment with specific libraries.

### Installed Dependencies
Below are the packages installed in the `.venv` virtual environment for this tutorial:

*   **`youtube-transcript-api`** (`v1.2.4`): Used to communicate directly with YouTube's subtitle server to retrieve manual or auto-generated transcripts.
*   **`pytube`** (`v15.0.0`): Used for extracting video metadata (title, author, view counts). *(Note: See stability precautions below).*

To install these dependencies:
```powershell
.venv\Scripts\pip install youtube-transcript-api pytube
```

---

## 2. Step 1: YouTube Transcript Retrieval

### The Approach
To ensure maximum reliability and avoid parsing/scraping breaks associated with unmaintained scraping tools (like `pytube`), we fetch the transcript by communicating directly with YouTube's internal caption endpoint. We achieve this by using the `youtube-transcript-api` library directly, extracting the video ID from any standard YouTube URL, and wrapping the final transcript in a LangChain `Document` object.

### Implementation: `youtube_transcript.py`
We implement transcript fetching in [`youtube_transcript.py`](file:///c:/Coding/langChain/15_langchain_rag/youtube_transcript.py).

Here is the core implementation:
```python
from urllib.parse import urlparse, parse_qs
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    url = url.strip()
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            v_list = query.get('v')
            if v_list:
                return v_list[0]
        path_parts = parsed_url.path.split('/')
        for part in ('embed', 'v', 'shorts'):
            if part in path_parts:
                idx = path_parts.index(part)
                if idx + 1 < len(path_parts):
                    return path_parts[idx + 1]
    return None

def fetch_youtube_transcript(video_url: str):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None
    try:
        # Fetch transcript using YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US"])
        # Flatten chunks into a single text string
        transcript_text = " ".join(snippet.text for snippet in transcript_list)
        return Document(
            page_content=transcript_text,
            metadata={"source": video_id, "url": video_url}
        )
    except Exception as e:
        print(f"Error occurred while fetching YouTube transcript: {e}")
        return None
```

### Execution & Verification
To test the script, execute:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_transcript.py "https://www.youtube.com/watch?v=zjkBMFhNj_g"
```

---

## 3. Step 2: Chunking & Creating Embeddings

### The Concepts
Once the transcript is loaded as a single text string, we process it into a vector space representation using two key RAG components:
1.  **Text Splitters**: A single YouTube transcript can contain tens of thousands of characters. Passing this entire text into an embedding model or LLM prompt causes context length issues and dilutes relevance. We use `RecursiveCharacterTextSplitter` to chunk the transcript into blocks of `1000` characters with an overlap of `200` characters.
2.  **Vector Embeddings**: We convert each chunk into a mathematical vector representation. We initialize the `HuggingFaceEndpointEmbeddings` client to load the `"sentence-transformers/all-MiniLM-L6-v2"` model via Hugging Face's hosted Inference API. This converts our text chunks into 384-dimensional dense semantic vectors.
3.  **Vector Store (FAISS)**: Since we need to retrieve relevant transcript chunks in real-time, we load the chunks and their generated vectors into an in-memory `FAISS` database.

### Implementation: `youtube_rag.py`
We implement this in [`youtube_rag.py`](file:///c:/Coding/langChain/15_langchain_rag/youtube_rag.py).

Here is the chunking and vector store creation:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents([doc])

# 2. Embedding Model Setup
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# 3. In-memory FAISS indexing
vector_store = FAISS.from_documents(chunks, embeddings)
```

### Execution & Verification
To run the similarity search pipeline:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_rag.py "https://www.youtube.com/watch?v=zjkBMFhNj_g"
```

---

## 4. Section 2: Retrieval via Retriever Interface

### The Concept
To integrate our vector store into complex LangChain Expression Language (LCEL) chains, we convert the `FAISS` database into a generic `Retriever` interface using:
`retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})`

This abstracts away the database-specific query logic. Instead of calling database methods (like `similarity_search`), we query the retriever using:
`results = retriever.invoke(query)`

This method is standardized across all LangChain retrievers, making our pipeline modular.

### Implementation: `youtube_rag.py`
We implement this in [`youtube_rag.py`](file:///c:/Coding/langChain/15_langchain_rag/youtube_rag.py).

Here is the retriever instantiation and invocation:
```python
# Instantiate the Retriever from FAISS vector store
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Retrieve matching document chunks
results = retriever.invoke(query)
```

### Execution & Verification
To run the retriever query interface:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_rag.py "https://www.youtube.com/watch?v=zjkBMFhNj_g"
```

---

## 5. Step 4: Prompt Augmentation

### The Concept
Prompt Augmentation is the bridge between retrieval and generation. Rather than sending the user's raw query directly to the LLM, we inject the retrieved context documents into a structured prompt template. 

We define a `ChatPromptTemplate` containing:
- A **System message** instructing the model to use *only* the retrieved context documents to answer the question.
- A **Human message** containing the user's raw question.

We concatenate the text from the top matching chunks into a single context string and dynamically format the prompt:
```
System Prompt + Retrieved Chunks Context -> Dynamic System Message
User Question -> Human Message
```
This ensures the LLM generates a response anchored strictly in the video's content, reducing hallucinations.

### Implementation: `youtube_rag.py`
We implement this in [`youtube_rag.py`](file:///c:/Coding/langChain/15_langchain_rag/youtube_rag.py).

Here is the prompt template configuration and context binding:
```python
from langchain_core.prompts import ChatPromptTemplate

# 1. Define prompt template mapping context and question
system_prompt = (
    "You are a helpful assistant. Use the following context retrieved from the YouTube video transcript "
    "to answer the question. If you don't know the answer, say that you don't know.\n\n"
    "Context:\n{context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 2. Extract retrieved document text and format context string
context_text = "\n\n".join(doc.page_content for doc in results)

# 3. Augment prompt template with context and question variables
augmented_prompt = prompt_template.invoke({
    "context": context_text,
    "question": query
})
```

### Execution & Verification
To run the prompt augmentation test:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_rag.py "https://www.youtube.com/watch?v=zjkBMFhNj_g"
```

---

## 6. Step 5: Generation (QA Step-by-Step)

### The Concept
Generation is the final step where the formatted augmented prompt is passed to the Language Model (LLM) to generate the final answer. 

Rather than wrapping the flow in an abstract chain abstraction (like LCEL pipe chaining), we write the execution explicitly part-by-part:
1. **Retrieval**: Extract documents: `results = retriever.invoke(query)`
2. **Augmentation**: Concatenate document contents into `context_text` and format prompt: `augmented_prompt = prompt_template.invoke({"context": context_text, "question": query})`
3. **Generation**: Directly pass `augmented_prompt` to `chat_model.invoke(augmented_prompt)` and print the resulting message content `response_message.content`.

This explicit approach makes every data flow and variable mapping completely visible and easy to debug.

### Implementation: `youtube_rag.py`
We implement this in [`youtube_rag.py`](file:///c:/Coding/langChain/15_langchain_rag/youtube_rag.py).

Here is the LLM initialization and RAG invocation helper function:
```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# 1. Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
chat_model = ChatHuggingFace(llm=llm)

# 2. Explicit Step-by-Step execution function
def run_rag_step(query: str, retriever, prompt_template, chat_model):
    # Step 1: Retrieval
    results = retriever.invoke(query)
    
    # Step 2: Augmentation
    context_text = "\n\n".join(doc.page_content for doc in results)
    augmented_prompt = prompt_template.invoke({
        "context": context_text,
        "question": query
    })
    
    # Step 3: Generation
    response_message = chat_model.invoke(augmented_prompt)
    print("AI Response:", response_message.content)
```

### Execution & Verification
To run the QA chat program:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_rag.py "https://www.youtube.com/watch?v=zjkBMFhNj_g"
```
Or run a non-interactive test directly from the command line:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_rag.py "https://www.youtube.com/watch?v=zjkBMFhNj_g" "What are the two files that make up llama 2?"
```
This prints the steps explicitly and generates the model's answer.

