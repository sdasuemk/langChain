# length based text splitting

from langchain_text_splitters import CharacterTextSplitter


text = """Paragraph 1:
This is the first paragraph. It contains some detailed information that needs to be split properly.

Paragraph 2:
This is the second paragraph. It also contains text that will be chunked based on size and separator.

Paragraph 3:
This is the third paragraph.
"""

# Initialize the splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n", 
    #CharacterTextSplitter will not break apart a piece of text if it doesn't 
    # find your designated separator ("\n\n" by default) within the target chunk_size.
    # Since Paragraph 1 has no "\n\n" inside it, the splitter cannot split it any
    # further without breaking its rules, so it keeps the entire 114-character paragraph 
    # intact rather than cutting off in the middle of words.  separator="", in this case it will split
    # after 100 character reached.
    # separator=""
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# Split a raw string into text chunks
chunks = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)