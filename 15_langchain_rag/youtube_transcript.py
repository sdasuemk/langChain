import os
import sys
import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Load environment variables
load_dotenv()

def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    """
    url = url.strip()
    parsed_url = urlparse(url)
    
    # Handle youtu.be short URLs
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    
    # Handle youtube.com URLs
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        # watch?v=VIDEO_ID
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            v_list = query.get('v')
            if v_list:
                return v_list[0]
        # embed/VIDEO_ID or v/VIDEO_ID or shorts/VIDEO_ID
        path_parts = parsed_url.path.split('/')
        for part in ('embed', 'v', 'shorts'):
            if part in path_parts:
                idx = path_parts.index(part)
                if idx + 1 < len(path_parts):
                    return path_parts[idx + 1]
                    
    # Fallback regex search for 11-char ID
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        return match.group(1)
        
    return None

def fetch_youtube_transcript(video_url: str):
    """
    Extracts the video ID from the URL and fetches the transcript using YouTubeTranscriptApi.
    Flattens the transcript and wraps it in a LangChain Document.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        print(f"Could not extract a valid YouTube video ID from URL: {video_url}")
        return None
        
    print(f"Extracted video ID: {video_id}")
    print("Attempting to fetch transcript...")
    try:
        # Fetch transcript using YouTubeTranscriptApi instance
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US"])
        
        # Flatten chunks into a single text string
        transcript_text = " ".join(snippet.text for snippet in transcript_list)
        
        # Return as a LangChain Document with metadata
        return Document(
            page_content=transcript_text,
            metadata={"source": video_id, "url": video_url}
        )
        
    except TranscriptsDisabled:
        print("No captions available for this video (transcripts are disabled).")
        return None
    except NoTranscriptFound:
        print("No English captions or transcripts found for this video.")
        return None
    except Exception as e:
        print(f"Error occurred while fetching YouTube transcript: {e}")
        return None

def main():
    # A default video URL (Intro to Large Language Models by Andrej Karpathy)
    default_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = input(f"Enter YouTube URL (press Enter for default '{default_url}'): ").strip()
        if not video_url:
            video_url = default_url

    doc = fetch_youtube_transcript(video_url)
    
    if doc:
        print("\n=== Video Metadata ===")
        metadata = doc.metadata
        print(f"Title:        {metadata.get('title', 'N/A')}")
        print(f"Author:       {metadata.get('author', 'N/A')}")
        print(f"Source/URL:   {metadata.get('source', 'N/A')}")
        print(f"Length (sec): {metadata.get('length', 'N/A')}")
        print(f"View Count:   {metadata.get('view_count', 'N/A')}")
        
        print("\n=== Transcript Snippet (First 500 chars) ===")
        content_snippet = doc.page_content[:500]
        print(content_snippet)
        print("...")
        print(f"\nTotal transcript length: {len(doc.page_content)} characters.")

if __name__ == "__main__":
    main()
