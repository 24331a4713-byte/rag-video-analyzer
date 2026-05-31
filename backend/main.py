from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from yt_dlp import YoutubeDL
from langchain_community.retrievers import BM25Retriever
import os
from langchain_groq import ChatGroq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Global state
current_bm25_retriever = None
current_metadata_y = None
current_metadata_i = None

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("invalid youtube url")
    return match.group(1)

def get_transcript(url):
    video_id = extract_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=['en'])
    text = " ".join(snippet.text for snippet in transcript)
    return text

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def get_video_metadata(url):
    ydl_opts = {"quiet": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "duration": info.get("duration")
    }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractRequest(BaseModel):
    youtube_url: str
    instagram_url: str

class ChatRequest(BaseModel):
    question: str
    video_a_id: str
    video_b_id: str

@app.post("/extract")
async def extract_videos(req: ExtractRequest):
    global current_bm25_retriever, current_metadata_y, current_metadata_i
    
    url_yt = req.youtube_url
    url_ig = req.instagram_url

    youtube_text = get_transcript(url_yt)
    youtube_text = clean_text(youtube_text)
    current_metadata_y = get_video_metadata(url_yt)
    youtube_video_id = extract_video_id(url_yt)

    with YoutubeDL({"cookiefile": "cookies.txt", "quiet": True}) as ydl:
        info = ydl.extract_info(url_ig, download=False)

    current_metadata_i = {
        "id": info.get("id"),
        "title": info.get("title"),
        "creator": info.get("uploader"),
        "views": info.get("view_count") or 0,
        "likes": info.get("like_count") or 0,
    }

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_youtube = splitter.split_text(youtube_text)

    documents = []
    for chunk in chunks_youtube:
        documents.append(
            Document(page_content=chunk, metadata={
                "source": "youtube",
                "video_id": youtube_video_id,
                "title": current_metadata_y["title"],
                "views": current_metadata_y["views"],
                "likes": current_metadata_y["likes"]
            })
        )

    # Only BM25 - no embeddings
    current_bm25_retriever = BM25Retriever.from_documents(documents)
    current_bm25_retriever.k = 4

    return {
        "video_a": {
            "video_id": youtube_video_id,
            "title": current_metadata_y["title"],
            "views": current_metadata_y["views"],
            "likes": current_metadata_y["likes"],
            "channel": current_metadata_y["channel"]
        },
        "video_b": {
            "video_id": current_metadata_i["id"],
            "title": current_metadata_i["title"],
            "views": current_metadata_i["views"],
            "likes": current_metadata_i["likes"],
            "creator": current_metadata_i["creator"]
        }
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not current_bm25_retriever:
        return {"response": "Extract videos first"}
    
    docs = current_bm25_retriever.invoke(req.question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an AI Content Comparison Assistant.

YOUTUBE VIDEO:
- Title: {current_metadata_y['title']}
- Views: {current_metadata_y['views']}
- Likes: {current_metadata_y['likes']}
- Channel: {current_metadata_y['channel']}

INSTAGRAM VIDEO:
- Title: {current_metadata_i['title']}
- Views: {current_metadata_i['views']}
- Likes: {current_metadata_i['likes']}
- Creator: {current_metadata_i['creator']}

Content:
{context}

Question: {req.question}

Analysis:
"""
    response = llm.invoke(prompt)
    return {"response": response.content}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)