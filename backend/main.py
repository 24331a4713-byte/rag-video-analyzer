from dotenv import load_dotenv
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from yt_dlp import YoutubeDL
from langchain_community.retrievers import BM25Retriever
import os
from langchain_groq import ChatGroq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

current_vectorstore = None
current_hybrid_retriever = None
current_metadata_y = None
current_metadata_i = None

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("invalid youtube url")
    return match.group(1)

def transcribe_audio(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join(s.text for s in segments)

def get_youtube_transcript(url: str) -> str:
    audio_path = "/tmp/yt_audio.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "/tmp/yt_audio.%(ext)s",
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    transcript = transcribe_audio(audio_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return transcript

def get_instagram_transcript(url: str) -> str:
    audio_path = "/tmp/ig_audio.mp3"
    cookies_path = "cookies.txt"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "/tmp/ig_audio.%(ext)s",
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    transcript = transcribe_audio(audio_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return transcript

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_video_metadata(url: str, cookies_path: str = None) -> dict:
    ydl_opts = {"quiet": True, "skip_download": True}
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title":   info.get("title", ""),
        "channel": info.get("uploader", ""),
        "views":   info.get("view_count") or 0,
        "likes":   info.get("like_count") or 0,
        "duration": info.get("duration") or 0,
        "id":      info.get("id", ""),
        "creator": info.get("uploader", ""),
        "comments": info.get("comment_count") or 0,
    }

class HybridRetriever:
    def __init__(self, mmr, bm25):
        self.mmr = mmr
        self.bm25 = bm25

    def invoke(self, query):
        results1 = self.mmr.invoke(query)
        results2 = self.bm25.invoke(query)
        combined = {doc.metadata.get('video_id'): doc for doc in results1}
        for doc in results2:
            if doc.metadata.get('video_id') not in combined:
                combined[doc.metadata.get('video_id')] = doc
        return list(combined.values())[:4]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    global current_vectorstore, current_hybrid_retriever, current_metadata_y, current_metadata_i

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # YouTube — yt-dlp + Whisper (no IP block issue)
        youtube_text     = clean_text(get_youtube_transcript(req.youtube_url))
        current_metadata_y = get_video_metadata(req.youtube_url)
        youtube_video_id = extract_video_id(req.youtube_url)

        # Instagram — yt-dlp + Whisper
        insta_text       = clean_text(get_instagram_transcript(req.instagram_url))
        current_metadata_i = get_video_metadata(req.instagram_url, cookies_path="cookies.txt")

        # Chunk
        chunks_youtube = splitter.split_text(youtube_text)
        chunks_insta   = splitter.split_text(insta_text)

        # Documents
        documents = []
        for chunk in chunks_youtube:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source":   "youtube",
                    "video_id": youtube_video_id,
                    "title":    current_metadata_y["title"],
                    "views":    current_metadata_y["views"],
                    "likes":    current_metadata_y["likes"],
                }
            ))
        for chunk in chunks_insta:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source":   "instagram",
                    "video_id": current_metadata_i["id"],
                    "title":    current_metadata_i["title"],
                    "views":    current_metadata_i["views"],
                    "likes":    current_metadata_i["likes"],
                }
            ))

        # FAISS + retrievers
        current_vectorstore = FAISS.from_documents(documents, embeddings)
        mmr_retriever = current_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.6}
        )
        bm25_retriever   = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4
        current_hybrid_retriever = HybridRetriever(mmr_retriever, bm25_retriever)

        return {
            "video_a": {
                "video_id": youtube_video_id,
                "title":    current_metadata_y["title"],
                "views":    current_metadata_y["views"],
                "likes":    current_metadata_y["likes"],
                "channel":  current_metadata_y["channel"],
            },
            "video_b": {
                "video_id": current_metadata_i["id"],
                "title":    current_metadata_i["title"],
                "views":    current_metadata_i["views"],
                "likes":    current_metadata_i["likes"],
                "creator":  current_metadata_i["creator"],
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    if not current_hybrid_retriever:
        return {"response": "Extract videos first"}

    docs    = current_hybrid_retriever.invoke(req.question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an AI Content Comparison Assistant.

YOUTUBE VIDEO METADATA:
- Title: {current_metadata_y['title']}
- Views: {current_metadata_y['views']}
- Likes: {current_metadata_y['likes']}
- Channel: {current_metadata_y['channel']}
- Duration: {current_metadata_y['duration']}

INSTAGRAM VIDEO METADATA:
- Title: {current_metadata_i['title']}
- Views: {current_metadata_i['views']}
- Likes: {current_metadata_i['likes']}
- Creator: {current_metadata_i['creator']}

Context:
{context}

Question:
{req.question}

Comparison:
"""
    response = llm.invoke(prompt)
    return {"response": response.content}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)