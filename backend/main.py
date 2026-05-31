from dotenv import load_dotenv
from faster_whisper import WhisperModel
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from yt_dlp import YoutubeDL
from langchain_community.retrievers import BM25Retriever
import os
from langchain_groq import ChatGroq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

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

whisper_model = WhisperModel("base")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

segments, _ = whisper_model.transcribe("insta_video.mp4")
insta_transcript = ""
for segment in segments:
    insta_transcript += segment.text + " "
print(insta_transcript)


url = 'https://www.youtube.com/watch?v=LPZh9BOjkQs'
youtube_text = get_transcript(url)
youtube_text = clean_text(youtube_text)
metadata_y = get_video_metadata(url)
youtube_video_id = extract_video_id(url)
print(youtube_text[:500])


url_insta = "https://www.instagram.com/p/DTsl40oj7jQ/"
with YoutubeDL({"cookiefile": "cookies.txt", "quiet": True}) as ydl:
    info = ydl.extract_info(url_insta, download=False)

metadata_i = {
    "id": info.get("id"),
    "title": info.get("title"),
    "description": info.get("description"),
    "creator": info.get("uploader"),
    "views": info.get("view_count") or 0,
    "likes": info.get("like_count") or 0,
    "comments": info.get("comment_count") or 0,
    "duration": info.get("duration"),
    "url": info.get("webpage_url")
}
print(metadata_i)


insta_text = clean_text(insta_transcript)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks_youtube = splitter.split_text(youtube_text)
chunks_insta = splitter.split_text(insta_text)


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

documents = []
for chunk in chunks_youtube:
    documents.append(
        Document(page_content=chunk, metadata={
            "source": "youtube",
            "video_id": youtube_video_id,
            "title": metadata_y["title"],
            "views": metadata_y["views"],
            "likes": metadata_y["likes"]
        })
    )
for chunk in chunks_insta:
    documents.append(
        Document(page_content=chunk, metadata={
            "source": "instagram",
            "video_id": metadata_i["id"],
            "title": metadata_i["title"],
            "views": metadata_i["views"],
            "likes": metadata_i["likes"]
        })
    )


vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")
print("FAISS index saved")

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.6}
)

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4

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

hybrid_retriever = HybridRetriever(mmr_retriever, bm25_retriever)

def get_retriever_for_video(video_id: str):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "filter": {"video_id": video_id}}
    )


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
    url_yt = req.youtube_url
    url_ig = req.instagram_url

    youtube_text = get_transcript(url_yt)
    youtube_text = clean_text(youtube_text)
    metadata_y = get_video_metadata(url_yt)

    with YoutubeDL({"cookiefile": "cookies.txt", "quiet": True}) as ydl:
        info = ydl.extract_info(url_ig, download=False)

    metadata_i = {
        "id": info.get("id"),
        "title": info.get("title"),
        "creator": info.get("uploader"),
        "views": info.get("view_count") or 0,
        "likes": info.get("like_count") or 0,
    }

    return {
        "video_a": {"video_id": extract_video_id(url_yt), "title": metadata_y["title"], "views": metadata_y["views"], "likes": metadata_y["likes"], "channel": metadata_y["channel"]},
        "video_b": {"video_id": metadata_i["id"], "title": metadata_i["title"], "views": metadata_i["views"], "likes": metadata_i["likes"], "creator": metadata_i["creator"]}
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    docs = hybrid_retriever.invoke(req.question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an AI Content Comparison Assistant.

YOUTUBE VIDEO METADATA:
- Title: {metadata_y['title']}
- Views: {metadata_y['views']}
- Likes: {metadata_y['likes']}
- Channel: {metadata_y['channel']}
- Duration: {metadata_y['duration']}

INSTAGRAM VIDEO METADATA:
- Title: {metadata_i['title']}
- Views: {metadata_i['views']}
- Likes: {metadata_i['likes']}
- Creator: {metadata_i['creator']}
- Duration: {metadata_i['duration']}

Context:
{context}

Question:
{req.question}

Comparison:
"""
    response = llm.invoke(prompt)
    return {"response": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)