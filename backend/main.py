# main.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_engine import load_and_index_documents, search_documents
from model import extract_query_info, get_llm_decision
from langchain_community.vectorstores import FAISS
import os
import shutil
# import uvicorn

# app = FastAPI()
app = FastAPI(
    docs_url="/",       # Swagger UI available at root "/"
    redoc_url=None,     # Disable ReDoc (optional)
    openapi_url="/openapi.json"  # (Optional, keep default OpenAPI schema)
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


# import os

# port = int(os.environ.get("PORT", 8000))  # Fallback to 8000 for local dev
# uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


UPLOAD_DIR = "data/docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic model for query input
class QueryInput(BaseModel):
    query: str


# ✅ Upload and index insurance documents (PDF/DOCX)
@app.post("/upload")
def upload_docs(files: list[UploadFile] = File(...)):
    try:
        uploaded = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded.append(file_path)
        print(f"✅ Uploaded files: {uploaded}")
        load_and_index_documents(UPLOAD_DIR)
        return {"message": "Files uploaded and indexed successfully."}
    except Exception as e:
        print("❌ UPLOAD ERROR:", e)
        return {"error": "File upload or indexing failed", "details": str(e)}


# ✅ Analyze user query against indexed policy documents
@app.post("/analyze")
def analyze_query(input: QueryInput):
    try:
        query = input.query
        print(f"🔍 Received query: {query}")
        extracted = extract_query_info(query)
        docs = search_documents(query)
        result = get_llm_decision(query, docs, extracted)
        return result
    except Exception as e:
        print("❌ ANALYSIS ERROR:", e)
        return {"error": "Query analysis failed", "details": str(e)}
