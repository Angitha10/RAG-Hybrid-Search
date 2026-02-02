from fastapi import APIRouter, Form
from qdrant_upload_retrieve import PdfProcessor, QdrantUpload, QdrantSearch
from answer_agent import AnswwerCuratorAgent
import os, shutil, tempfile
from fastapi import UploadFile, File
from pydantic import BaseModel
answer_router = APIRouter()


class SearchRequest(BaseModel):
    query_text: str
    collection_name: str = "docs-parser"


processor = PdfProcessor()

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "pdf_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@answer_router.post("/upload")
async def upload_pdf(
    collection_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload and process a PDF file, then upload to Qdrant.
    
    Args:
        collection_name: Name of the Qdrant collection (required)
        file: The PDF file to upload (required)
    """
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the uploaded file
    nodes, pdf_filename = processor.run(file_location)
    
    # Upload to Qdrant
    uploader = QdrantUpload(collection_name=collection_name)
    uploader.run(nodes, pdf_filename)
    
    return {
        "info": f"file '{file.filename}' processed and uploaded successfully",
        "collection_name": collection_name,
        "pdf_filename": pdf_filename
    }


@answer_router.get("/final-answer")
async def answer(query_text: str, collection_name: str):
    """
    Get an answer based on the query text from the specified collection.
    
    Args:
        query_text: The query to search for
        collection_name: Name of the Qdrant collection
    """
    searcher = QdrantSearch(collection_name=collection_name)
    
    # Check if collection exists
    collection_check = searcher.collection_exists()
    if collection_check == "Collection does not exist!!":
        return {"error": f"Collection '{collection_name}' does not exist"}
    
    results = searcher.multi_step_search(query_text)
    
    ans = AnswwerCuratorAgent(retrieved_content=results)
    final_ans= await ans.answer(query_text)
    return final_ans


