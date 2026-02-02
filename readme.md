# RAG Hybrid Search System

This project is a Retrieval-Augmented Generation (RAG) system capable of ingesting PDF documents, performing hybrid search (Dense + Sparse/BM25 + Late Interaction/ColBERT), and answering user queries using OpenAI's GPT models.

## Features

-   **PDF Ingestion**: Uses `LlamaParse` for high-quality PDF parsing (including tables).
-   **Advanced text Splitting**: Semantic Double Merging Splitter for better context retention.
-   **Hybrid Search**: Combines OpenAI embeddings, BM25 (Sparse), and ColBERT (Late Interaction) using Reciprocal Rank Fusion (RRF) in Qdrant.
-   **AI Agent**: `AnswwerCuratorAgent` uses GPT-4o-mini to synthesize answers strictly from retrieved content.
-   **FastAPI Backend**: Provides endpoints for document upload and querying.

## Prerequisites

-   Python 3.9+
-   Docker & Docker Compose (for Qdrant)
-   [LlamaCloud Account](https://cloud.llamaindex.ai/) (for parsing)
-   OpenAI API Key

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Start Qdrant (Vector Database):**
    Ensure you have `docker-compose.yml` set up or run Qdrant manually.
    
    ```bash
    docker-compose up -d
    ```
    *Note: The code expects Qdrant to be available at host `qdrant` on port `6333` (typical for Docker networking) or `localhost` if running locally outside a container. Check `qdrant_upload_retrieve.py` line 62/158 if you need to adjust hostnames.*

## Usage

### 1. Start the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 2. API Endpoints

#### Upload Document
**POST** `/api/upload`

Uploads a PDF, processes it, and indexes it into Qdrant.

-   **Parameters:**
    -   `file`: The PDF file (multipart/form-data).
    -   `collection_name`: Name for the vector collection (form-data).

#### Ask Question
**GET** `/api/final-answer`

Retrieves relevant context and answers a query.

-   **Parameters:**
    -   `query_text`: Your question.
    -   `collection_name`: The collection to search in.

### 3. Interactive Docs

Visit `http://localhost:8000/docs` to test the API endpoints using the Swagger UI.

## Project Structure

-   `main.py`: Entry point for the FastAPI application.
-   `routers/answer.py`: API routes for uploading documents and generating answers.
-   `answer_agent.py`: Logic for the AI agent that curates answers from retrieved text.
-   `qdrant_upload_retrieve.py`: Core logic for PDF processing, embedding generation, Qdrant indexing, and hybrid retrieval.
-   `requirements.txt`: Python package dependencies.
