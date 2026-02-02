import os
import uuid
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
from typing import Any, List
from llama_index.core import SimpleDirectoryReader
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig
)
from qdrant_client import QdrantClient, models



nest_asyncio.apply()              

load_dotenv()

class PdfProcessor:
    def __init__(self, llama_cloud_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY")):
        
        self.text_parser = LlamaParse(
            api_key=llama_cloud_api_key,
            disable_ocr=True,
            disable_image_extraction=True,
            output_tables_as_HTML=True)
        self.spacy_config = LanguageConfig(language="english", spacy_model="en_core_web_md")
        self.splitter = SemanticDoubleMergingSplitterNodeParser(
            language_config=self.spacy_config,
            initial_threshold=0.4,
            appending_threshold=0.5,
            merging_threshold=0.4,
            max_chunk_size=5000,
        )

    def load_documents(self, filename):
        return SimpleDirectoryReader(input_dir=[filename]).load_data(show_progress=True)
    
    def get_text_nodes(self, json_list: List[dict]) -> List[TextNode]:
        return [TextNode(text=page["text"], metadata={"page": page["page"]}) for page in json_list]

    def index_data(self, documents: List[TextNode]):
        nodes = self.splitter.get_nodes_from_documents(documents)
        return nodes

    def run(self, filename: str):
        json_objs = self.text_parser.get_json_result(filename)
        json_list = json_objs[0]["pages"]
        nodes = self.get_text_nodes(json_list)
        nodes = self.index_data(nodes)
        return nodes, os.path.basename(filename)


class QdrantUpload:
    def __init__(self, collection_name: str, openai_api_key: str = os.getenv("OPENAI_API_KEY")):
        self.collection_name = collection_name
        self.client = QdrantClient(
        host="qdrant",   
        port=6333)
        self.colbert_encoder = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
        self.bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    def get_openai_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Get embeddings using OpenAI's API."""
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(
            input=[text], 
            model=model
        ).data[0].embedding

    def _ensure_collection_exists(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-embedding-3-large": models.VectorParams(
                        size=3072, 
                        distance=models.Distance.COSINE
                    ),
                    "colbertv2.0": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        )
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
  
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WHITESPACE,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )

    def build_and_upload_point(self, nodes: Any, pdf_filename: str) -> str:
        """Build and upload a point to Qdrant."""

        # Generate embeddings
        for i in range(len(nodes)):
            combined_text = nodes[i].text
            dense_vector = self.get_openai_embedding(combined_text)
            colbert_vector = list(self.colbert_encoder.embed([combined_text]))[0]
            bm25_vector_data = list(self.bm25_encoder.embed([combined_text]))[0]
            sparse_vector = models.SparseVector(
                indices=bm25_vector_data.indices,
                values=bm25_vector_data.values
            )

            # Create and upload point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        payload={
                            "text": combined_text,
                            "page": nodes[i].metadata["page"],
                            "pdf": pdf_filename
                        },
                        vector={
                            "text-embedding-3-large": dense_vector,
                            "colbertv2.0": colbert_vector,
                            "bm25": sparse_vector
                        },
                    )
                ]
            )

    def run(self, nodes, pdf_filename: str):
        self._ensure_collection_exists()
        print("collection created")
        self.build_and_upload_point(nodes, pdf_filename)
        print("data uploaded")
    

class QdrantSearch:
    def __init__(self, collection_name: str, openai_api_key: str = os.getenv("OPENAI_API_KEY")):
        self.collection_name = collection_name
        self.client = QdrantClient(
            host="qdrant",   
            port=6333
        )
        self.bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    def get_openai_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Get embeddings using OpenAI's API."""
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(
            input=[text], 
            model=model
        ).data[0].embedding

    def collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            return "Collection does not exist!!"
        return True

    def multi_step_search(self, query_text: str) -> List[dict]:
        """Perform multi-step retrieval using dense, sparse, and late interaction vectors."""

        dense_vector = self.get_openai_embedding(query_text)
        bm25_vector_data = list(self.bm25_encoder.embed([query_text]))[0]

        sparse_vector = models.SparseVector(
            indices=bm25_vector_data.indices,
            values=bm25_vector_data.values
        )

        # Perform multi-step retrieval
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using="text-embedding-3-large",
                            limit=7,
                        )
                    ],
                    query=sparse_vector,
                    using="bm25",
                    limit=4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            score_threshold=0.3,
        )
        return results