from os import getcwd
from posixpath import join as pathjoin
from typing import cast
from zlib import crc32

from chromadb import Embeddings, GetResult, Metadata, PersistentClient, QueryResult
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from numpy import percentile

from aider.io import InputOutput

CONTEXTUALIZED_EMBEDDINGS_MODEL = "voyage-context-3"
RERANKER_MODEL = "rerank-2.5"
CHROMADB_LOCAL_PATH = ".aider.chroma"
CHROMADB_COLLECTION_NAME = "aider-rag"
MAX_CONTEXTUALIZED_EMBEDDINGS_BATCH_SIZE = 1000
RERANKER_MODEL_SAFE_BATCH_SIZE = 100


class RagManager:
    def __init__(self, io: InputOutput):
        self.embed_client = VoyageAIEmbeddings(
            model=CONTEXTUALIZED_EMBEDDINGS_MODEL,
            batch_size=MAX_CONTEXTUALIZED_EMBEDDINGS_BATCH_SIZE,
        )
        if self.embed_client._is_context_model() is False:
            raise Exception(
                f"The embedding model '{CONTEXTUALIZED_EMBEDDINGS_MODEL}' is not a valid contextualized embedding model."
            )

        self.rerank_client = VoyageAIRerank(model=RERANKER_MODEL)
        self.chromadb_collection = PersistentClient(
            path=pathjoin(getcwd(), CHROMADB_LOCAL_PATH)
        ).get_or_create_collection(
            name=CHROMADB_COLLECTION_NAME, embedding_function=None
        )
        self.io = io

    def _get_file_chunk_ids(self, file_chunks: list[Document]) -> list[str]:
        file_name: str = file_chunks[0].metadata["file_name"]
        chunk_count = len(file_chunks)

        return [f"{file_name}#chunk_{i + 1}/{chunk_count}" for i in range(chunk_count)]

    def _store_file_chunks(
        self, file_chunks: list[Document], file_chunks_embeddings: list[list[float]]
    ):
        ids = self._get_file_chunk_ids(file_chunks)
        metadatas = [doc.metadata for doc in file_chunks]
        for i, metadata in enumerate(metadatas):
            metadata["chunk_number"] = ids[i].split("#")[1]

        return self.chromadb_collection.upsert(
            ids=ids,
            documents=[doc.page_content for doc in file_chunks],
            metadatas=cast(list[Metadata], metadatas),
            embeddings=cast(Embeddings, file_chunks_embeddings),
        )

    def _retrieve_embedding(
        self, embedding: list[float], file_names: list[str]
    ) -> QueryResult:
        return self.chromadb_collection.query(
            query_embeddings=embedding,
            where={"file_name": {"$in": file_names}},  # type: ignore // this is correct, but the typing is bugged
            n_results=len(file_names)
            * 10,  # the retrieval will be reranked, so it's fine if there are a lot of results
        )

    def _rerank_retrieved_results(self, query: str, results: QueryResult):
        document_contents = results.get("documents")
        document_metadatas = results.get("metadatas")
        if not document_contents or not document_metadatas:
            raise KeyError("Retrieval results are empty!")

        documents: list[Document] = []
        for chunk_list_i, chunk_list in enumerate(document_contents):
            for chunk_i, chunk_content in enumerate(chunk_list):
                documents.append(
                    Document(
                        chunk_content,
                        metadata=document_metadatas[chunk_list_i][chunk_i],
                    )
                )

        reranked_documents: list[Document] = []
        for i in range(0, len(documents), RERANKER_MODEL_SAFE_BATCH_SIZE):
            batch = documents[i : i + RERANKER_MODEL_SAFE_BATCH_SIZE]
            reranked_documents.extend(
                (self.rerank_client.compress_documents(batch, query))
            )
        return reranked_documents

    def _get_stored_chunks(self, fname: str) -> GetResult:
        return self.chromadb_collection.get(where={"file_name": fname})

    def _get_stored_metadata(self, stored_chunk: GetResult, metadata_key: str):
        stored_chunk_metadatas = stored_chunk.get("metadatas")
        if stored_chunk_metadatas is None or len(stored_chunk_metadatas) == 0:
            return None
        return stored_chunk_metadatas[0].get(metadata_key)

    def chunk_files(
        self, file_names: list[str]
    ) -> tuple[list[list[Document]], list[str]]:
        all_chunks: list[list[Document]] = []
        changed_file_names: list[str] = []
        for fname in file_names:
            content = self.io.read_text(fname)
            if not content:
                self.io.tool_output(f"File {fname} is empty.")
                continue

            stored_file_chunks = self._get_stored_chunks(fname)
            stored_crc32_hash = self._get_stored_metadata(
                stored_file_chunks, "crc32_hash"
            )
            file_crc32_hash = crc32(content.encode("utf-8"))

            if file_crc32_hash == stored_crc32_hash:
                stored_documents = [
                    Document(content, metadata=chunk_metadata)
                    for content, chunk_metadata in zip(
                        stored_file_chunks.get("documents") or [],
                        stored_file_chunks.get("metadatas") or [],
                    )
                ]
                all_chunks.append(stored_documents)
                continue

            self.io.tool_output(f"Chunking {fname}")

            file_chunks = RecursiveCharacterTextSplitter(
                separators=["\n"]
            ).create_documents(
                [content],
                [{"file_name": fname, "crc32_hash": file_crc32_hash}],
            )
            all_chunks.append(file_chunks)
            changed_file_names.append(fname)

        return all_chunks, changed_file_names

    def embed_files(
        self, files_chunks_list: list[list[Document]], changed_file_names: list[str]
    ):
        for file_chunks in files_chunks_list:
            fname = file_chunks[0].metadata["file_name"]
            if fname in changed_file_names:
                self.io.tool_output(f"Embedding and storing {fname}")
                file_chunk_ids = self._get_file_chunk_ids(file_chunks)
                self.chromadb_collection.delete(ids=file_chunk_ids)
                texts = [doc.page_content for doc in file_chunks]
                embeddings = self.embed_client.embed_documents(texts)
                self._store_file_chunks(file_chunks, embeddings)

        return None

    def retrieve(self, query: str, file_names: list[str], top_k_percentile: float):
        embedded_query = self.embed_client.embed_query(query)
        retrieved_results = self._retrieve_embedding(embedded_query, file_names)

        reranked_results = self._rerank_retrieved_results(query, retrieved_results)

        relevance_score_percentile = percentile(
            [result.metadata["relevance_score"] for result in reranked_results],
            top_k_percentile,
        )

        results_at_percentile: list[Document] = list(
            filter(
                lambda result: result.metadata["relevance_score"]
                >= relevance_score_percentile,
                reranked_results,
            )
        )

        return results_at_percentile
