from collections import defaultdict
from copy import deepcopy
from os import environ, getcwd
from posixpath import join as pathjoin
from typing import List, Optional, cast
from zlib import crc32

from chromadb import Embeddings, GetResult, QueryResult, chromadb
from grep_ast import filename_to_lang
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_voyageai import VoyageAIEmbeddings
from langchain_voyageai.embeddings import DEFAULT_VOYAGE_3_BATCH_SIZE
from tree_sitter import Language, Node, Parser, Tree
from tree_sitter_language_pack import get_parser

from aider.io import InputOutput


class ASTChunker:
    def __init__(self, parser: Parser):
        self.parser = parser
        self.language: Language = getattr(parser, "language")

    def _chunk_tree(self, file_tree: Tree) -> dict[str, list[str]]:
        grouped_node_type_chunks: dict[str, list[str]] = defaultdict(list[str])
        cursor = file_tree.walk()
        current_node: Node
        node_text: bytes

        if cursor.goto_first_child():
            current_node = getattr(cursor, "node")
            node_text = getattr(current_node, "text")
            grouped_node_type_chunks[current_node.type].append(bytes.decode(node_text))

            while cursor.goto_next_sibling():
                current_node = getattr(cursor, "node")
                node_text = getattr(current_node, "text")
                grouped_node_type_chunks[current_node.type].append(
                    bytes.decode(node_text)
                )

        return grouped_node_type_chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            start_index = 0
            tree = self.parser.parse(bytes(text, "utf-8"))

            for grammar_type, grouped_chunks in self._chunk_tree(tree).items():
                metadata = deepcopy(_metadatas[i])
                metadata["grammar_type"] = grammar_type
                grouped_text = "\n".join(grouped_chunks)
                metadata = metadata
                new_doc = Document(page_content=grouped_text, metadata=metadata)
                documents.append(new_doc)
                start_index += len(grouped_chunks)
        return documents


class RagManager:
    def __init__(self, io: InputOutput):
        if environ.get("VOYAGE_API_KEY") is None:
            raise EnvironmentError("Voyage.ai api key environment variable not set.")

        self._io = io
        self._voyage_embeddings = VoyageAIEmbeddings(
            model="voyage-3.5", batch_size=DEFAULT_VOYAGE_3_BATCH_SIZE
        )
        self._chromadb_client = chromadb.PersistentClient(
            path=pathjoin(getcwd(), ".aider.chroma")
        )
        self._chromadb_collection = self._chromadb_client.get_or_create_collection(
            name="aider-rag", embedding_function=None
        )

    def _get_chunk_ids(self, chunks: list[Document]) -> list[str]:
        ids: list[str] = []

        for i, doc in enumerate(chunks):
            file_name: str = doc.metadata["file_name"]
            grammar_type: str | None = doc.metadata.get("grammar_type")
            if grammar_type is None:
                ids.append(f"{file_name}#semantic_chunk_{i}")
            else:
                ids.append(f"{file_name}#{grammar_type}")

        return ids

    def _store_embeddings(self, chunks: list[Document], embeddings: list[list[float]]):
        return self._chromadb_collection.upsert(
            ids=self._get_chunk_ids(chunks),
            documents=[doc.page_content for doc in chunks],
            metadatas=[doc.metadata for doc in chunks],
            embeddings=cast(Embeddings, embeddings),
        )

    def _retrieve_embeddings(
        self, embeddings: list[list[float]], file_names: list[str]
    ):
        results: list[QueryResult] = []
        for fname in file_names:
            results.append(
                self._chromadb_collection.query(
                    query_embeddings=cast(Embeddings, embeddings),
                    where={"file_name": fname},
                )
            )
        return results

    def _get_stored_chunks(self, fname: str):
        return self._chromadb_collection.get(where={"file_name": fname})

    def _get_stored_metadata(self, stored_chunk: GetResult, metadata_key: str):
        stored_chunk_metadatas = stored_chunk.get("metadatas")
        if stored_chunk_metadatas is None or len(stored_chunk_metadatas) == 0:
            return None
        return stored_chunk_metadatas[0].get(metadata_key)

    def _decide_chunker(self, fname: str):
        detected_language = filename_to_lang(fname)
        try:
            supported_parser = get_parser(detected_language)  # type: ignore
            return (ASTChunker(supported_parser), "ASTChunker")
        except Exception:
            pass
        return (SemanticChunker(self._voyage_embeddings), "SemanticChunker")

    def chunk_files(self, file_names: list[str]) -> list[list[Document]]:
        all_chunks: list[list[Document]] = []
        for fname in file_names:
            content = self._io.read_text(fname)
            if not content:
                self._io.tool_output(f"File {fname} is empty.")
                continue

            stored_file_chunks = self._get_stored_chunks(fname)
            stored_crc32_hash = self._get_stored_metadata(
                stored_file_chunks, "crc32_hash"
            )
            file_crc32_hash = crc32(content.encode("utf-8"))

            if file_crc32_hash == stored_crc32_hash:
                continue

            chunker, chunker_name = self._decide_chunker(fname)
            self._io.tool_output(f"Chunking {fname} with {chunker_name}")

            file_chunks = chunker.create_documents(
                [content], [{"file_name": fname, "crc32_hash": file_crc32_hash}]
            )
            all_chunks.append(file_chunks)

        return all_chunks

    def embed_store_chunks(self, all_chunks: list[list[Document]]):
        for file_chunks in all_chunks:
            fname = file_chunks[0].metadata["file_name"]
            self._io.tool_output(f"Embedding and storing {fname}")
            file_chunk_ids = self._get_chunk_ids(file_chunks)

            self._chromadb_collection.delete(ids=file_chunk_ids)

            texts = [doc.page_content for doc in file_chunks]
            embeddings = self._voyage_embeddings.embed_documents(texts)
            self._store_embeddings(file_chunks, embeddings)
        return None

    def embed_retrieve_query(self, query: str, file_names: list[str]):
        embedded_query = self._voyage_embeddings.embed_query(query)
        retrieved_results = self._retrieve_embeddings([embedded_query], file_names)

        return retrieved_results
