from collections import defaultdict
from copy import deepcopy
from os import getcwd
from posixpath import join as pathjoin
from typing import List, Optional, cast
from zlib import crc32

from chromadb import Embeddings, GetResult, QueryResult, chromadb
from grep_ast import filename_to_lang
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain_voyageai.embeddings import DEFAULT_VOYAGE_3_BATCH_SIZE
from numpy import percentile
from tree_sitter import Language, Node, Parser, Tree
from tree_sitter_language_pack import get_parser

from aider.io import InputOutput

from functools import cached_property


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
    @cached_property
    def voyage_embeddings():
        return VoyageAIEmbeddings(
            model="voyage-3.5", batch_size=DEFAULT_VOYAGE_3_BATCH_SIZE
        )

    @cached_property
    def voyage_rerank():
        return VoyageAIRerank(model="rerank-2.5")

    @cached_property
    def chromadb_collection():
        return chromadb.PersistentClient(
            path=pathjoin(getcwd(), ".aider.chroma")
        ).get_or_create_collection(name="aider-rag", embedding_function=None)

    @staticmethod
    def _get_chunk_ids(chunks: list[Document]) -> list[str]:
        ids: list[str] = []

        for i, doc in enumerate(chunks):
            file_name: str = doc.metadata["file_name"]
            grammar_type: str | None = doc.metadata.get("grammar_type")
            if grammar_type is None:
                ids.append(f"{file_name}#semantic_chunk_{i}")
            else:
                ids.append(f"{file_name}#{grammar_type}")

        return ids

    @staticmethod
    def _store_embeddings(chunks: list[Document], embeddings: list[list[float]]):
        return RagManager.chromadb_collection.func().upsert(
            ids=RagManager._get_chunk_ids(chunks),
            documents=[doc.page_content for doc in chunks],
            metadatas=[doc.metadata for doc in chunks],
            embeddings=cast(Embeddings, embeddings),
        )

    @staticmethod
    def _retrieve_embedding(
        embedding: list[float], file_names: list[str]
    ) -> QueryResult:
        return RagManager.chromadb_collection.func().query(
            query_embeddings=embedding,
            where={"file_name": {"$in": file_names}},  # type: ignore // this is correct, but the typing is bugged
            n_results=len(file_names)
            * 10,  # the retrieval will be reranked, so it's fine if there are a lot of results
        )

    @staticmethod
    def _rerank_retrieved_results(query: str, results: QueryResult):
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

        return RagManager.voyage_rerank.func().compress_documents(documents, query)

    @staticmethod
    def _get_stored_chunks(fname: str) -> GetResult:
        return RagManager.chromadb_collection.func().get(where={"file_name": fname})

    @staticmethod
    def _get_stored_metadata(stored_chunk: GetResult, metadata_key: str):
        stored_chunk_metadatas = stored_chunk.get("metadatas")
        if stored_chunk_metadatas is None or len(stored_chunk_metadatas) == 0:
            return None
        return stored_chunk_metadatas[0].get(metadata_key)

    @staticmethod
    def _decide_chunker(fname: str):
        detected_language = filename_to_lang(fname)
        try:
            supported_parser = get_parser(detected_language)  # type: ignore
            return (ASTChunker(supported_parser), "ASTChunker")
        except Exception:
            pass
        return (SemanticChunker(RagManager.voyage_embeddings.func()), "SemanticChunker")

    @staticmethod
    def chunk_files(
        io: InputOutput, file_names: list[str]
    ) -> tuple[list[list[Document]], list[str]]:
        all_chunks: list[list[Document]] = []
        changed_file_names: list[str] = []
        for fname in file_names:
            content = io.read_text(fname)
            if not content:
                io.tool_output(f"File {fname} is empty.")
                continue

            stored_file_chunks = RagManager._get_stored_chunks(fname)
            stored_crc32_hash = RagManager._get_stored_metadata(
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

            chunker, chunker_name = RagManager._decide_chunker(fname)
            io.tool_output(f"Chunking {fname} with {chunker_name}")

            file_chunks = chunker.create_documents(
                [content], [{"file_name": fname, "crc32_hash": file_crc32_hash}]
            )
            all_chunks.append(file_chunks)
            changed_file_names.append(fname)

        return all_chunks, changed_file_names

    @staticmethod
    def embed_store_chunks(
        io: InputOutput, all_chunks: list[list[Document]], changed_file_names: list[str]
    ):
        for file_chunks in all_chunks:
            fname = file_chunks[0].metadata["file_name"]
            if fname in changed_file_names:
                io.tool_output(f"Embedding and storing {fname}")
                file_chunk_ids = RagManager._get_chunk_ids(file_chunks)

                RagManager.chromadb_collection.func().delete(ids=file_chunk_ids)

                texts = [doc.page_content for doc in file_chunks]
                embeddings = RagManager.voyage_embeddings.func().embed_documents(texts)
                RagManager._store_embeddings(file_chunks, embeddings)
        return None

    @staticmethod
    def embed_retrieve_query(
        query: str, file_names: list[str], top_k_percentile: float
    ):
        embedded_query = RagManager.voyage_embeddings.func().embed_query(query)
        retrieved_results = RagManager._retrieve_embedding(embedded_query, file_names)

        reranked_results = RagManager._rerank_retrieved_results(
            query, retrieved_results
        )

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
