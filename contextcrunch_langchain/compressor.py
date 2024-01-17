from typing import Sequence
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.utils.utils import convert_to_secret_str
from langchain_core.utils import (
    check_package_version,
    get_from_env,
)
from langchain_core.documents import Document
from contextcrunch import ContextCrunchClient

class ContextCrunchDocumentCompressor(BaseDocumentCompressor):
    compression_ratio: float = 0.9
    client: ContextCrunchClient = None
    
    def __init__(self, compression_ratio=0.9):
        # super().__init__(compression_ratio=compression_ratio)
        super().__init__()
        self.compression_ratio = compression_ratio
        contextcrunch_api_key = convert_to_secret_str(
            get_from_env( "contextcrunch_api_key", "CONTEXTCRUNCH_API_KEY")
        )
        # Get custom api url from environment.
        contextcrunch_api_url = get_from_env(
            "contextcrunch_api_url",
            "CONTEXTCRUNCH_API_URL",
            default="https://contextcrunch.com/api",
        )
        try:
            import contextcrunch

            check_package_version("contextcrunch", gte_version="1.0.2")
            self.client = contextcrunch.ContextCrunchClient(
                api_key=contextcrunch_api_key.get_secret_value(),
                url=contextcrunch_api_url
            )

        except ImportError:
            raise ImportError(
                "Could not import contextcrunch python package. "
                "Please it install it with `pip install contextcrunch`."
            )
        
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks = None,
    ) -> Sequence[Document]:
        """
        Using contextcrunch, compresses a sequence of documents into a (highly compressed) single document.
        Depends on the compression ratio
        """
        doc_content = [doc.page_content for doc in documents]
        compressed = self.client.compress(doc_content, query, type='rag')
        new_document = Document(page_content=compressed)
        return [new_document]
    
    class Config:
        arbitrary_types_allowed = True