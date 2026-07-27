

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document





docs = []

docs.append(Document(
    page_content='Here is some random information. The password of the day is 6767.',
    metadata={'date': 'Monday'}
))
docs.append(Document(
    page_content='Here is some random information. The password of the day is lionfields.',
    metadata={'date': 'Tuesday'}
))


embedding = OllamaEmbeddings(model='embeddinggemma')
vector_store = Chroma(
    collection_name='abcdef',
    embedding_function=embedding,
    persist_directory='dir'
)
vector_store.add_documents(docs)

