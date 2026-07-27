

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def make_retriever(k: int):
    embedding = OllamaEmbeddings(model='embeddinggemma')
    vector_store = Chroma(
        collection_name='abcdef',
        embedding_function=embedding,
        persist_directory='dir'
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': k})
    return retriever

retriever = make_retriever(k=3)
docs = retriever.invoke('What is the password of the day? Today is Tuesday.')

for doc in docs:
    print(f'{doc.metadata} {doc.page_content}')

