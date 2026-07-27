

from langchain_chroma import Chroma
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from sentence_transformers import CrossEncoder


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
    def score(self, text_pairs):
        return self.model.predict(text_pairs)

def make_retriever(k: int):
    embedding = OllamaEmbeddings(model='embeddinggemma')
    vector_store = Chroma(
        collection_name='abcdef',
        embedding_function=embedding,
        persist_directory='dir'
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': k})
    return retriever

def make_reranker(top_n: int):
    cross_enc = SentenceTransformerCrossEncoder('/home/greateric/models/bge-reranker-base')
    reranker = CrossEncoderReranker(model=cross_enc, top_n=top_n)
    return reranker


q = 'What is the password of the day? Today is Tuesday.'
retriever = make_retriever(k=3)
docs = retriever.invoke(q)
for doc in docs:
    print(f'{doc.metadata} {doc.page_content}')
reranker = make_reranker(top_n=1)

docs = reranker.compress_documents(docs, query=q)
for doc in docs:
    print(f'{doc.metadata} {doc.page_content}')




# model = ChatOllama(model='gemma4:e2b-it-qat')
#
#
# q = 'What is the password of the day?'
# docs = retriever.invoke(q)
# formatted_docs = '\n'.join([f'{doc.metadata} {doc.page_content}' for doc in docs])
# q = f'Answer the following question based on the context below:\n\nContext:\n{formatted_docs}\n\nQuestion: {q}'
# resp = model.stream(q)
# for chunk in resp:
#     print(chunk.text, end='')






# # Would need to do prompt.invoke({'question': ..., 'context': ...}) and return the formatted string
# prompt = PromptTemplate.from_template(
#     'Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}'
# )
#
#
# def format_docs(docs):
#     return '\n'.join([f'{doc.metadata} {doc.page_content}' for doc in docs])
#
# # RunnablePassthrough() forwards the input `question` entry
# # retriever is called with the entire dictionary
# # Explicit is better than implicit :(
# rag_chain = {'question': RunnablePassthrough(), 'context': retriever | format_docs}  \
#     | prompt  \
#     | model
#
# resp = rag_chain.invoke({'question': 'What is the password of the day? Today is Tuesday.'})
# print(resp.text)