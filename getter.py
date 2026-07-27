

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


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
# docs = retriever.invoke('What is the password of the day? Today is Tuesday.')
# for doc in docs:
#     print(f'{doc.metadata} {doc.page_content}')


model = ChatOllama(model='gemma4:e2b-it-qat')


# q = 'What is the password of the day?'
# docs = retriever.invoke(q)
# formatted_docs = '\n'.join([f'{doc.metadata} {doc.page_content}' for doc in docs])
# q = f'Answer the following question based on the context below:\n\nContext:\n{formatted_docs}\n\nQuestion: {q}'
# resp = model.stream(q)
# for chunk in resp:
#     print(chunk.text, end='')


# Would need to do prompt.invoke({'question': ..., 'context': ...}) and return the formatted string
prompt = PromptTemplate.from_template(
    'Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}'
)


def format_docs(docs):
    return '\n'.join([f'{doc.metadata} {doc.page_content}' for doc in docs])

# RunnablePassthrough() forwards the input `question` entry
# retriever is called with the entire dictionary
# Explicit is better than implicit :(
rag_chain = {'question': RunnablePassthrough(), 'context': retriever | format_docs}  \
    | prompt  \
    | model

resp = rag_chain.invoke({'question': 'What is the password of the day? Today is Tuesday.'})
print(resp.text)