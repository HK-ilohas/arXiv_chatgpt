import os

import chainlit as cl
import requests
from langchain.chains import (
    ConversationalRetrievalChain,
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import ArxivRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


model = ChatOpenAI(model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def initialize_chains():
    map_template = """
    以下の文章を要約してください。
    ただし、次のことに注意してください。
    - 簡潔に表現する
    - 不明な単語や人名と思われるものは英語のまま表示する
    
    入力:
    {docs}

    要約:
    """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=model, prompt=map_prompt)

    reduce_template = """
    以下は一連の文書です。日本語で要約してください。
    ただし、次のことに注意してください。
    - 簡潔に表現する
    - 不明な単語や人名と思われるものは英語のまま表示する
    
    入力:
    {docs}

    要約:
    """
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    return map_reduce_chain


async def handle_chat_start():
    retriever = ArxivRetriever(load_max_docs=1)
    docs = None

    while docs is None:
        res = await cl.AskUserMessage("arXivのURLを入力してください！").send()
        if res is None:
            continue
        url = res["output"]
        if not url.startswith(
            (
                "https://arxiv.org/abs/",
                "https://arxiv.org/pdf/",
                "http://arxiv.org/abs/",
                "http://arxiv.org/pdf/",
            )
        ):
            await cl.Message(
                "URLが正しい形式ではありません。`https://arxiv.org/abs/`または`https://arxiv.org/pdf/`の形式で入力してください。"
            ).send()
            continue

        paper_id = url.split("/")[-1].replace(".pdf", "")
        docs = retriever.get_relevant_documents(query=paper_id)

    return docs


async def process_document(docs):
    id = docs[0].metadata["Entry ID"].split("/")[-1]
    pdf_url = f"https://arxiv.org/pdf/{id}.pdf"
    response = requests.get(pdf_url)

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    with open(f"tmp/{id}.pdf", "wb") as f:
        f.write(response.content)

    pdf_docs = PyMuPDFLoader(f"tmp/{id}.pdf").load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(pdf_docs)

    return split_docs


@cl.on_chat_start
async def chat_start():
    docs = await handle_chat_start()
    split_docs = await process_document(docs)

    database = Chroma(embedding_function=embeddings)
    database.add_documents(split_docs)
    cl.user_session.set("database", database)

    map_reduce_chain = initialize_chains()
    res = map_reduce_chain.invoke(split_docs)

    msg = f"## 論文\nタイトル: {docs[0].metadata['Title']}\n著者: {docs[0].metadata['Authors']}\n## 要約\n{res['output_text']}"
    await cl.Message(msg).send()


@cl.on_message
async def main(message):
    database = cl.user_session.get("database")
    pdf_qa = ConversationalRetrievalChain.from_llm(
        model, database.as_retriever(), return_source_documents=True
    )

    result = pdf_qa({"question": message.content, "chat_history": chat_history})
    chat_history.append((message.content, result["answer"]))
    await cl.Message(result["answer"]).send()


chat_history = []
