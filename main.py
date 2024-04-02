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

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

retriever = ArxivRetriever(load_max_docs=1)

map_template = """以下の文章を要約してください。
ただし、以下のルールに従ってください。

- 簡潔に表現する
- 不明な単語や人名と思われるものは英語のまま表示する

それでは開始します。:

{docs}

要約:"""
map_template = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=model, prompt=map_template)

# Reduce
reduce_template = """以下は一連の文書です:

{docs}

このドキュメントのリストに基づいて、簡潔にわかりやすく日本語で要約してください。
ただし、以下のルールに従ってください。

- 簡潔に表現する
- 不明な単語や人名と思われるものは英語のまま表示する

要約:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Run chain
reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

chat_history = []


@cl.on_chat_start
async def chat_start():
    docs = None
    while docs is None:
        res = await cl.AskUserMessage("arXivのURLを入力してください！").send()
        if res is None:
            continue
        url = res["output"]
        if not (
            url.startswith("https://arxiv.org/abs/")
            or url.startswith("https://arxiv.org/pdf/")
            or url.startswith("http://arxiv.org/abs/")
            or url.startswith("http://arxiv.org/pdf/")
        ):
            await cl.Message(
                "URLが正しい形式ではありません。`https://arxiv.org/abs/`または`https://arxiv.org/pdf/`の形式で入力してください。"
            ).send()
        else:
            paper_id = url.split("/")[-1]
            if paper_id.endswith(".pdf"):
                paper_id = paper_id[:-4]
            docs = retriever.get_relevant_documents(query=paper_id)
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
    database = Chroma(embedding_function=embeddings)
    database.add_documents(split_docs)
    cl.user_session.set("database", database)

    res = map_reduce_chain.invoke(split_docs)

    msg = f"""## 論文
タイトル: {docs[0].metadata['Title']}
著者: {docs[0].metadata['Authors']}
## 要約
{res['output_text']}"""

    await cl.Message(msg).send()
    return


@cl.on_message
async def main(message):
    database = cl.user_session.get("database")
    pdf_qa = ConversationalRetrievalChain.from_llm(
        model, database.as_retriever(), return_source_documents=True
    )
    result = pdf_qa({"question": message.content, "chat_history": chat_history})
    chat_history.append((message.content, result["answer"]))
    await cl.Message(result["answer"]).send()
