import chainlit as cl

from langchain_community.retrievers import ArxivRetriever


@cl.on_chat_start
async def chat_start():
    await cl.Message("arXivのURLを入力してください！").send()


@cl.on_message
async def main(message):
    url = message.content
    if not (
        url.startswith("https://arxiv.org/abs/")
        or url.startswith("https://arxiv.org/pdf/")
        or url.startswith("http://arxiv.org/abs/")
        or url.startswith("http://arxiv.org/pdf/")
    ):
        await cl.Message(
            "URLが正しい形式ではありません。`https://arxiv.org/abs/`または`https://arxiv.org/pdf/`の形式で入力してください。"
        ).send()
        return
    paper_id = url.split("/")[-1]
    if paper_id.endswith(".pdf"):
        paper_id = paper_id[:-4]
    retriever = ArxivRetriever(load_max_docs=1)
    docs = retriever.get_relevant_documents(query=paper_id)
    await cl.Message(f"{docs[0].metadata}").send()
