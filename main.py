import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
print('start loading embedding......', end='')
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={'normalize_embeddings': False},
    multi_process=False
)
print('done')
print('start loading text_splitter,llm,memory,fastapi......', end='')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", "\n", ""]
)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法：GET, POST 等
    allow_headers=["*"],  # 允许所有头信息
)
print('done')


def document_adapter_docx(source_dir: str):
    loader = Docx2txtLoader(source_dir)
    document = loader.load()
    return document[0]


def create_db():
    global db
    all_docs = []
    files = os.listdir('docs')
    for file in files:
        if file.endswith('.docx'):
            all_docs.append(document_adapter_docx(os.path.join('docs', file)))
    all_splits = text_splitter.split_documents(all_docs)
    db = FAISS.from_documents(documents=all_splits, embedding=embeddings)
    db.save_local(folder_path="./data/faiss_db")
    return db


def load_db():
    global db
    db = FAISS.load_local(
        folder_path="./data/faiss_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return db


if not os.path.exists("./data/faiss_db"):
    print('start creating db......', end='')
    db = create_db()
    print('done')
else:
    print('start loading db......', end='')
    db = load_db()
    print('done')


# 获取历史对话列表
def get_memory_messages():
    messages = []
    for message in memory.chat_memory.messages:
        message_dict = {}
        if type(message) == HumanMessage:
            message_dict["Human"] = message.content
        elif type(message) == AIMessage:
            message_dict["AI"] = message.content
        messages.append(message_dict)
    return messages


@app.post("/chat")
async def chat(request: Request):
    json_data = await request.json()
    prompt = json_data.get("prompt")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_type='mmr'),
        memory=memory,
        output_key="answer",  # 明确指定输出键
        return_source_documents=True
    )
    # 获取链的结果
    result = qa_chain({"question": prompt})
    answer = result["answer"]

    source_documents = result["source_documents"]

    # 加入参考资料
    references = "\n".join(
        [
            f"{i + 1}、《{os.path.basename(doc.metadata.get('source', '未知文档'))}》\n "
            + re.sub(r"\s+", " ", doc.page_content)[:200] + "..."
            for i, doc in enumerate(source_documents)
        ]
    )

    # 返回响应内容，包含回答和参考资料
    response_message = {
        "AI": f"{answer}\n\n--------------------\n参考资料:\n{references}"
    }
    if memory.chat_memory.messages and isinstance(memory.chat_memory.messages[-1], AIMessage):
        memory.chat_memory.messages[-1] = AIMessage(content=response_message["AI"])
    else:
        memory.chat_memory.add_message(AIMessage(content=response_message["AI"]))
    return JSONResponse(content=get_memory_messages())


# 获取历史对话列表
@app.get("/message")
async def get_message():
    return JSONResponse(content=get_memory_messages())


# 获取历史对话列表
@app.get("/clear")
async def clear_message():
    memory.clear()


# 运行后端服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
