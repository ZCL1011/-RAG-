import gradio as gr
import os
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from typing import List

# --- 1. 基础配置 ---
load_dotenv()
dashscope.api_key = os.getenv("OPENAI_API_KEY")

# 全局变量
current_db = None
current_llm = None
current_filename = None

# --- 2. 定义通义千问 Embeddings 类 ---
class QwenEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 10 
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=batch,
                text_type='document'
            )
            if resp.status_code != 200:
                raise Exception(f"API Error: {resp.message}")
            for item in resp.output['embeddings']:
                embeddings.append(item['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input=text,
            text_type='query'
        )
        if resp.status_code != 200:
            raise Exception(f"API Error: {resp.message}")
        return resp.output['embeddings'][0]['embedding']

# --- 3. 核心逻辑函数 ---

def process_file(file):
    global current_db, current_llm, current_filename
    
    if file is None:
        return gr.update(visible=False), "请先上传一个 PDF 文件。"
    
    try:
        file_path = file.name
        current_filename = os.path.basename(file_path)
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        clean_texts = [t.page_content for t in texts]

        embeddings = QwenEmbeddings()
        current_db = FAISS.from_texts(clean_texts, embeddings)

        # 初始化 LLM (注意：这里只初始化 LLM，不初始化 QA Chain，因为我们要手动控制流式输出)
        current_llm = ChatOpenAI(
            model="qwen-turbo",
            temperature=0,
            streaming=True, # 开启流式支持
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )
        
        return gr.update(visible=True), f"✅ 成功加载文档：{current_filename}，现在可以提问了！"
        
    except Exception as e:
        return gr.update(visible=False), f"❌ 处理文件时出错：{str(e)}"

def respond(message, history):
    """
    处理用户提问，支持流式输出和引用归因
    """
    global current_db, current_llm
    
    if current_db is None or current_llm is None:
        yield "请先在左侧上传 PDF 文档。"
        return

    # 1. 检索相关文档
    retriever = current_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(message)
    
    # 2. 构建上下文 (Context)
    context = "\n\n".join([d.page_content for d in docs])
    
    # 3. 构建 Prompt
    # 我们明确告诉 AI：仅根据上下文回答
    prompt = f"""
    请根据下面的已知信息回答问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题"，不允许编造答案。
    
    已知信息:
    {context}
    
    问题:
    {message}
    """

    # 4. 流式生成回答
    # 使用 current_llm.stream 获取生成器
    full_response = ""
    for chunk in current_llm.stream(prompt):
        full_response += chunk.content
        # 实时 yield 给 Gradio 界面
        yield full_response

    # 5. 添加引用归因
    # 在回答结束后，附上检索到的原文片段
    sources = "\n\n**参考来源：**\n"
    for i, doc in enumerate(docs):
        # 截取部分内容作为预览，避免太长
        preview = doc.page_content[:100].replace("\n", " ") + "..."
        sources += f"\n📄 **片段 {i+1}**: {preview}\n"
    
    # 最终输出包含回答 + 来源
    yield full_response + sources

# --- 4. 创建 Gradio 界面 ---

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 本地知识库智能助手 (流式+引用)")
    
    with gr.Row():
        # 左侧栏
        with gr.Column(scale=1):
            file_input = gr.File(label="上传 PDF 文档", file_types=[".pdf"])
            status_box = gr.Textbox(label="状态", interactive=False, value="等待上传文件...")
            upload_btn = gr.Button("加载文档", variant="primary")
            
            upload_btn.click(
                fn=process_file,
                inputs=file_input,
                outputs=[gr.State(), status_box]
            )
        
        # 右侧栏：聊天界面
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                title="与文档对话",
                description="上传文档后，点击左侧“加载文档”按钮，即可开始提问。回答将包含参考来源。"
            )

if __name__ == "__main__":
    demo.launch()
