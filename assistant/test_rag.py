import os
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding

# 1. 加载环境变量
load_dotenv()

# 2. 导入 LangChain 相关模块
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from typing import List

# 3. 配置通义千问的 Key
dashscope.api_key = os.getenv("OPENAI_API_KEY")

# 4. 定义文档路径
FILE_PATH = "test.pdf" 

# 5. 自定义 Embeddings 类
class QwenEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        print(f"正在处理 {len(texts)} 个文本块...")
        embeddings = []
        # 关键修改：batch_size 改为 10
        batch_size = 10 
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"正在处理第 {i} - {i+len(batch)} 个块...")
            
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=batch,
                text_type='document'
            )
            
            if resp.status_code != 200:
                print(f"API 错误详情: {resp}") # 打印详细错误
                raise Exception(f"API Error: {resp.message}")
                
            for item in resp.output['embeddings']:
                embeddings.append(item['embedding'])
        
        print("所有文本块向量化完成。")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input=text,
            text_type='query'
        )
        if resp.status_code != 200:
            raise Exception(f"API Error: {resp.message}")
        return resp.output['embeddings'][0]['embedding']

def main():
    print("--- 开始处理文档 ---")
    
    # A. 加载文档
    if not os.path.exists(FILE_PATH):
        print(f"错误：找不到文件 {FILE_PATH}，请确认文件名是否正确。")
        return

    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    print(f"加载了 {len(documents)} 页文档")

    # B. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # 提取纯文本内容
    clean_texts = [t.page_content for t in texts]
    print(f"切分后共 {len(clean_texts)} 个文本块")

    # C. 向量化
    print("正在生成向量并构建索引（使用通义千问原生 Embedding）...")
    embeddings = QwenEmbeddings()
    
    db = FAISS.from_texts(clean_texts, embeddings)
    print("向量数据库构建完成")

    # D. 创建问答链
    print("正在加载通义千问模型...")
    llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    # E. 进行测试
    print("\n--- 准备就绪，开始提问 ---")
    query = "请总结一下这份文档的主要内容？"
    
    response = qa_chain.invoke(query)
    
    print(f"问题: {query}")
    print(f"回答: {response['result']}")

if __name__ == "__main__":
    main()
