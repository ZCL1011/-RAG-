# 🤖 本地知识库智能问答系统 (RAG)

一个基于 LangChain 和通义千问 (Qwen) 的本地知识库问答助手。该项目利用 RAG (Retrieval-Augmented Generation) 技术，能够读取用户上传的 PDF 文档，通过语义检索精准定位信息，并生成带有引用来源的流式回答。

## ✨ 功能特性

- **📄 智能文档解析**：支持上传 PDF 文档，自动进行文本切分与向量化。
- **🔍 语义精准检索**：基于 FAISS 向量数据库，实现毫秒级的语义相似度搜索。
- **🧠 RAG 增强生成**：结合通义千问大模型，仅基于文档内容回答问题，有效抑制“幻觉”。
- **⚡ 流式输出体验**：支持打字机效果，实时展示生成过程，交互体验流畅。
- **📚 引用归因溯源**：在回答下方自动列出参考的原文片段及页码，提升回答的可信度。

## 🛠️ 技术栈

- **核心架构**：RAG (Retrieval-Augmented Generation)
- **开发语言**：Python 3.10+
- **LLM 框架**：LangChain
- **大模型接口**：阿里云通义千问
- **向量数据库**：FAISS (Facebook AI Similarity Search)
- **Web 框架**：Gradio
- **文档处理**：PyPDF2, Unstructured

## 📦 安装指南

### 1. 环境要求
- Python 3.10 或更高版本
- Conda (推荐) 或 Pip

### 2. 创建并激活虚拟环境

使用 Conda (推荐):
```bash
conda create -n rag_assistant python=3.10
conda activate rag_assistant
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录下创建 `.env` 文件，填入你的通义千问 API Key：

```env
OPENAI_API_KEY="sk-你的通义千问API-Key"
OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

> **注意**：你需要前往 [阿里云百炼平台](https://bailian.console.aliyun.com/) 申请 API Key。

## 🚀 使用方法

### 启动应用

在终端运行以下命令启动 Web 服务：

```bash
python app.py
```

启动成功后，终端会显示本地访问地址（通常是 `http://127.0.0.1:7860`），点击链接即可在浏览器中打开。

### 操作步骤

1.  **上传文档**：在左侧侧边栏点击“上传 PDF 文档”，选择本地的 PDF 文件。
2.  **加载知识库**：点击“加载文档”按钮，系统将自动读取文档、构建向量索引（状态栏会显示“成功加载”）。
3.  **开始提问**：在右侧聊天框输入问题，系统将基于文档内容进行流式回答，并在下方显示参考来源。

## 🧠 核心逻辑说明

### 1. 向量化
由于通义千问 Embedding 接口的特殊性，本项目自定义了 `QwenEmbeddings` 类，直接调用 DashScope SDK，解决了 LangChain 默认封装的兼容性问题。

```python
class QwenEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # ... 实现分批调用 DashScope API 的逻辑
```

### 2. 检索与生成
为了实现流式输出和引用归因，项目没有直接使用封装好的 `RetrievalQA` 链，而是手动拆解了逻辑：
1.  利用 `retriever.get_relevant_documents` 检索 Top-K 文档片段。
2.  构建 Prompt，强制 LLM 仅基于上下文回答。
3.  使用 `llm.stream` 实现流式输出。
4.  在回答结束后，拼接检索到的原文片段作为引用。

## 📝 待办事项

- [ ] 支持更多文档格式
- [ ] 增加向量库持久化存储 (本地保存/加载)
- [ ] 支持多轮对话记忆
- [ ] 增加对话历史导出功能

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
```

---

### 2. 新建文件：`requirements.txt`
(复制下面的所有内容)

```text
gradio>=4.0.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20
faiss-cpu>=1.7.4
pypdf>=3.17.0
dashscope>=1.14.0
python-dotenv>=1.0.0
```