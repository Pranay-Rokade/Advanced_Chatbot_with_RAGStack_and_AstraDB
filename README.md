# 🧠 Advanced Chatbot with RAGStack and Astra DB Vector Store

## ❓ What is RAGStack?

**RAGStack** is a **modular open-source Retrieval-Augmented Generation (RAG) architecture** built by **DataStax** that integrates:

- ⚙️ **LangChain** – for LLM orchestration and chaining
- ☁️ **Astra DB** – serverless vector database built on Apache Cassandra®
- 🧠 **LLMs** – such as OpenAI, Ollama, or Groq for generation
- 📦 **Embedding models** – to convert text into semantic vectors

RAGStack enables developers to build scalable, production-grade **RAG pipelines** that combine LLMs with private data for more accurate, grounded, and reliable responses.

---

## 🚀 Key Features

- 🔗 Integrates **LangChain** with **Astra DB Vector Store**
- 📦 Loads and embeds **philosopher quotes dataset** from Hugging Face
- 🧬 Uses **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`)
- 🤖 Answers questions based strictly on stored context
- ⚡ Uses **Groq** LLaMA3 inference engine via OpenAI-compatible API
- 📑 Supports semantic retrieval and metadata-based storage
- ☁️ Serverless, scalable, production-ready architecture

---

## 🛠️ Stack Overview

| Component        | Role                                     |
|------------------|------------------------------------------|
| **Astra DB**      | Serverless, cloud-native vector store    |
| **LangChain**     | RAG pipeline orchestration               |
| **Groq + LLaMA3** | Fast and efficient LLM for answering     |
| **HuggingFace**   | Embedding generation                     |
| **Philosopher Quotes Dataset** | Document source for semantic search |

---

## 📦 Setup Instructions

### 1. Install Dependencies

```bash
pip install langchain langchain-community langchain-groq langchain-astra cassio datasets python-dotenv
````

### 2. Add Environment Variables

Create a `.env` file:

```env
ASTRA_DB_API_ENDPOINT=https://your-astra-endpoint.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:your_token
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the Script

```bash
python app.py
```

---

## 📖 How It Works

### 📚 Step 1: Load Dataset

```python
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
```

### 🧱 Step 2: Convert to LangChain Documents

Each quote is wrapped in a `Document` object with metadata (author + tags).

### 🧬 Step 3: Embedding & Storage

```python
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vstore = AstraDBVectorStore(...)
vstore.add_documents(docs)
```

### 🔍 Step 4: Retrieval

```python
retriever = vstore.as_retriever(search_kwargs={"k": 3})
```

### 🧠 Step 5: RAG Pipeline

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

---

## 💬 Example Queries

```python
chain.invoke("What is the importance of controlling emotions?")
```

---

## ✅ What You Learned

* ✅ What **RAGStack** is and how it simplifies RAG development
* ✅ How to integrate **Astra DB** with LangChain
* ✅ How to process, embed, and store documents using metadata
* ✅ How to retrieve and answer context-specific questions using **Groq + LLaMA3**
* ✅ How to build a custom **retrieval pipeline**

---

## 🔮 Future Enhancements

* Add Streamlit / FastAPI frontend
* Use LangSmith for debugging and observability
* Add metadata filtering
* Save FAISS/Astra state for reuse

---

## 🙌 Credits

* [LangChain](https://www.langchain.com/)
* [RAGStack by DataStax](https://docs.datastax.com/en/astra/docs/ragstack/)
* [Groq](https://groq.com/)
* [Hugging Face Datasets](https://huggingface.co/datasets)
* [CassIO](https://github.com/CassioML/cassio)

---

## 🚀 Build Scalable, Grounded Chatbots with RAGStack!
