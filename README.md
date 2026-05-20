# 📄 PDF Query App

A conversational AI app that lets you upload any PDF and ask questions about its content — powered by **LangChain**, **Groq (LLaMA 3.3 70B)**, **FAISS**, and **Streamlit**.

---

## ✨ Features

- 📁 Upload any PDF file and have it indexed instantly
- 💬 Ask natural language questions about the document
- 🧠 Maintains conversation history for multi-turn Q&A
- ⚡ Fast vector search via FAISS with MMR retrieval
- 🤗 Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- 🦙 LLM responses via Groq's LLaMA 3.3 70B model

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| PDF Loader | LangChain `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Memory | `ConversationBufferMemory` |
| QA Chain | `ConversationalRetrievalChain` |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pdf-query-app.git
cd pdf-query-app
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`.

---

## 🧭 Usage

1. Upload a PDF using the file uploader
2. Wait for the document to be indexed (one-time per upload)
3. Type your question in the input box and click **Submit**
4. The app remembers the conversation — ask follow-up questions naturally
5. Click **Clear Chat** to reset the conversation

---

## 📁 Project Structure

```
pdf-query-app/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed to Git)
├── .gitignore
└── README.md
```

---

## 🔒 .gitignore

Make sure your `.env` file is never committed. Add this to your `.gitignore`:

```
.env
__pycache__/
*.pyc
venv/
.streamlit/
```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/) for the RAG framework
- [Groq](https://groq.com/) for ultra-fast LLM inference
- [Hugging Face](https://huggingface.co/) for the embedding model
- [Streamlit](https://streamlit.io/) for the UI framework
- [FAISS](https://github.com/facebookresearch/faiss) by Meta for vector search
