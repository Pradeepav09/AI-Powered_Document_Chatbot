# 🧠 DocuMind – AI-Powered Document Chatbot

DocuMind is an AI-powered document chatbot that allows users to upload PDF, TXT, and Markdown files, ask questions, and receive context-aware answers based on the uploaded documents.

Built using **Python, Streamlit, Google Gemini API, ChromaDB, and Sentence Transformers**, the application implements Retrieval-Augmented Generation (RAG) to retrieve relevant document information and generate accurate responses with source references.

---

## ✨ Features

- 📄 **Multi-Format Document Upload** – Upload PDF, TXT, and Markdown files.
- 🤖 **AI-Powered Chat** – Ask questions and receive responses using Google Gemini.
- 🔍 **Semantic Search** – Retrieve relevant information using vector embeddings.
- 🧠 **RAG Architecture** – Generate answers based on retrieved document context.
- 📚 **Source References** – Display document names and page numbers for retrieved information.
- 🔎 **Automatic OCR** – Extract text from scanned PDF documents using Tesseract OCR.
- 💬 **Chat History** – Save and retrieve previous conversations.
- 🗑️ **Chat Management** – Create new conversations and delete existing ones.
- 💡 **Suggested Questions** – Automatically generate questions based on uploaded documents.
- 🎨 **Dark & Light Themes** – Switch between different interface themes.
- 📥 **Export Conversations** – Download chat conversations in Markdown format.
- ⚡ **Persistent Vector Storage** – Store document embeddings using ChromaDB.
- 🔐 **API Key Management** – Accept Gemini API keys through the application interface.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| AI Model | Google Gemini |
| AI Integration | Google GenAI SDK |
| RAG Architecture | Retrieval-Augmented Generation |
| Vector Database | ChromaDB |
| Embedding Model | Sentence Transformers |
| Embedding Model Used | all-MiniLM-L6-v2 |
| Database | SQLite |
| PDF Processing | PyPDF |
| OCR Engine | Tesseract OCR |
| PDF Conversion | Poppler |
| Python Libraries | PyTorch, pdf2image, pytesseract |

---

## 🏗️ System Architecture

DocuMind follows a Retrieval-Augmented Generation (RAG) architecture.

1. **Document Upload:** Users upload PDF, TXT, or Markdown documents.
2. **Text Extraction:** Extract text from documents using PyPDF or OCR.
3. **Text Chunking:** Divide extracted content into smaller chunks.
4. **Embedding Generation:** Convert text chunks into vector embeddings using Sentence Transformers.
5. **Vector Storage:** Store embeddings and document metadata in ChromaDB.
6. **Semantic Retrieval:** Retrieve relevant document chunks based on the user's question.
7. **AI Response Generation:** Pass retrieved context to Google Gemini.
8. **Source Attribution:** Display relevant document names and page references.

---

## 📂 Project Structure

```
DocuMind/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── .chroma_db/          # Generated automatically
│
└── chat_history.db      # Generated automatically
```

**Note:** `.chroma_db/` and `chat_history.db` are generated when the application runs. They do not need to be uploaded to GitHub.

---

# 🚀 Installation & Setup

Follow the instructions below to configure and run DocuMind on your system.

## 1. Prerequisites

Install the following software before setting up the project:

- Python 3.10 or higher
- pip
- Git
- Google Gemini API Key
- Tesseract OCR
- Poppler

Check your Python installation:

```
python --version
```

Check pip:

```
pip --version
```

---

## 2. Clone the Repository

```
git clone https://github.com/your-username/DocuMind.git
```

Navigate into the project directory:

```
cd DocuMind
```

---

## 3. Install Poppler

Poppler is used by `pdf2image` to convert PDF pages into images for OCR processing.

### Windows Installation

**Step 1:** Visit the Poppler Windows releases page:

https://github.com/oschwartz10612/poppler-windows/releases/latest

**Step 2:** Download the latest available ZIP release.

**Step 3:** Extract the downloaded ZIP file.

**Step 4:** Move the extracted Poppler folder to a convenient location.

For example:

```
C:\poppler\
```

**Step 5:** Locate the `bin` directory inside the extracted `Library` folder.

Your Poppler executable directory should look like:

```
C:\poppler\Library\bin
```

**Step 6:** Add the following directory to your Windows PATH environment variable:

```
C:\poppler\Library\bin
```

**Step 7:** Restart your terminal and verify the installation:

```
pdftoppm -v
```

If the version information appears, Poppler is installed successfully.

### Debian / Linux Installation

Install Poppler using APT:

```
sudo apt update
sudo apt install poppler-utils
```

Verify the installation:

```
pdftoppm -v
```

---

## 4. Install Tesseract OCR

Tesseract OCR is used to extract text from scanned PDF pages and image-based documents.

### Windows Installation

**Step 1:** Visit the UB Mannheim Tesseract OCR download page:

https://github.com/UB-Mannheim/tesseract/wiki

**Step 2:** Download the latest Windows installer.

**Step 3:** Run the installer and complete the installation.

**Step 4:** The default installation directory is generally:

```
C:\Program Files\Tesseract-OCR
```

**Step 5:** Add the following directory to your Windows PATH:

```
C:\Program Files\Tesseract-OCR
```

**Step 6:** Restart your terminal.

Verify the installation:

```
tesseract --version
```

### Debian / Linux Installation

Install Tesseract OCR:

```
sudo apt update
sudo apt install tesseract-ocr
```

Install English language data:

```
sudo apt install tesseract-ocr-eng
```

Verify the installation:

```
tesseract --version
```

Check the installed language:

```
tesseract --list-langs
```

---

## 5. Configure OCR Paths in app.py

**Important:** The current application contains hardcoded Windows paths for Tesseract and Poppler. Update these paths according to your operating system before running the application.

Locate the `extract_pdf_pages()` function in `app.py`.

### For Windows

Update the paths as follows:

```
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

poppler_bin_path = r"C:\poppler\Library\bin"
```

Make sure these paths match your actual installation directories.

### For Debian / Linux

Use:

```
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

poppler_bin_path = None
```

On Linux, Poppler utilities installed through APT are normally available through the system PATH.

---

## 6. Create a Python Virtual Environment

Creating a virtual environment helps isolate project dependencies.

### Windows

Create the environment:

```
python -m venv venv
```

Activate it:

```
venv\Scripts\activate
```

### Debian / Linux

Install the virtual environment package if required:

```
sudo apt install python3-venv
```

Create the environment:

```
python3 -m venv venv
```

Activate it:

```
source venv/bin/activate
```

---

## 7. Install Python Dependencies

Upgrade pip:

```
python -m pip install --upgrade pip
```

Install the required packages:

```
pip install -r requirements.txt
```

### Dependencies Included

- Streamlit
- ChromaDB
- Google GenAI
- PyPDF
- Sentence Transformers
- PyTorch
- Pytesseract
- pdf2image

**Note:** The first installation may take some time because PyTorch and Sentence Transformers have additional dependencies. The embedding model `all-MiniLM-L6-v2` is also downloaded automatically the first time the application loads it.

---

## 8. Configure Google Gemini API Key

DocuMind uses the Google Gemini API to generate AI responses.

**Step 1:** Visit Google AI Studio:

https://aistudio.google.com/apikey

**Step 2:** Sign in with your Google account.

**Step 3:** Generate an API key.

**Step 4:** Copy your API key.

**Step 5:** Start the application and enter the API key in the sidebar under **Gemini API Key**.

**Step 6:** Select an available Gemini model from the model dropdown.

Keep your API key private. Never upload API keys or credentials to a public GitHub repository.

---

# ▶️ Running the Application

After completing all installation and configuration steps, execute:

```
streamlit run app.py
```

The application will start, and Streamlit will display a local URL.

Open the following address in your browser:

```
http://localhost:8501
```

---

## 📖 How to Use DocuMind

1. Open the application in your browser.
2. Enter your Gemini API key in the sidebar.
3. Select an available Gemini model.
4. Upload your PDF, TXT, or Markdown documents.
5. Enable Auto-OCR if you want to process scanned PDF pages.
6. Click **Process Documents**.
7. Wait for document processing and embedding generation.
8. Enter your questions in the chat input.
9. View AI-generated answers based on the uploaded documents.
10. Expand **Show sources** to inspect document references.
11. Export your conversations using the Download Chat option.

---

## 🗄️ Database Management

DocuMind uses two storage systems:

### SQLite

SQLite stores:

- Chat sessions
- Conversation titles
- User messages
- AI responses
- Source references
- Conversation timestamps

Database file:

```
chat_history.db
```

The database and required tables are created automatically when the application starts.

### ChromaDB

ChromaDB stores:

- Document text chunks
- Vector embeddings
- Document names
- Page numbers
- Session metadata

Storage directory:

```
.chroma_db/
```

The vector database is initialized automatically by the application.

---

## 🔒 Security & Data Management

- Gemini API keys are entered through the application interface.
- Password-free API key handling through Streamlit session state.
- Chat records are stored locally using SQLite.
- Document embeddings are stored locally using ChromaDB.
- Document retrieval is scoped using the API key hash.
- Uploaded documents are processed for contextual information.

**Important:** This project is intended for educational and development purposes. Review API key handling, local database permissions, and deployment security before using it with sensitive documents or exposing it publicly.

---

## ⚠️ Troubleshooting

| Error | Possible Solution |
|---|---|
| `TesseractNotFoundError` | Verify the Tesseract installation and executable path. |
| `PDFInfoNotInstalledError` | Install Poppler and configure its executable path. |
| `pdftoppm: command not found` | Add Poppler's `bin` directory to PATH or install `poppler-utils` on Linux. |
| `ModuleNotFoundError` | Activate the virtual environment and install requirements. |
| Gemini API Error | Verify the API key, model availability, and API quota. |
| ChromaDB Error | Check the local database directory and write permissions. |
| Slow First Startup | Allow time for the embedding model to download and initialize. |
| OCR Not Extracting Text | Verify Tesseract language data and test with a readable scanned PDF. |

---

## 🔮 Future Enhancements

- Support for additional document formats.
- Multiple embedding model options.
- Advanced document filtering.
- Improved document preview functionality.
- User authentication and authorization.
- Cloud-based document storage.
- Conversation export in PDF format.
- Improved OCR language support.
- Docker-based deployment.
