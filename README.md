# Local PDF Study Companion

A fully local AI study assistant that reads your PDF files and lets you generate study guides, quizzes, and ask questions without sending anything to the cloud.

Built with **LangChain**, **Ollama**, and **ChromaDB**.

---

## Features

- **Summarize Everything:** Generates a structured study guide from all your PDFs and exports it as a PDF file
- **Quiz Mode:** Creates 10 multiple-choice questions with a full answer key, exported as a PDF
- **Chat Mode:** Ask any question about your documents in a back-and-forth loop
- **English / Greek language support:** choose your language at startup; all answers and exported PDFs follow your choice
- **Persistent vector store:** PDFs are processed and embedded once; every run after that loads the database instantly

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/local-pdf-study.git
cd local-pdf-study
```

**2. Install Python dependencies**

```bash
pip install langchain langchain-community langchain-ollama langchain-chroma chromadb pypdf reportlab
```

**3. Pull the required Ollama models**

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

> `llama3` is the chat model. `nomic-embed-text` is used to embed your documents into the vector store.

---

## Usage

**1. Add your PDFs**

Create a folder called `my_exam_files` in the project directory and place your PDF files inside it.

```
local-pdf-study/
├── rag.py
└── my_exam_files/
    ├── lecture1.pdf
    ├── lecture2.pdf
    └── ...
```

**2. Start Ollama** (if it isn't already running)

```bash
ollama serve
```

**3. Run the script**

```bash
python rag.py
```

On first run, the script will read and embed all your PDFs into a local database saved at `./study_db`. This takes a few minutes depending on how many files you have. Every run after that skips this step and loads instantly.

---

## Menu

```
┌─────────────────────────────────────────┐
│               MAIN MENU                 │
├─────────────────────────────────────────┤
│  [1]  Summarize Everything              │
│  [2]  Quiz Mode (10 MCQs)               │
│  [3]  Ask a Question                    │
│  [0]  Exit                              │
└─────────────────────────────────────────┘
```

Generated PDFs are saved to the `./study_guides/` folder with a timestamp in the filename so nothing gets overwritten.

---

## Language Support

At startup you will be asked to choose between English and Greek. The choice applies to all three features — answers from the LLM and the PDF headers will both follow the selected language.

For Greek PDFs to render correctly, a Unicode-capable font must be available on your system. The script checks automatically. To install one on Linux:

```bash
sudo apt install fonts-dejavu
```

On macOS or Windows, Arial is typically already present and will be detected automatically.

---

## Project Structure

```
local-pdf-study/
├── rag.py              # main script
├── my_exam_files/      # your input PDFs (created by you)
├── study_db/           # auto-generated ChromaDB vector store
└── study_guides/       # auto-generated PDF outputs
```

> `study_db/` and `study_guides/` are created automatically on first run. You can add them to `.gitignore`.

---

## Configuration

All settings are at the top of `rag.py` and can be changed without touching anything else:

| Variable | Default | Description |
|---|---|---|
| `PDF_FOLDER` | `./my_exam_files` | Folder containing input PDFs |
| `DB_FOLDER` | `./study_db` | Where the vector database is stored |
| `OUTPUT_FOLDER` | `./study_guides` | Where generated PDFs are saved |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `LLM_MODEL` | `llama3` | Ollama model used for answers |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for embeddings |
| `TOP_K` | `6` | Number of chunks retrieved per query |

To rebuild the database (e.g. after adding new PDFs), simply delete the `study_db/` folder and re-run.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection Refused` | Run `ollama serve` in a separate terminal |
| `Model Not Found` | Run `ollama pull llama3` and `ollama pull nomic-embed-text` |
| `No PDFs Found` | Make sure your files are inside `my_exam_files/` |
| Greek text shows boxes in PDF | Install DejaVu fonts (`sudo apt install fonts-dejavu`) |
| Slow on first run | Normal — embedding takes time once, then it's instant |

---

## License

MIT License. Feel free to use, modify, and share.