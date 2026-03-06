# RAG Study System
# Uses LangChain + Ollama + ChromaDB to answer questions from your PDF files.
#
# Install dependencies:
#   pip install langchain langchain-community langchain-ollama langchain-chroma chromadb pypdf reportlab
#
# Pull Ollama models (once):
#   ollama pull llama3
#   ollama pull nomic-embed-text

import os
import sys
from datetime import datetime

# Document loading and text splitting
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ollama LLM and embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ChromaDB vector store
from langchain_chroma import Chroma

# LangChain chain and prompt
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ---------------------------------------------------------------------------
# CONFIGURATION — edit these paths and settings to match your setup
# ---------------------------------------------------------------------------

PDF_FOLDER    = "./my_exam_files"  # folder containing your PDF files
DB_FOLDER     = "./study_db"       # where ChromaDB saves the vector store
OUTPUT_FOLDER = "./study_guides"   # where generated PDFs are saved
CHUNK_SIZE    = 1000               # characters per text chunk
CHUNK_OVERLAP = 100                # overlap between chunks
LLM_MODEL     = "llama3"           # Ollama chat model
EMBED_MODEL   = "nomic-embed-text" # Ollama embedding model
TOP_K         = 6                  # number of chunks retrieved per query


# ---------------------------------------------------------------------------
# UNICODE FONT SETUP
# Needed so Greek (and other non-Latin) characters render correctly in PDFs.
# The script checks common system font paths and registers the first one found.
# Falls back to Helvetica if none are available (Greek may not display).
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]

UNICODE_FONT = "Helvetica"  # default fallback

for _candidate in _FONT_CANDIDATES:
    if os.path.isfile(_candidate):
        try:
            pdfmetrics.registerFont(TTFont("UnicodeFont", _candidate))
            UNICODE_FONT = "UnicodeFont"
        except Exception:
            pass
        break


# ---------------------------------------------------------------------------
# LANGUAGE SELECTION
# Shown once at startup. Returns 'en' or 'el'.
# ---------------------------------------------------------------------------

def choose_language() -> str:
    print("\n┌─────────────────────────────────┐")
    print("│       SELECT LANGUAGE           │")
    print("├─────────────────────────────────┤")
    print("│  [1]  English                   │")
    print("│  [2]  Greek                     │")
    print("└─────────────────────────────────┘")

    while True:
        try:
            choice = input("\nChoose language: ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "1"

        if choice == "1":
            print("Language: English.")
            return "en"
        elif choice == "2":
            print("Γλώσσα: Ελληνικά.")
            return "el"
        else:
            print("Please enter 1 or 2.")


# ---------------------------------------------------------------------------
# PDF HELPERS
# Both functions use the same ReportLab styling logic.
# Lines starting with '#' become headings, '##' sub-headings,
# '-' or '*' become bullet points, everything else is body text.
# ---------------------------------------------------------------------------

def _build_pdf_styles(base):
    # Returns a dict of named ParagraphStyle objects using UNICODE_FONT
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"],
            fontSize=22, fontName=UNICODE_FONT,
            textColor=colors.HexColor("#1a3a5c"), spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle", parent=base["Normal"],
            fontSize=11, fontName=UNICODE_FONT,
            textColor=colors.HexColor("#666666"), spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "DocH1", parent=base["Heading1"],
            fontSize=15, fontName=UNICODE_FONT,
            textColor=colors.HexColor("#1a3a5c"),
            spaceBefore=20, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "DocH2", parent=base["Heading2"],
            fontSize=12, fontName=UNICODE_FONT,
            textColor=colors.HexColor("#2e6da4"),
            spaceBefore=12, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "DocBody", parent=base["Normal"],
            fontSize=10, fontName=UNICODE_FONT,
            leading=16, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "DocBullet", parent=base["Normal"],
            fontSize=10, fontName=UNICODE_FONT,
            leading=15, leftIndent=20, bulletIndent=8, spaceAfter=3,
        ),
    }


def _text_to_story(text: str, styles: dict) -> list:
    # Converts plain LLM output into a list of ReportLab flowables
    story = []
    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        # Escape XML characters so ReportLab doesn't crash
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if safe.startswith("## "):
            story.append(Paragraph(safe[3:], styles["h2"]))
        elif safe.startswith("# "):
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#cccccc"),
                spaceBefore=10, spaceAfter=4,
            ))
            story.append(Paragraph(safe[2:], styles["h1"]))
        elif safe.startswith(("- ", "* ")):
            story.append(Paragraph(f"&#8226; {safe[2:]}", styles["bullet"]))
        else:
            story.append(Paragraph(safe, styles["body"]))

    return story


def save_study_guide_as_pdf(text: str, lang: str = "en") -> str:
    # Saves the study guide text as a formatted PDF and returns the file path
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_FOLDER, f"study_guide_{timestamp}.pdf")

    # Localised header labels
    title_text    = "Οδηγός Μελέτης" if lang == "el" else "Study Guide"
    generated_lbl = "Δημιουργήθηκε"  if lang == "el" else "Generated on"
    system_lbl    = "Σύστημα RAG"    if lang == "el" else "RAG Study System"

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title=title_text, author="RAG Study System",
    )

    styles = _build_pdf_styles(getSampleStyleSheet())
    story  = []

    # Header block
    story.append(Paragraph(title_text, styles["title"]))
    story.append(Paragraph(
        f"{generated_lbl} {datetime.now().strftime('%d %B %Y, %H:%M')} &#8212; {system_lbl}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#1a3a5c"), spaceAfter=14))

    story += _text_to_story(text, styles)
    doc.build(story)
    return output_path


def save_quiz_as_pdf(text: str, lang: str = "en") -> str:
    # Saves the quiz text as a formatted PDF and returns the file path
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_FOLDER, f"quiz_{timestamp}.pdf")

    # Localised header labels
    title_text    = "Κουίζ Εξέτασης" if lang == "el" else "Quiz"
    generated_lbl = "Δημιουργήθηκε"  if lang == "el" else "Generated on"
    system_lbl    = "Σύστημα RAG"    if lang == "el" else "RAG Study System"

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title=title_text, author="RAG Study System",
    )

    styles = _build_pdf_styles(getSampleStyleSheet())
    story  = []

    # Header block
    story.append(Paragraph(title_text, styles["title"]))
    story.append(Paragraph(
        f"{generated_lbl} {datetime.now().strftime('%d %B %Y, %H:%M')} &#8212; {system_lbl}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a5c"), spaceAfter=14))

    story += _text_to_story(text, styles)
    doc.build(story)
    return output_path


# ---------------------------------------------------------------------------
# STEP 1 — LOAD AND SPLIT PDFs
# Scans PDF_FOLDER, loads every PDF, and splits pages into chunks.
# Exits early with a clear error message if the folder is missing or empty.
# ---------------------------------------------------------------------------

def load_and_split_pdfs(folder: str) -> list:
    if not os.path.isdir(folder):
        print(f"\n[ERROR] Folder '{folder}' does not exist.")
        print("Create it and place your PDFs inside, then re-run.\n")
        sys.exit(1)

    print(f"\nScanning '{folder}' for PDFs ...")
    loader = DirectoryLoader(
        folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,  # speeds up loading many files
    )
    documents = loader.load()

    if not documents:
        print(f"\n[ERROR] No PDFs found in '{folder}'. Add your files and re-run.\n")
        sys.exit(1)

    print(f"Loaded {len(documents)} page(s).")

    # Split pages into smaller overlapping chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# ---------------------------------------------------------------------------
# STEP 2 — BUILD OR LOAD VECTOR STORE
# On first run: embeds all chunks and saves to DB_FOLDER (takes a few minutes).
# On later runs: loads the saved database instantly, skipping re-embedding.
# ---------------------------------------------------------------------------

def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.isdir(DB_FOLDER) and os.listdir(DB_FOLDER):
        # Database already exists — load it from disk
        print(f"\nLoading existing database from '{DB_FOLDER}' ...")
        vectorstore = Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=embeddings,
        )
        print("Vector store loaded.")
    else:
        # First run — process PDFs and build the database
        print("\nNo database found. Building from PDFs (one-time setup) ...")
        chunks = load_and_split_pdfs(PDF_FOLDER)

        print(f"\nEmbedding {len(chunks)} chunks with '{EMBED_MODEL}' ...")
        print("This may take a few minutes. Grab a coffee.\n")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_FOLDER,
        )
        print(f"\nDatabase saved to '{DB_FOLDER}'. Future runs will skip this step.")

    return vectorstore


# ---------------------------------------------------------------------------
# STEP 3 — BUILD THE RETRIEVAL CHAIN
# Connects the Ollama LLM to the ChromaDB retriever via a RetrievalQA chain.
# ---------------------------------------------------------------------------

def build_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

    # Retrieve the TOP_K most relevant chunks for each query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # System prompt — instructs the LLM to only use the retrieved context
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a knowledgeable study assistant. "
            "Use ONLY the context below to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )
    return chain


# ---------------------------------------------------------------------------
# FEATURE 1 — SUMMARIZE EVERYTHING
# Generates a full study guide and saves it as a PDF.
# ---------------------------------------------------------------------------

def summarize_everything(chain: RetrievalQA, lang: str = "en") -> None:
    print("\n  Generating study guide ...")

    # Appended to every query when Greek is selected
    lang_instruction = "Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ στα Ελληνικά." if lang == "el" else ""

    query = (
        "Give me a comprehensive, high-level study guide that covers "
        "all the major topics, key concepts, definitions, and themes "
        "found in the documents. Organise it by subject area using "
        "markdown-style headings (# for main topics, ## for sub-topics) "
        "and bullet points starting with '- ' for key facts."
        + lang_instruction
    )

    response = chain.invoke({"query": query})
    study_guide_text = response["result"]

    print("\n" + study_guide_text)

    # Save the result to a timestamped PDF
    print("\nSaving to PDF ...")
    try:
        pdf_path = save_study_guide_as_pdf(study_guide_text, lang)
        print(f"Saved to {pdf_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save PDF: {exc}")
        print("Make sure reportlab is installed: pip install reportlab")


# ---------------------------------------------------------------------------
# FEATURE 2 — QUIZ MODE
# Generates 10 multiple-choice questions with an answer key, saves as PDF.
# ---------------------------------------------------------------------------

def quiz_mode(chain: RetrievalQA, lang: str = "en") -> None:
    print("\nGenerating quiz")

    lang_instruction = "Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ στα Ελληνικά." if lang == "el" else ""

    query = (
        "Create exactly 10 challenging multiple-choice exam questions "
        "based on the documents. Each question must have 4 options "
        "(A, B, C, D). Number them 1-10. After all 10 questions, "
        "print a clearly separated ANSWER KEY section that lists "
        "the correct answer and a one-sentence explanation for each."
        + lang_instruction
    )

    response = chain.invoke({"query": query})
    quiz_text = response["result"]

    print("\n" + quiz_text)

    # Save the quiz and answer key to a timestamped PDF
    print("\nSaving to PDF...")
    try:
        pdf_path = save_quiz_as_pdf(quiz_text, lang)
        print(f"Saved -> {pdf_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save PDF: {exc}")
        print("Make sure reportlab is installed: pip install reportlab")


# ---------------------------------------------------------------------------
# FEATURE 3 — CHAT LOOP
# Interactive Q&A. Type 'exit' or 'quit' to return to the main menu.
# ---------------------------------------------------------------------------

def chat_loop(chain: RetrievalQA, lang: str = "en") -> None:
    print("\nChat mode — type 'exit' to return to the menu.\n")

    lang_instruction = "Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ στα Ελληνικά." if lang == "el" else ""

    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Returning to menu")
            break

        print("\nThinking...\n")
        try:
            response = chain.invoke({"query": question + lang_instruction})
            print("Answer:\n")
            print(response["result"])
            print()
        except Exception as exc:
            print(f"[ERROR] {exc}")


# ---------------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------------

BANNER = r"""
 ____      _    ____   ____  _               _
|  _ \    / \  / ___| / ___|| |_ _   _  __| |_   _
| |_) |  / _ \| |  _  \___ \| __| | | |/ _` | | | |
|  _ <  / ___ \ |_| |  ___) | |_| |_| | (_| | |_| |
|_| \_\/_/   \_\____| |____/ \__|\__,_|\__,_|\__, |
                                              |___/
         Local RAG -- LangChain + Ollama + ChromaDB
"""


def main_menu(chain: RetrievalQA, lang: str = "en") -> None:
    print(BANNER)
    while True:
        print("\n┌─────────────────────────────────────────┐")
        print("│               MAIN MENU                 │")
        print("├─────────────────────────────────────────┤")
        print("│  [1]  Summarize Everything              │")
        print("│  [2]  Quiz Mode (10 MCQs)               │")
        print("│  [3]  Ask a Question                    │")
        print("│  [0]  Exit                              │")
        print("└─────────────────────────────────────────┘")

        try:
            choice = input("\nChoose an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "1":
            summarize_everything(chain, lang)
        elif choice == "2":
            quiz_mode(chain, lang)
        elif choice == "3":
            chat_loop(chain, lang)
        elif choice == "0":
            print("\nGood luck on your exams!\n")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 0, 1, 2, or 3.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Starting RAG Study System ...")

    # Load or build the vector store
    try:
        vectorstore = get_vectorstore()
    except Exception as exc:
        print(f"\n[FATAL] Could not initialise vector store: {exc}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    # Wire up the LLM retrieval chain
    try:
        chain = build_chain(vectorstore)
        print("  LLM chain ready.")
    except Exception as exc:
        print(f"\n[FATAL] Could not build chain: {exc}")
        print("Make sure llama3 and nomic-embed-text are pulled in Ollama.")
        sys.exit(1)

    # Ask for language preference, then open the menu
    lang = choose_language()
    main_menu(chain, lang)