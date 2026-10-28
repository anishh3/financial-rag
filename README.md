# Financial RAG

Financial RAG is a Streamlit question-answering app for exploring Infosys
financial reports. Ask a question in natural language and the app searches
report excerpts, sends the relevant evidence to a language model, and returns
an answer with document and page references.

> **This project does not train or fine-tune a language model.** It uses
> pretrained models for semantic search, reranking, and answer generation.
> The reports below are the retrieval corpus (the source material the app
> searches), not model-training data.

## What it does

- Accepts financial-report questions in a Streamlit chat interface.
- Retrieves evidence using both keyword search and semantic vector search.
- Combines the rankings and reranks the most relevant excerpts.
- Grounds generated answers in retrieved report evidence and displays sources.
- Keeps recent chat history for follow-up questions.
- Includes basic prompt guardrails and rejects unsupported source citations.

## Screenshot

![Financial Reports chat interface](./financial-retrieval/images/financial-reports-chat.png)

## Report corpus

The local development corpus contains four Infosys Limited financial-report
PDFs for FY2026:

| Report | Period covered |
|---|---|
| Q1 FY2026 | Quarter ended June 30, 2025 |
| Q2 FY2026 | Quarter ended September 30, 2025 |
| Q3 FY2026 | Quarter ended December 31, 2025 |
| Additional information 2025-26 | Includes FY2026 full-year information, including March 31, 2026 |

The reports include financial statements and other company-reported financial
and operating information. The raw PDFs, extracted records, text chunks, and
local ChromaDB index are excluded from this GitHub repository; they are not
uploaded when you clone it. Obtain the source reports from Infosys's official
investor-relations publications, and prepare the local retrieval artifacts
before starting the app.

## How retrieval and answering work

1. Financial-report content is represented as structured records and split
   into searchable chunks with source metadata such as document, page, period,
   metric, and unit.
2. **BM25** (`rank-bm25`) finds keyword matches. **Sentence Transformers**
   embeds the query and chunks with `all-MiniLM-L6-v2`; **ChromaDB** performs
   semantic nearest-neighbor search over those vectors.
3. Reciprocal Rank Fusion combines the keyword and semantic rankings. A
   pretrained `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks candidates.
4. The top evidence excerpts are sent to the configured Ollama model,
   `qwen2.5:1.5b` by default. The answer prompt asks the model to use the
   supplied evidence, avoid guessing, and cite the supplied source labels.

The generation model is not adapted to these reports. Retrieval provides
report-specific context at question time.

## Technologies

| Technology | Role |
|---|---|
| Python | Application and retrieval pipeline |
| Streamlit | Chat user interface |
| Ollama | Local language-model serving and answer generation |
| Qwen 2.5 1.5B | Default answer-generation model |
| `sentence-transformers` / `all-MiniLM-L6-v2` | Text embeddings for semantic retrieval |
| ChromaDB | Persistent vector store |
| `rank-bm25` | Keyword retrieval |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Candidate reranking |

## Run locally

### 1. Install Python dependencies

From the repository root:

```powershell
python -m pip install -r .\financial-retrieval\requirements.txt
```

### 2. Install and start Ollama

Install Ollama for your operating system, then pull the default model:

```powershell
ollama pull qwen2.5:1.5b
```

Keep the Ollama service running while you use the app. By default, the app
connects to `http://localhost:11434`.

### 3. Prepare report data and retrieval artifacts

The repo does not include the PDFs or generated index, so a fresh clone cannot
retrieve answers until those local assets are prepared. The expected local
directories are:

```text
financial-retrieval/
  data/raw/                 # source PDFs
  artifacts/
    normalized/             # structured records
    chunks/                 # retrieval chunks
    vector_store/chroma/    # persistent vector index
```

The normalization script accepts **DoclingDocument JSON**, not PDFs. Convert
each source PDF to that JSON format with your preferred PDF-to-Docling
workflow, then normalize each JSON file. For example, from the repository
root:

```powershell
python .\financial-retrieval\src\normalize.py `
  .\path\to\q1-2026.json `
  --output-dir .\financial-retrieval\artifacts\normalized\q1-2026
```

Repeat for the other reports, then create chunks and build the vector index:

```powershell
python .\financial-retrieval\src\chunk_records.py
python .\financial-retrieval\src\build_vector_index.py
```

The normalization, chunking, and indexing scripts are in
`financial-retrieval/src/`. Generated files are local and ignored by Git.

### 4. Launch the chat app

```powershell
python -m streamlit run .\financial-retrieval\src\chat_app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your
browser.

## Command-line question answering

The same retrieval and Ollama answer flow can be used without the UI:

```powershell
python .\financial-retrieval\src\answer_question.py "What was Infosys revenue in Q1 FY2026?"
```

Options include `--top-k` to change the number of evidence chunks, `--model`
to select another locally installed Ollama model, `--ollama-url` to select the
Ollama server address, and `--no-rerank` to skip cross-encoder reranking.

## Limitations

- Answers are limited by the reports and artifacts available to the retrieval
  pipeline. Missing or ambiguous evidence can lead to incomplete answers.
- This is a report-search and summarization tool, not investment advice or a
  substitute for the original filings.
- Local Ollama is the default. If you configure a remote model service, review
  its privacy and data-handling policies before sending report excerpts.
