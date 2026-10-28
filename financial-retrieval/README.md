# Financial retrieval

## Screenshot

![Financial Reports chat interface](../images/financial-reports-chat.png)

## Ask a question with a local language model

The answer command retrieves relevant report chunks, then sends those chunks
and the question to Ollama. By default, it uses the locally installed
`qwen2.5:1.5b` model and provides the five highest-ranked chunks as evidence.
The model is instructed to answer only from that evidence and cite sources by
document and page.

From the workspace root, with Ollama running:

```powershell
.\.venv\Scripts\python.exe .\financial-retrieval\src\answer_question.py "What was Infosys revenue in Q1 FY2026?"
```

Options:

- `--top-k 3` changes how many retrieved chunks are passed to the model.
- `--model qwen2.5:1.5b` selects another model already available in Ollama.
- `--ollama-url http://localhost:11434` changes the Ollama server address.
- `--no-rerank` skips cross-encoder reranking.

The retrieval and embedding models must be available to the Python
environment. Ollama must have the selected generation model installed. If the
default model is not installed, run `ollama pull qwen2.5:1.5b`.

## Chat interface

Install the Streamlit dependency into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\financial-retrieval\requirements.txt
```

From the workspace root, start the chat UI while Ollama is running:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\financial-retrieval\src\chat_app.py
```

The bright, simplified chat remembers the current conversation in Streamlit's
session state, so follow-up questions can refer to earlier questions. The
interface offers sample questions to get started, uses the default local
Ollama model and five evidence chunks, keeps at most the recent three
exchanges for model context, limits each question to 1,000 characters, and
rejects requests to override its instructions or reveal hidden prompts. The
model is instructed to answer only from retrieved report evidence. Each answer
displays the document/page sources used; if the model cites a source label
that was not retrieved, its generated answer is withheld rather than presented
as supported.
