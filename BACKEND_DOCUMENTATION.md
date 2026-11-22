# NER Project - Backend Documentation

## 1. Overview

The backend is a small [FastAPI](https://fastapi.tiangolo.com/) service that runs the fine-tuned NER model on text sent by the frontend. It can:

- **Extract** the named entities in a text (people, organizations and locations).
- **Redact** the text, replacing each entity with its label (`[PER]`) or with block characters.

```
Frontend (HTML/JS)  --JSON-->  api/main.py (FastAPI)  -->  src/infer.py  -->  Hugging Face pipeline
                    <--JSON--                                                  (DistilBERT NER model)
```

- Text is processed on the server that runs the API. The only outside connection is the one-time model download from the Hugging Face Hub.
- File parsing for the frontend (PDF, Word, OCR) happens in the browser. Only the extracted text is sent to the API.

## 2. Backend Files

| File | Purpose |
| :--- | :--- |
| `api/main.py` | FastAPI app: endpoints, request validation, CORS |
| `api/__init__.py` | Empty file that makes `api` a package |
| `src/infer.py` | Model loading, entity detection, long-text handling, redaction, command-line tool |
| `frontend/` | Page that calls the API (optional location, see section 7) |

The API contains no model logic. It calls `predict` and `redact_text` from `src/infer.py`, so the command-line tool and the API always behave the same way.

## 3. Setup

### Requirements
- Python 3.10 or newer
- Packages: `fastapi`, `uvicorn`, `transformers`, `torch`

```bash
pip install fastapi uvicorn transformers torch
```

Make sure these are also listed in `requirements.txt`.

### The model
By default the backend loads the model from the Hugging Face Hub:

```
VeeJack/malaysian-ner-distilbert
```

- The first start downloads the model (several hundred MB), so it needs internet access and takes longer. Later starts use the local cache (`~/.cache/huggingface`).
- To use another model, set `NER_MODEL_DIR` to either a Hub repo name (`username/repo-name`) or a local folder that has `config.json` directly inside it (for example `models/checkpoints/<run_name>`).
- The repo or folder must contain the model weights (`model.safetensors`) and the tokenizer files. If the Hub repo is private, the download fails.
- After the model is cached, `HF_HUB_OFFLINE=1` makes the server use only the cached copy.

## 4. Running the API

Run from the project root, because the imports (`from src.infer import ...`) depend on it:

```bash
uvicorn api.main:app --reload
```

- The model loads once when the server starts, not on every request.
- Interactive docs, where each endpoint can be tried in the browser: http://127.0.0.1:8000/docs
- `--reload` restarts the server when a file changes. Leave it off outside development.

### Environment variables

| Variable | Meaning | Default |
| :--- | :--- | :--- |
| `NER_MODEL_DIR` | Model to load: local folder or Hub repo name | `VeeJack/malaysian-ner-distilbert` |
| `CORS_ORIGINS` | Comma-separated origins allowed to call the API | `*` (any origin) |

Setting them in PowerShell:

```powershell
$env:NER_MODEL_DIR = "models/checkpoints/<run_name>"
$env:CORS_ORIGINS = "http://127.0.0.1:5500"
uvicorn api.main:app --reload
```

And in bash:

```bash
export NER_MODEL_DIR=models/checkpoints/<run_name>
export CORS_ORIGINS=http://127.0.0.1:5500
uvicorn api.main:app --reload
```

## 5. API Reference

All endpoints take and return JSON. The model is chosen when the server starts (`NER_MODEL_DIR`), not per request.

### `GET /health`

Checks that the server is running.

```json
{"status": "ok", "model": "malaysian-ner-distilbert"}
```

`model` is the last part of the model path or repo name.

### `POST /extract`

Returns the entities found in a text.

**Request**

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `text` | string | yes | Text to analyse, up to 100,000 characters |
| `min_score` | number | no | Only return entities with a confidence of at least this value (0 to 1, default `0`) |

**Response**

```json
{
  "entities": [
    {"entity": "Ahmad bin Ali", "type": "PER", "score": 0.99, "start": 0, "end": 13}
  ]
}
```

| Entity field | Description |
| :--- | :--- |
| `entity` | The text of the entity |
| `type` | `PER` (person), `ORG` (organization) or `LOC` (location) |
| `score` | The model's confidence, 0 to 1, rounded to 3 decimals |
| `start`, `end` | Character positions in the submitted text (`end` is exclusive), so `text[start:end]` gives the entity |

The score is the model's confidence, not the probability that the entity is correct.

### `POST /redact`

Returns the text with each entity replaced, plus the entities that were redacted.

**Request**: the same fields as `/extract`, plus:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `style` | `"label"` or `"mask"` | `"label"` | `label` replaces each entity with the placeholder; `mask` replaces each character with `█` |
| `placeholder` | string | `"[{type}]"` | Replacement format for `label` style (up to 100 characters) |

The placeholder can use only two fields: `{type}` (the entity type) and `{score}` (the confidence), with an optional number format such as `{score:.2f}`. Any other field is rejected with a 422 error. The `mask` style ignores the placeholder.

| Placeholder | Result for "Ahmad bin Ali" |
| :--- | :--- |
| `[{type}]` | `[PER]` |
| `<{type}>` | `<PER>` |
| `[REDACTED-{type}]` | `[REDACTED-PER]` |
| `[{type} {score:.2f}]` | `[PER 0.99]` |

**Example**

Request:

```json
{"text": "Ahmad bin Ali works at Petronas.", "style": "label"}
```

Response (the scores are illustrative):

```json
{
  "redacted_text": "[PER] works at [ORG].",
  "entities": [
    {"entity": "Ahmad bin Ali", "type": "PER", "score": 0.99, "start": 0, "end": 13},
    {"entity": "Petronas", "type": "ORG", "score": 0.97, "start": 23, "end": 31}
  ]
}
```

`entities` lists the entities that were redacted, so with `min_score` set, entities below the threshold are neither redacted nor listed.

### Errors

| Status | Cause |
| :--- | :--- |
| `422` | Invalid request: missing `text`, text longer than 100,000 characters, `min_score` outside 0 to 1, an unknown `style`, or an invalid placeholder |
| `500` | The model failed while running |

A validation error returns a `detail` field describing the problem.

### Trying it

In PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/redact `
  -ContentType "application/json" `
  -Body '{"text": "Ahmad bin Ali works at Petronas."}'
```

In bash:

```bash
curl -X POST http://127.0.0.1:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "Ahmad bin Ali works at Petronas."}'
```

Or open http://127.0.0.1:8000/docs and use the "Try it out" button.

## 6. How the Inference Works (`src/infer.py`)

| Function | Purpose |
| :--- | :--- |
| `get_model_dir` | Picks the model: the `model_dir` argument, then `NER_MODEL_DIR`, then the default |
| `load_ner_pipeline` | Loads the Hugging Face pipeline once and caches it, so later calls reuse it |
| `predict` | Runs the model on a text and returns the entities with positions in the original text |
| `split_long_line`, `find_break` | Cut over-long lines into chunks the model can read |
| `redact_text` | Replaces entities in the text with a placeholder or block characters |
| `print_table`, `main` | Command-line tool |

**Why these choices were made**

- **`aggregation_strategy="first"`.** During training, only the first subword piece of each word has a label. Taking each word's prediction from its first piece matches that, and keeps names such as `Iman` whole. The `"simple"` strategy trusts the later pieces, which the model never learned, and can split names into fragments.
- **Splitting long text.** The model reads at most 512 tokens and silently ignores the rest, which for redaction would leave names unredacted. `predict` first splits the text into lines. Any line longer than `MAX_CHUNK_TOKENS` (400) is cut into chunks, preferring the end of a sentence, otherwise the start of a word, so names are not cut in half. Each entity's position is shifted back to the original text.
- **Redacting from the end.** `redact_text` replaces entities starting from the last one, so earlier replacements do not move the positions of later ones.
- **Model loaded once.** `load_ner_pipeline` is cached. The API loads it at startup and every request reuses it.

### Command-line tool

The same code can be used without the API, from the project root:

```bash
python -m src.infer "Ahmad bin Ali works at Petronas in Kuala Lumpur."
python -m src.infer --file sample.txt
python -m src.infer --redact "Ahmad bin Ali works at Petronas."
python -m src.infer --redact --style mask "Ahmad bin Ali works at Petronas."
python -m src.infer --redact --placeholder "[{type} {score:.2f}]" --min-score 0.9 "Some text"
python -m src.infer --model models/checkpoints/<run_name> "Some text"
```

| Option | Meaning |
| :--- | :--- |
| `--file` | Read the text from a UTF-8 file |
| `--model` | Local folder or Hub repo name of the model |
| `--redact` | Print the redacted text instead of the table of entities |
| `--style` | `label` (default) or `mask` |
| `--placeholder` | Label format using `{type}` and `{score}` |
| `--min-score` | Only redact entities at or above this confidence (the table still lists all entities) |

## 7. Connecting the Frontend

The frontend sends the text with `fetch` and fills its table from the response:

```js
const response = await fetch('http://127.0.0.1:8000/redact', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: textValue, style: 'label' })
});
const result = await response.json();
// result.entities -> results table, result.redacted_text -> redacted output
```

There are two ways to serve the page:

Default method:
- **Separate server (Live Server, a file):** the page and the API have different origins, so the browser applies CORS. The default setting (`*`) allows this during development. Set `CORS_ORIGINS` to the page's address before sharing the API.
Alternative method:
- **Served by FastAPI (no CORS needed):** put the frontend files in a `frontend/` folder and add this at the very end of `api/main.py`, after all routes:

  ```python
  from fastapi.staticfiles import StaticFiles
  app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
  ```
  Then open http://127.0.0.1:8000. The mount must come last, or it hides the API routes.

- Meaning: A static file mount tells FastAPI to serve a folder of ready-made files (HTML, CSS, JavaScript, images) straight to the browser, with no Python code running for each file. "Static" means the files are sent exactly as they are on disk, and "mount" means attaching that folder to a URL path in your app.

**Is it needed?**
- No, it's optional. Without it, you run the page separately (Live Server, or opening the file) and call the API at http://127.0.0.1:8000, which works because CORS allows any origin by default. The mount is just the simpler setup for demos. It also isn't aimed at heavy traffic, so for a real deployment people often put the frontend on a separate static host.

Render results with `textContent`, not `innerHTML`, because the entity text comes from the user's document.

## 8. Configuration Summary

| Setting | Where | Value |
| :--- | :--- | :--- |
| Model | `NER_MODEL_DIR` / `DEFAULT_MODEL_DIR` in `src/infer.py` | `VeeJack/malaysian-ner-distilbert` |
| Allowed origins | `CORS_ORIGINS` | `*` |
| Maximum text length | `MAX_TEXT_CHARS` in `api/main.py` | 100,000 characters |
| Chunk size | `MAX_CHUNK_TOKENS` in `src/infer.py` | 400 tokens |
| Aggregation strategy | `load_ner_pipeline` in `src/infer.py` | `"first"` |
| Default placeholder | `DEFAULT_PLACEHOLDER` in `src/infer.py` | `[{type}]` |

The model runs on a GPU if one is available, otherwise on the CPU.

## 9. Limitations and Notes

- **Only three entity types.** The model detects people, organizations and locations. It does not find IC numbers, phone numbers, emails or addresses, so these stay in redacted text unless handled separately (for example with pattern matching).
- **Redaction is not guaranteed.** Any entity the model misses stays in the output. See the model documentation for evaluation results, and do not rely on this for cases where a leak matters without checking its recall on your own text.
- **Entities across chunk boundaries.** Long text is split without overlap. Cuts are placed at sentence ends or word starts to make this rare, but an entity that straddles a cut can still be missed.
- **Confidence is not accuracy.** Scores for Malay names tend to be lower than for English ones, so a high `min_score` removes real entities.
- **Model updates.** The Hub model is loaded at its latest version, so the API's results can change if its owner updates the repo.
- **No authentication.** Anyone who can reach the server can use it. Do not expose it publicly without adding some form of access control and HTTPS.
- **Privacy.** The inputs may be personal data. Do not log request bodies or send them to third-party services.
- **Concurrency.** The endpoints are regular (not async) functions, so FastAPI runs them in a thread pool, and all requests share one model. Long documents on a CPU can take several seconds.

## 10. Troubleshooting

| Problem | Likely cause and fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Run `uvicorn` from the project root, and check that `src/__init__.py` exists |
| `ImportError: cannot import name 'DEFAULT_PLACEHOLDER'` | `src/infer.py` is an older version without that constant. Update it |
| Server fails at startup with a Hub or "repository not found" error | The repo name is wrong or private, or there is no internet on the first run |
| Server is slow to start | The model is downloading or loading. This is normal on the first run |
| Browser shows "Cannot reach the API" | The server isn't running, the URL or port is wrong, or the origin isn't in `CORS_ORIGINS` |
| `422` from `/redact` | The placeholder uses a field other than `{type}` or `{score}`, or has a bad number format |
| Names come back in pieces (`Ima` and `##n`) | The pipeline is using `aggregation_strategy="simple"`. Use `"first"` |
| Changes to `src/infer.py` have no effect | The running server has the old code. Restart it, or use `--reload` |
