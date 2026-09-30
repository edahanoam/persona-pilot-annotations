# Persona Pilot Annotations

A small Streamlit app that presents a welcome screen, draws questions in a random order, and appends each submitted response to a JSONL file.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Responses are written to `data/responses.jsonl`. Each line is a complete JSON object, so a successfully submitted response is retained even if the browser closes before the participant finishes.

To use another output location, set `RESULTS_PATH` before starting the app:

```powershell
$env:RESULTS_PATH = "C:\path\to\responses.jsonl"
streamlit run app.py
```

## Edit the question set

Update `questions.json`. Every item must have a unique `id`, a `question`, and a `description`:

```json
{
  "id": "unique-id",
  "question": "The question shown to the participant",
  "description": "Extra context or guidance"
}
```

