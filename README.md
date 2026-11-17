# DataOps Auto-Healer

An academic prototype for pipeline failure diagnosis. It retrieves similar historical failure cases from a small local knowledge base and can ask an optional LLM to draft a context-aware diagnostic. A sample pipeline can inject missing-column, type-drift, and empty-input failures.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Optionally copy `.env.example` to `.env` and configure `OPENAI_API_KEY` for LLM-assisted drafts.
5. Run `streamlit run app.py`.

The app only displays suggestions; it does not change code or apply repairs. Logs may contain credentials or personal information, so sanitize them before pasting. The bundled failure cases are illustrative examples, not a production incident database.
