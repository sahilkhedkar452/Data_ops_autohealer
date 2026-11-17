"""DataOps Auto-Healer: retrieve similar incident records and draft safe diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
CASES_PATH = Path(__file__).with_name("failure_cases.json")


@st.cache_data
def load_cases() -> list[dict[str, str]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def retrieve(log_text: str, cases: list[dict[str, str]], top_k: int = 3) -> list[tuple[dict[str, str], float]]:
    documents = [f"{case['symptoms']} {case['cause']} {case['fix']}" for case in cases]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(documents + [log_text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [(cases[int(index)], float(scores[int(index)])) for index in indices]


def llm_suggestion(log_text: str, matches: list[tuple[dict[str, str], float]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ""
    from openai import OpenAI

    examples = "\n\n".join(f"Incident: {case['symptoms']}\nCause: {case['cause']}\nSuggested pattern: {case['fix']}\nExample: {case['code']}" for case, _ in matches)
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a cautious data-pipeline incident assistant. Diagnose only from the log and retrieved examples. Suggest a minimal code pattern, label uncertainty, and never claim you executed a fix. Never request or expose secrets."},
            {"role": "user", "content": f"Pipeline log:\n{log_text}\n\nRetrieved historical examples:\n{examples}"},
        ],
    )
    return response.choices[0].message.content or "No suggestion was returned."


def main() -> None:
    st.set_page_config(page_title="DataOps Auto-Healer", page_icon="🛠️", layout="wide")
    st.title("🛠️ DataOps Auto-Healer")
    st.caption("A learning prototype that matches pipeline logs to historical failure patterns and drafts remediation ideas.")
    cases = load_cases()
    with st.sidebar:
        st.header("Try a sample pipeline")
        failure = st.selectbox("Inject a failure", ["none", "schema_missing_column", "schema_type_drift", "empty_input"])
        if st.button("Run sample pipeline", type="primary", use_container_width=True):
            try:
                from sample_pipeline import run_pipeline
                output = run_pipeline(failure)
                st.session_state.pipeline_log = f"Pipeline completed successfully. Output rows: {len(output)}"
                st.session_state.pipeline_output = output
            except Exception as exc:
                st.session_state.pipeline_log = f"{type(exc).__name__}: {exc}"
                st.session_state.pop("pipeline_output", None)
    if "pipeline_log" in st.session_state:
        st.subheader("Sample run")
        st.code(st.session_state.pipeline_log)
        if "pipeline_output" in st.session_state:
            st.dataframe(st.session_state.pipeline_output, use_container_width=True)

    uploaded = st.file_uploader("Or upload a plain-text pipeline log", type=["txt", "log"])
    default_log = st.session_state.get("pipeline_log", "")
    log_text = st.text_area("Pipeline error or log excerpt", value=(uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else default_log), height=160, placeholder="Paste a sanitized stack trace or error message…")
    if st.button("Analyze failure", disabled=not log_text.strip()):
        matches = retrieve(log_text.strip(), cases)
        st.subheader("Closest historical incidents")
        for case, score in matches:
            with st.expander(f"{case['id']} · similarity {score:.2f}", expanded=True):
                st.markdown(f"**Observed symptoms:** {case['symptoms']}\n\n**Likely cause:** {case['cause']}\n\n**Suggested approach:** {case['fix']}")
                st.code(case["code"], language="python")
        try:
            with st.spinner("Preparing a diagnostic suggestion…"):
                suggestion = llm_suggestion(log_text.strip(), matches)
            if suggestion:
                st.subheader("LLM-assisted diagnostic draft")
                st.write(suggestion)
            else:
                st.info("No OPENAI_API_KEY is configured. The retrieved incident patterns above are the full analysis.")
        except Exception as exc:
            st.warning(f"The LLM request failed; the retrieved examples are still available. Details: {exc}")
    st.divider()
    st.caption("Suggestions are drafts for a developer to review. This prototype never modifies source code or reruns production jobs.")


if __name__ == "__main__":
    main()
