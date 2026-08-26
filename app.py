
import json
import os
import tempfile
import time
from typing import List, Optional, Literal

import streamlit as st
import torch
import whisper

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class Decision(BaseModel):
    decision: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class ActionItem(BaseModel):
    action: str
    owner: Optional[str] = None
    deadline: Optional[str] = None

    status: Literal[
        "assigned",
        "proposed",
        "unclear"
    ]

    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class MeetingIntelligence(BaseModel):
    meeting_title: str
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    unresolved_issues: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)


# ============================================================
# PROMPTS
# ============================================================

GENERIC_PROMPT = """
Analyse the meeting transcript.

Identify the summary, topics, decisions, action items,
owners, deadlines, unresolved issues and ambiguities.

Use only information from the transcript.

TRANSCRIPT:
{transcript}
"""


STRUCTURED_PROMPT = """
You are a context-aware meeting intelligence system.

Follow these rules:

1. A decision must be a confirmed outcome.
2. Do not treat suggestions, opinions, possibilities or
   rejected proposals as confirmed decisions.
3. An assigned action must be explicitly assigned or accepted.
4. Use proposed when a task was only suggested.
5. Use unclear when responsibility is ambiguous.
6. Use null when an owner or deadline is not stated.
7. Evidence must be an exact transcript quotation.
8. Never invent decisions, actions, owners or deadlines.
9. Put unanswered matters under unresolved issues.
10. Put uncertain interpretations under ambiguities.
11. Use empty lists when no supported items exist.

TRANSCRIPT:
{transcript}
"""


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def clean_transcript(text):
    lines = []

    for line in text.replace("\r", "\n").splitlines():
        line = " ".join(line.split()).strip()

        if line:
            lines.append(line)

    cleaned = "\n".join(lines).strip()

    if not cleaned:
        raise ValueError("The transcript is empty.")

    return cleaned


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(
        "base.en",
        device=device
    )

    return model, device


def transcribe_uploaded_audio(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temporary_file:

        temporary_file.write(
            uploaded_file.getbuffer()
        )

        temporary_path = temporary_file.name

    try:
        model, device = load_whisper_model()

        started = time.perf_counter()

        result = model.transcribe(
            temporary_path,
            language="en",
            task="transcribe",
            temperature=0.0,
            fp16=(device == "cuda"),
            verbose=False
        )

        processing_seconds = (
            time.perf_counter() - started
        )

        transcript = result.get("text", "").strip()

        if not transcript:
            raise RuntimeError(
                "Whisper returned an empty transcript."
            )

        return transcript, processing_seconds, device

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def extract_meeting(
    transcript,
    api_key,
    model_name,
    prompt_type
):
    prompt_template = (
        STRUCTURED_PROMPT
        if prompt_type == "Structured"
        else GENERIC_PROMPT
    )

    prompt = prompt_template.format(
        transcript=transcript
    )

    client = genai.Client(api_key=api_key)

    started = time.perf_counter()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=MeetingIntelligence
        )
    )

    processing_seconds = (
        time.perf_counter() - started
    )

    if response.parsed is not None:
        if isinstance(
            response.parsed,
            MeetingIntelligence
        ):
            validated = response.parsed
        else:
            validated = (
                MeetingIntelligence.model_validate(
                    response.parsed
                )
            )
    else:
        validated = (
            MeetingIntelligence.model_validate_json(
                response.text
            )
        )

    return validated.model_dump(), processing_seconds


def create_markdown(result, transcript):
    lines = [
        f"# {result['meeting_title']}",
        "",
        "## Summary",
        result["summary"],
        "",
        "## Key Topics"
    ]

    for topic in result["key_topics"]:
        lines.append(f"- {topic}")

    lines.extend(["", "## Decisions"])

    if result["decisions"]:
        for item in result["decisions"]:
            lines.extend([
                f"- {item['decision']}",
                f"  - Evidence: {item['evidence']}",
                f"  - Confidence: {item['confidence']}"
            ])
    else:
        lines.append("- No confirmed decisions identified.")

    lines.extend(["", "## Action Items"])

    if result["action_items"]:
        for item in result["action_items"]:
            lines.extend([
                f"- Task: {item['action']}",
                f"  - Owner: {item.get('owner') or 'Not stated'}",
                f"  - Deadline: {item.get('deadline') or 'Not stated'}",
                f"  - Status: {item['status']}",
                f"  - Evidence: {item['evidence']}"
            ])
    else:
        lines.append("- No action items identified.")

    lines.extend([
        "",
        "## Unresolved Issues"
    ])

    for item in result["unresolved_issues"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Ambiguities"
    ])

    for item in result["ambiguities"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Reviewed Transcript",
        transcript,
        "",
        "---",
        "AI-generated draft. Human review is required."
    ])

    return "\n".join(lines)


# ============================================================
# INTERFACE
# ============================================================

st.title("🧠 AI Meeting Intelligence System")

st.write(
    "Convert meeting audio or transcripts into structured, "
    "reviewable decisions and action items."
)

st.warning(
    "AI-generated outputs must be checked against the "
    "original recording or transcript before use."
)

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    st.header("System Settings")

    input_type = st.radio(
        "Meeting input",
        ["Audio file", "Transcript text"]
    )

    prompt_type = st.selectbox(
        "Prompt method",
        ["Structured", "Generic"]
    )

    model_name = st.selectbox(
        "Gemini model",
        [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.0-flash-001"
        ],
        index=1
    )

    st.caption(
        "Temperature is fixed at 0.2 for controlled evaluation."
    )


if input_type == "Audio file":
    uploaded_audio = st.file_uploader(
        "Upload meeting audio",
        type=["wav", "mp3", "m4a", "mp4"]
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button(
            "Transcribe audio",
            type="primary"
        ):
            try:
                with st.spinner(
                    "Running Whisper transcription..."
                ):
                    transcript, whisper_time, device = (
                        transcribe_uploaded_audio(
                            uploaded_audio
                        )
                    )

                st.session_state["transcript"] = transcript
                st.session_state["whisper_time"] = whisper_time
                st.session_state["whisper_device"] = device

                st.success(
                    "Whisper transcription completed."
                )

            except Exception as error:
                st.error(
                    f"Transcription failed: {error}"
                )

else:
    pasted_transcript = st.text_area(
        "Paste the meeting transcript",
        height=300,
        placeholder=(
            "Speaker 1: Let us confirm the launch date..."
        )
    )

    if pasted_transcript.strip():
        st.session_state["transcript"] = (
            clean_transcript(pasted_transcript)
        )


if "transcript" in st.session_state:
    st.subheader("Transcript review")

    reviewed_transcript = st.text_area(
        "Review and correct the transcript",
        value=st.session_state["transcript"],
        height=300
    )

    st.session_state["transcript"] = (
        clean_transcript(reviewed_transcript)
    )

    if "whisper_time" in st.session_state:
        first, second = st.columns(2)

        first.metric(
            "Whisper time",
            f"{st.session_state['whisper_time']:.2f} seconds"
        )

        second.metric(
            "Whisper device",
            st.session_state["whisper_device"]
        )

    if st.button(
        "Generate meeting intelligence",
        type="primary"
    ):
        if not API_KEY:
            st.error(
                "GEMINI_API_KEY is missing from "
                "Streamlit Secrets."
            )

        else:
            try:
                with st.spinner(
                    "Running Gemini extraction..."
                ):
                    result, gemini_time = extract_meeting(
                        transcript=reviewed_transcript,
                        api_key=API_KEY,
                        model_name=model_name,
                        prompt_type=prompt_type
                    )

                st.session_state["result"] = result
                st.session_state["gemini_time"] = gemini_time

                st.success(
                    "Gemini extraction and schema "
                    "validation completed."
                )

            except Exception as error:
                st.error(
                    f"Gemini extraction failed: {error}"
                )


if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()
    st.header(result["meeting_title"])

    st.metric(
        "Gemini processing time",
        f"{st.session_state['gemini_time']:.2f} seconds"
    )

    summary_tab, decision_tab, action_tab, review_tab = (
        st.tabs([
            "Summary",
            "Decisions",
            "Action Items",
            "Review"
        ])
    )

    with summary_tab:
        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Key topics")

        for topic in result["key_topics"]:
            st.write(f"• {topic}")

    with decision_tab:
        if not result["decisions"]:
            st.info(
                "No confirmed decisions identified."
            )

        for number, item in enumerate(
            result["decisions"],
            start=1
        ):
            with st.container(border=True):
                st.markdown(
                    f"### Decision {number}"
                )
                st.write(item["decision"])
                st.caption(
                    f"Evidence: {item['evidence']}"
                )
                st.progress(item["confidence"])

    with action_tab:
        if not result["action_items"]:
            st.info("No action items identified.")

        for number, item in enumerate(
            result["action_items"],
            start=1
        ):
            with st.container(border=True):
                st.markdown(f"### Action {number}")
                st.write(item["action"])

                owner_column, deadline_column = (
                    st.columns(2)
                )

                owner_column.write(
                    "**Owner:** "
                    + (item.get("owner") or "Not stated")
                )

                deadline_column.write(
                    "**Deadline:** "
                    + (
                        item.get("deadline")
                        or "Not stated"
                    )
                )

                st.write(
                    f"**Status:** {item['status']}"
                )

                st.caption(
                    f"Evidence: {item['evidence']}"
                )

    with review_tab:
        st.subheader("Unresolved issues")

        if result["unresolved_issues"]:
            for item in result["unresolved_issues"]:
                st.write(f"• {item}")
        else:
            st.write("None identified.")

        st.subheader("Ambiguities")

        if result["ambiguities"]:
            for item in result["ambiguities"]:
                st.write(f"• {item}")
        else:
            st.write("None identified.")

    markdown_report = create_markdown(
        result,
        st.session_state["transcript"]
    )

    json_report = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.subheader("Download reports")

    json_column, markdown_column = st.columns(2)

    json_column.download_button(
        "Download JSON report",
        data=json_report,
        file_name="meeting_report.json",
        mime="application/json",
        use_container_width=True
    )

    markdown_column.download_button(
        "Download Markdown report",
        data=markdown_report,
        file_name="meeting_report.md",
        mime="text/markdown",
        use_container_width=True
    )
