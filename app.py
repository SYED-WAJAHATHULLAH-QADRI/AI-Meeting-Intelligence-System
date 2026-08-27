# ============================================================
# AI MEETING INTELLIGENCE SYSTEM
# Complete Streamlit application
# ============================================================

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Literal, Optional

import imageio_ffmpeg
import streamlit as st
import torch
import whisper

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# FFMPEG CONFIGURATION
# This must run before Whisper transcription.
# ============================================================

def configure_ffmpeg():
    """
    Make an FFmpeg executable available to Whisper.

    First use the system FFmpeg installed through packages.txt.
    If unavailable, use imageio-ffmpeg as a fallback.
    """

    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    if not bundled_ffmpeg:
        raise RuntimeError(
            "No FFmpeg executable was found."
        )

    bundled_ffmpeg = Path(bundled_ffmpeg)

    if not bundled_ffmpeg.exists():
        raise FileNotFoundError(
            f"Bundled FFmpeg does not exist: {bundled_ffmpeg}"
        )

    ffmpeg_directory = Path("/tmp/ffmpeg-bin")

    ffmpeg_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    ffmpeg_link = ffmpeg_directory / "ffmpeg"

    if ffmpeg_link.exists() or ffmpeg_link.is_symlink():
        ffmpeg_link.unlink()

    try:
        ffmpeg_link.symlink_to(bundled_ffmpeg)

    except OSError:
        shutil.copy2(
            bundled_ffmpeg,
            ffmpeg_link
        )

    ffmpeg_link.chmod(0o755)

    current_path = os.environ.get("PATH", "")

    os.environ["PATH"] = (
        str(ffmpeg_directory)
        + os.pathsep
        + current_path
    )

    configured_ffmpeg = shutil.which("ffmpeg")

    if not configured_ffmpeg:
        raise RuntimeError(
            "FFmpeg fallback configuration failed."
        )

    return configured_ffmpeg


try:
    FFMPEG_PATH = configure_ffmpeg()
    FFMPEG_ERROR = None

except Exception as error:
    FFMPEG_PATH = None
    FFMPEG_ERROR = str(error)


# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# PYDANTIC OUTPUT SCHEMA
# ============================================================

class Decision(BaseModel):
    decision: str = Field(
        description="A confirmed meeting decision."
    )

    evidence: str = Field(
        description="Exact supporting transcript quotation."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class ActionItem(BaseModel):
    action: str = Field(
        description="An assigned, proposed or unclear task."
    )

    owner: Optional[str] = Field(
        default=None,
        description="Responsible person or null."
    )

    deadline: Optional[str] = Field(
        default=None,
        description="Stated deadline or null."
    )

    status: Literal[
        "assigned",
        "proposed",
        "unclear"
    ]

    evidence: str = Field(
        description="Exact supporting transcript quotation."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class MeetingIntelligence(BaseModel):
    meeting_title: str
    summary: str

    key_topics: List[str] = Field(
        default_factory=list
    )

    decisions: List[Decision] = Field(
        default_factory=list
    )

    action_items: List[ActionItem] = Field(
        default_factory=list
    )

    unresolved_issues: List[str] = Field(
        default_factory=list
    )

    ambiguities: List[str] = Field(
        default_factory=list
    )


# ============================================================
# PROMPTS
# ============================================================

GENERIC_PROMPT = """
Analyse the following meeting transcript.

Identify:
- meeting title;
- summary;
- key topics;
- decisions;
- action items;
- owners;
- deadlines;
- unresolved issues; and
- ambiguities.

Use only information contained in the transcript.

TRANSCRIPT:
{transcript}
"""


STRUCTURED_PROMPT = """
You are a context-aware meeting intelligence system.

Analyse only the supplied transcript.

Rules:

1. A decision must be a confirmed outcome.
2. Do not classify suggestions, opinions, possibilities or
   rejected proposals as confirmed decisions.
3. An assigned action must be explicitly assigned or accepted.
4. Use "proposed" when a task was only suggested.
5. Use "unclear" when responsibility is ambiguous.
6. Use null when an owner or deadline is not stated.
7. Evidence must be an exact transcript quotation.
8. Never invent decisions, tasks, owners or deadlines.
9. Put unanswered matters under unresolved_issues.
10. Put uncertain interpretations under ambiguities.
11. Use empty lists when no supported item exists.
12. Do not use outside knowledge.

TRANSCRIPT:
{transcript}
"""


# ============================================================
# TRANSCRIPT PROCESSING
# ============================================================

def clean_transcript(text):
    """
    Normalise whitespace without removing speaker labels,
    punctuation, dates or contextual information.
    """

    if not isinstance(text, str):
        raise TypeError(
            "The transcript must be text."
        )

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    cleaned_lines = []

    for line in text.splitlines():
        line = " ".join(line.split()).strip()

        if line:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    if not cleaned:
        raise ValueError(
            "The transcript is empty."
        )

    return cleaned


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """
    Load the Whisper model once and reuse it.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = whisper.load_model(
        "base.en",
        device=device
    )

    return model, device


def transcribe_uploaded_audio(uploaded_file):
    """
    Save uploaded audio temporarily and transcribe it using
    the real Whisper model.
    """

    if not FFMPEG_PATH:
        raise RuntimeError(
            "FFmpeg is unavailable. "
            f"Configuration error: {FFMPEG_ERROR}"
        )

    original_suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    supported_suffixes = {
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".webm"
    }

    suffix = (
        original_suffix
        if original_suffix in supported_suffixes
        else ".wav"
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=suffix
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getbuffer()
            )

            temporary_file.flush()

            temporary_path = temporary_file.name

        if not Path(temporary_path).exists():
            raise FileNotFoundError(
                "Temporary audio file was not created."
            )

        if Path(temporary_path).stat().st_size == 0:
            raise ValueError(
                "The uploaded audio file is empty."
            )

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

        transcript = result.get(
            "text",
            ""
        ).strip()

        if not transcript:
            raise RuntimeError(
                "Whisper returned an empty transcript."
            )

        return {
            "transcript": clean_transcript(transcript),
            "processing_seconds": processing_seconds,
            "device": device,
            "model": "base.en",
            "ffmpeg_path": FFMPEG_PATH
        }

    finally:
        if (
            temporary_path
            and Path(temporary_path).exists()
        ):
            Path(temporary_path).unlink()


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_meeting(
    transcript,
    api_key,
    model_name,
    prompt_type
):
    """
    Extract structured meeting intelligence using Gemini and
    validate it with the Pydantic schema.
    """

    transcript = clean_transcript(transcript)

    prompt_template = (
        STRUCTURED_PROMPT
        if prompt_type == "Structured"
        else GENERIC_PROMPT
    )

    prompt = prompt_template.format(
        transcript=transcript
    )

    client = genai.Client(
        api_key=api_key
    )

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
            validated_result = response.parsed

        else:
            validated_result = (
                MeetingIntelligence.model_validate(
                    response.parsed
                )
            )

    else:
        validated_result = (
            MeetingIntelligence.model_validate_json(
                response.text
            )
        )

    return {
        "result": validated_result.model_dump(),
        "processing_seconds": processing_seconds,
        "schema_valid": True,
        "model": model_name,
        "temperature": 0.2,
        "prompt_type": prompt_type
    }


# ============================================================
# REPORT CREATION
# ============================================================

def create_markdown_report(
    result,
    transcript,
    metadata
):
    lines = [
        f"# {result['meeting_title']}",
        "",
        "## System Information",
        "",
        f"- Prompt type: {metadata['prompt_type']}",
        f"- Gemini model: {metadata['model']}",
        f"- Temperature: {metadata['temperature']}",
        "- Schema validation: Passed",
        "",
        "## Summary",
        "",
        result["summary"],
        "",
        "## Key Topics",
        ""
    ]

    if result["key_topics"]:
        for topic in result["key_topics"]:
            lines.append(f"- {topic}")
    else:
        lines.append("- None identified")

    lines.extend([
        "",
        "## Decisions",
        ""
    ])

    if result["decisions"]:
        for number, decision in enumerate(
            result["decisions"],
            start=1
        ):
            lines.extend([
                f"### Decision {number}",
                "",
                decision["decision"],
                "",
                f"**Evidence:** {decision['evidence']}",
                "",
                f"**Confidence:** {decision['confidence']:.2f}",
                ""
            ])
    else:
        lines.append(
            "No confirmed decisions identified."
        )

    lines.extend([
        "",
        "## Action Items",
        ""
    ])

    if result["action_items"]:
        for number, action in enumerate(
            result["action_items"],
            start=1
        ):
            lines.extend([
                f"### Action {number}",
                "",
                f"**Task:** {action['action']}",
                "",
                f"**Owner:** "
                f"{action.get('owner') or 'Not stated'}",
                "",
                f"**Deadline:** "
                f"{action.get('deadline') or 'Not stated'}",
                "",
                f"**Status:** {action['status']}",
                "",
                f"**Evidence:** {action['evidence']}",
                "",
                f"**Confidence:** {action['confidence']:.2f}",
                ""
            ])
    else:
        lines.append(
            "No action items identified."
        )

    lines.extend([
        "",
        "## Unresolved Issues",
        ""
    ])

    if result["unresolved_issues"]:
        for issue in result["unresolved_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- None identified")

    lines.extend([
        "",
        "## Ambiguities",
        ""
    ])

    if result["ambiguities"]:
        for ambiguity in result["ambiguities"]:
            lines.append(f"- {ambiguity}")
    else:
        lines.append("- None identified")

    lines.extend([
        "",
        "## Reviewed Transcript",
        "",
        transcript,
        "",
        "---",
        "",
        "AI-generated draft. Human review is required."
    ])

    return "\n".join(lines)


# ============================================================
# SESSION STATE
# ============================================================

default_session_values = {
    "transcript": "",
    "result_record": None,
    "whisper_metadata": None
}

for key, default_value in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🧠 AI Meeting Intelligence System")

st.write(
    "Convert meeting audio or transcript text into structured, "
    "reviewable decisions and action items."
)

st.warning(
    "AI-generated outputs are drafts. Always check decisions, "
    "owners and deadlines against the original meeting."
)


# Obtain the Gemini API key securely.

try:
    GEMINI_API_KEY = st.secrets[
        "GEMINI_API_KEY"
    ]

except Exception:
    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY"
    )


# Sidebar

with st.sidebar:
    st.header("System Settings")

    input_type = st.radio(
        "Meeting input",
        [
            "Audio file",
            "Transcript text"
        ]
    )

    prompt_type = st.selectbox(
        "Prompt method",
        [
            "Structured",
            "Generic"
        ]
    )

    model_name = st.selectbox(
        "Gemini model",
        [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.0-flash-001"
        ],
        index=0
    )

    st.caption(
        "Temperature is fixed at 0.2 for controlled comparison."
    )

    st.divider()

    st.subheader("System status")

    if FFMPEG_PATH:
        st.success("FFmpeg available")
        st.caption(FFMPEG_PATH)
    else:
        st.error("FFmpeg unavailable")
        st.caption(FFMPEG_ERROR or "Unknown error")

    if GEMINI_API_KEY:
        st.success("Gemini key configured")
    else:
        st.error("Gemini key missing")


# Input section

if input_type == "Audio file":
    uploaded_audio = st.file_uploader(
        "Upload meeting audio",
        type=[
            "wav",
            "mp3",
            "m4a",
            "mp4",
            "mpeg",
            "mpga",
            "webm"
        ]
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button(
            "Transcribe audio",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Loading Whisper and transcribing audio..."
                ):
                    transcription_record = (
                        transcribe_uploaded_audio(
                            uploaded_audio
                        )
                    )

                st.session_state["transcript"] = (
                    transcription_record["transcript"]
                )

                st.session_state[
                    "whisper_metadata"
                ] = transcription_record

                st.session_state["result_record"] = None

                st.success(
                    "Whisper transcription completed."
                )

            except Exception as error:
                st.error(
                    f"Audio transcription failed: {error}"
                )

else:
    pasted_transcript = st.text_area(
        "Paste the meeting transcript",
        value=st.session_state["transcript"],
        height=280,
        placeholder=(
            "Aisha: We agreed to retain the launch date.\n"
            "Manager: Bilal, complete the security test by Friday."
        )
    )

    if pasted_transcript.strip():
        try:
            st.session_state["transcript"] = (
                clean_transcript(
                    pasted_transcript
                )
            )
        except ValueError:
            pass


# Transcript review section

if st.session_state["transcript"]:
    st.divider()
    st.subheader("Transcript review")

    reviewed_transcript = st.text_area(
        "Review and correct the transcript before extraction",
        value=st.session_state["transcript"],
        height=300,
        key="reviewed_transcript"
    )

    if st.session_state["whisper_metadata"]:
        whisper_metadata = st.session_state[
            "whisper_metadata"
        ]

        column_one, column_two, column_three = (
            st.columns(3)
        )

        column_one.metric(
            "Whisper model",
            whisper_metadata["model"]
        )

        column_two.metric(
            "Processing time",
            f"{whisper_metadata['processing_seconds']:.2f}s"
        )

        column_three.metric(
            "Device",
            whisper_metadata["device"]
        )

    if st.button(
        "Generate meeting intelligence",
        type="primary",
        use_container_width=True
    ):
        if not GEMINI_API_KEY:
            st.error(
                "GEMINI_API_KEY is missing. Add it through "
                "Streamlit Manage app → Settings → Secrets."
            )

        else:
            try:
                cleaned_reviewed_transcript = (
                    clean_transcript(
                        reviewed_transcript
                    )
                )

                with st.spinner(
                    "Running Gemini extraction and schema validation..."
                ):
                    result_record = extract_meeting(
                        transcript=cleaned_reviewed_transcript,
                        api_key=GEMINI_API_KEY,
                        model_name=model_name,
                        prompt_type=prompt_type
                    )

                st.session_state["transcript"] = (
                    cleaned_reviewed_transcript
                )

                st.session_state["result_record"] = (
                    result_record
                )

                st.success(
                    "Gemini extraction completed and "
                    "schema validation passed."
                )

            except Exception as error:
                st.error(
                    f"Gemini extraction failed: {error}"
                )


# Results section

if st.session_state["result_record"]:
    result_record = st.session_state[
        "result_record"
    ]

    result = result_record["result"]

    st.divider()
    st.header(result["meeting_title"])

    metric_one, metric_two, metric_three = (
        st.columns(3)
    )

    metric_one.metric(
        "Gemini time",
        f"{result_record['processing_seconds']:.2f}s"
    )

    metric_two.metric(
        "Prompt",
        result_record["prompt_type"]
    )

    metric_three.metric(
        "Schema",
        "Passed"
    )

    (
        summary_tab,
        decisions_tab,
        actions_tab,
        review_tab,
        json_tab
    ) = st.tabs([
        "Summary",
        "Decisions",
        "Action Items",
        "Review",
        "JSON"
    ])

    with summary_tab:
        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Key topics")

        if result["key_topics"]:
            for topic in result["key_topics"]:
                st.write(f"• {topic}")
        else:
            st.info("No key topics identified.")

    with decisions_tab:
        if not result["decisions"]:
            st.info(
                "No confirmed decisions identified."
            )

        for number, decision in enumerate(
            result["decisions"],
            start=1
        ):
            with st.container(border=True):
                st.markdown(
                    f"### Decision {number}"
                )

                st.write(
                    decision["decision"]
                )

                st.caption(
                    f"Evidence: {decision['evidence']}"
                )

                st.write(
                    f"Confidence: "
                    f"{decision['confidence']:.2f}"
                )

                st.progress(
                    float(decision["confidence"])
                )

    with actions_tab:
        if not result["action_items"]:
            st.info(
                "No action items identified."
            )

        for number, action in enumerate(
            result["action_items"],
            start=1
        ):
            with st.container(border=True):
                st.markdown(
                    f"### Action {number}"
                )

                st.write(action["action"])

                owner_column, deadline_column = (
                    st.columns(2)
                )

                owner_column.write(
                    "**Owner:** "
                    + (
                        action.get("owner")
                        or "Not stated"
                    )
                )

                deadline_column.write(
                    "**Deadline:** "
                    + (
                        action.get("deadline")
                        or "Not stated"
                    )
                )

                st.write(
                    f"**Status:** {action['status']}"
                )

                st.caption(
                    f"Evidence: {action['evidence']}"
                )

                st.write(
                    f"Confidence: "
                    f"{action['confidence']:.2f}"
                )

                st.progress(
                    float(action["confidence"])
                )

    with review_tab:
        st.subheader("Unresolved issues")

        if result["unresolved_issues"]:
            for issue in result["unresolved_issues"]:
                st.write(f"• {issue}")
        else:
            st.write("None identified.")

        st.subheader("Ambiguities")

        if result["ambiguities"]:
            for ambiguity in result["ambiguities"]:
                st.write(f"• {ambiguity}")
        else:
            st.write("None identified.")

        st.subheader("Reviewed transcript")

        st.text_area(
            "Source transcript",
            value=st.session_state["transcript"],
            height=250,
            disabled=True
        )

    with json_tab:
        st.json(result)


    # Downloads

    markdown_report = create_markdown_report(
        result=result,
        transcript=st.session_state["transcript"],
        metadata=result_record
    )

    json_report = json.dumps(
        result_record,
        indent=2,
        ensure_ascii=False
    )

    st.subheader("Download reports")

    json_column, markdown_column = (
        st.columns(2)
    )

    json_column.download_button(
        label="Download JSON report",
        data=json_report,
        file_name="meeting_report.json",
        mime="application/json",
        use_container_width=True
    )

    markdown_column.download_button(
        label="Download Markdown report",
        data=markdown_report,
        file_name="meeting_report.md",
        mime="text/markdown",
        use_container_width=True
    )
