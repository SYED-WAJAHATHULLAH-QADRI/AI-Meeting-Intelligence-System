# AI Meeting Intelligence System

An end-to-end meeting-intelligence system that converts meeting
audio or transcripts into structured, reviewable outputs.

## Implemented Pipeline

1. Audio or transcript input
2. Whisper speech transcription
3. Transcript cleaning
4. Gemini-based contextual extraction
5. Pydantic schema validation
6. Decisions and action-item presentation
7. JSON, Markdown, Word and PDF report export
8. Performance and extraction visualisations

## Extracted Information

- Meeting summary
- Key topics
- Confirmed decisions
- Action items
- Owners
- Deadlines
- Supporting evidence
- Unresolved issues
- Ambiguities

## Privacy

API keys are not stored in this repository. Configure the
`GEMINI_API_KEY` as a secure environment variable or Streamlit secret.

## Human Review

All AI-generated meeting records are drafts and must be checked
against the original transcript before professional use.

## Current Model Configuration

- Speech recognition: Whisper `small.en`
- Language model: Gemini Flash model selected according to API access
- Structured validation: Pydantic
- Interface: Streamlit
