# Changelog

## 2026-05-31

### Added
- Structured JSON logging to stdout for Cloud Run ingestion.
- Request lifecycle logs in chat flow:
  - `request_started`
  - `request_completed`
- OpenAI call telemetry logs:
  - `openai_call_completed`
  - `openai_call_failed`
- Retrieval telemetry log:
  - `retrieval_completed`
- Upload telemetry logs:
  - `document_uploaded`
  - `document_uploaded_api`
  - `document_parse_failed`
  - `upload_failed`
- Health endpoint log event:
  - `healthcheck`

### Notes
- Logging avoids secrets and focuses on operational metadata (durations, sizes, counts, error types).
- Log format is JSON and suitable for Cloud Logging queries via `jsonPayload.event`.
