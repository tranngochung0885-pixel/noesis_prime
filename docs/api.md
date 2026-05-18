# API Documentation

NOESIS PRIME exposes an optional FastAPI server for experimentation.

## Run the API

```bash
python noesis_prime.py serve --host 127.0.0.1 --port 8000
```

## Endpoints

### `GET /healthz`
Returns service health.

### `POST /step`
Runs one full cognitive cycle.

Example request:

```json
{
  "session_id": "default",
  "observation": "What is predictive coding?",
  "reward": 0.0,
  "done": false
}
```

### `POST /chat`
Simplified chat interface.

Example request:

```json
{
  "session_id": "default",
  "message": "Explain memory reconsolidation.",
  "reward": 0.0
}
```

### `POST /inject`
Inject knowledge into memory.

### `POST /search`
Semantic search over memory.

### `POST /nostalgize`
Generate a nostalgia trace and narrative.

### `POST /goal`
Add a persistent goal.

### `GET /introspect/{session_id}`
Returns a structured internal state snapshot.

### `GET /sessions`
Lists active sessions.

### `DELETE /sessions/{session_id}`
Deletes a session.

### `WS /ws/chat`
WebSocket chat endpoint for interactive usage.

## Notes

- API mode requires `fastapi`, `uvicorn`, and `pydantic`.
- The server is intended for local or controlled experimental use.
- Authentication and rate limiting are not included by default.