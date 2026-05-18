# Security Policy

## Supported Versions

This project is currently maintained as a research prototype. Security fixes, if any, are applied to the latest version of the repository on the default branch.

| Version | Supported |
| ------- | --------- |
| 1.x     | Yes       |

---

## Reporting a Vulnerability

If you discover a security issue, please report it responsibly.

Please include:

- a clear description of the issue
- steps to reproduce
- affected files or components
- impact assessment
- suggested mitigation if available

If the issue involves secrets, API keys, prompt injection risks, local file exposure, or unsafe API behavior, please avoid posting full exploit details publicly in an issue before maintainers have time to respond.

---

## Security Notes for This Repository

Because NOESIS PRIME may interact with:

- local model files
- cloud LLM APIs
- persistent memory storage
- local filesystem paths
- FastAPI endpoints and WebSocket interfaces

please use caution when deploying it in shared or internet-facing environments.

### Recommendations

- do not commit API keys
- use environment variables for secrets
- do not expose the API publicly without authentication and rate limiting
- review CORS configuration before deployment
- isolate local model files and persistent storage
- avoid running experimental agent systems with excessive permissions

---

## Scope Disclaimer

This repository is currently a research and experimentation project, not a hardened production service. Users are responsible for reviewing, testing, and securing deployments in their own environments.
