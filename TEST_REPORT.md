# DiceRP AI Agent test report

**Run date** November 8, 2026

**Environment** Windows, Python 3.14.7, pytest 9.2.0

## Publication verification

The full offline suite was rerun on Windows with Python 3.14.7 and pytest 9.2.0.
It returned 45 passes and 7 opt-in skips. A separate real Ollama run returned 3 passes
in 8.49 seconds. Ruff lint and format checks, mypy, and the wheel and source builds
passed. The live OpenAI tests remain opt-in and were not rerun for publication.

## Ollama provider verification

**Environment** Windows, Python 3.14.7, pytest 9.2.0, Ollama 0.40.15. The
execution host reported an NVIDIA GeForce RTX 5070 with 12,227 MiB of VRAM.
Ollama's `/api/ps` reported both requested models fully resident in VRAM after
the first live run. GPU placement on an RTX 4070 Super was not measured.

| Command run from the DiceRP directory | Observed result |
| --- | --- |
| `python -m pytest` | 45 passed, 7 opt-in tests skipped |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | 26 files already formatted |
| `python -m mypy src tests streamlit_app.py` | Passed for 23 source files |
| `python -m build` | Wheel and source distribution built |
| `python -m dicerp.cli seed` | Created the local synthetic snapshot |
| `python -m dicerp.cli index` | Indexed the packaged handbook using local Ollama embeddings |
| `python -m streamlit run streamlit_app.py --server.headless true` | Loopback health endpoint returned HTTP 200 |

The offline suite includes the previous 32 tests. The added tests cover provider
selection and defaults, legacy index loading, rejection of cross-provider vectors,
Chat Completions tool replay, embedding response validation, and simulated
connection refusal, missing model, and model-load timeout. These failures are
mapped to sanitized HTTP 502 or 504 responses.

### Real local models

```powershell
$env:DICERP_RUN_OLLAMA_LIVE = "1"
python -m pytest -m ollama_live -v --tb=short
```

The suite passed all **3 real local tests** on both runs (53.06 s and 3.13 s).
The local API was at `http://127.0.0.1:11434/v1`, with `qwen3:8b` for chat and
`qwen3-embedding:0.6b` for both index creation and query embedding.

| Scenario | Observed result |
| --- | --- |
| Mixed handbook, player, and status question | All three tools ran. The answer cited `conduct.txt` line 10 verbatim, returned demo-001's 42 hours, and reported the timestamped 17/64 player snapshot. |
| Unsupported glove compartment rule | Insufficient evidence, with zero citations. |
| Missing player `demo-999` | The SQL tool returned no match and the agent abstained. |

A separate real HTTP smoke check started Uvicorn on a temporary loopback port
with the project `.env`. Both `GET /healthz` and the Metagaming `POST /v1/chat`
returned HTTP 200. The chat reply cited `conduct.txt`, line 10. The live cases
verify the listed prompts, while general retrieval relevance and tool accuracy
remain dependent on the model.

No OpenAI requests were sent during this verification. OpenAI's offline
integration tests and opt-in live suite remain available. The prior live
OpenAI result is recorded below.

## Previous live OpenAI rerun

The live-only pytest selection collected four tests and stopped at the first setup
error. The initial handbook indexing request to `/v1/embeddings` used
`text-embedding-3-small` and doubled as the credit availability check. OpenAI again
returned **HTTP 429** with error code `credit_balance_exhausted`. No further API
requests were sent. OpenAI returned no token usage, and no billing amount was measured.

| Live check | Observed result |
| --- | --- |
| Handbook embedding and index creation | One request rejected because the credit balance was exhausted |
| Real RAG retrieval, tool calling, SQL integration, and exact citations | Blocked by index setup |
| Abstention for a missing demo player | Unrun after the setup error |
| Abstention for a question the handbook does not answer | Unrun after the setup error |
| Live provider error mapped to a sanitized FastAPI 502 | Unrun after the setup error |

The rerun sent **one Embeddings request, zero Responses requests, and zero retries**.
The per-run caps remain nine Responses requests and five Embeddings requests.
The previous live attempt also stopped on `credit_balance_exhausted`.

## Previous offline verification

| Check | Result |
| --- | --- |
| `python -m pytest` | 32 passed, 3 live tests skipped at the time of the offline run |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed |
| `python -m mypy src tests streamlit_app.py` | Passed |
| `python -m build` | Wheel and source distribution built |

Offline coverage includes SDK function-call replay, deterministic embedding ranking,
exact citation line spans, both SQL-backed tools, SQLite write rejection, invalid
tool calls, missing evidence, provider failures, and Streamlit rendering. These
checks use test doubles for OpenAI and cannot establish live model behavior.

## Fixes confirmed during verification

- The workspace-root `.env` appeared as untracked in Git. A root-scoped ignore rule
  now protects it from ordinary staging. Git confirmed the workspace and project
  `.env` paths are ignored.
- The live setup failure initially displayed the upstream exception chain. The runner
  now suppresses that chain and reports allowlisted error codes. A regression test
  reproduces the credit error and checks that the raw provider diagnostic is absent.
