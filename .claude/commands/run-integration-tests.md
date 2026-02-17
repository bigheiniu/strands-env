Run integration tests in an isolated temporary venv created with uv.

Integration tests require a running SGLang server. Before running any commands, ask the user:

1. "What is the SGLang server base URL?" — offer these options:
   - `http://localhost:30000` (default — assumes local or SSH-tunneled server)
   - Let the user provide a custom URL

Then proceed:

1. Create a temporary venv at `/tmp/strands-env-test-venv` using `uv venv /tmp/strands-env-test-venv --python 3.12 -q`
2. Install the package with dev dependencies: `uv pip install -e ".[dev]" --python /tmp/strands-env-test-venv/bin/python -q`
3. If testing a specific environment that has a `requirements.txt` (e.g., `src/strands_env/environments/terminal_bench/requirements.txt`), install those too: `uv pip install -r <path>/requirements.txt --python /tmp/strands-env-test-venv/bin/python -q`
4. Run integration tests with the confirmed URL: `/tmp/strands-env-test-venv/bin/python -m pytest tests/integration/ -v --tb=short --sglang-base-url=<URL> $ARGUMENTS`

IMPORTANT: Never use the active shell's python/pytest — it may point to a different venv. Always use the temporary venv's python.
