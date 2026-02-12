# Terminal-Bench

Terminal-Bench evaluation using `TerminalBenchEnv` with Docker-based task execution.

## Overview

This example provides an environment hook for evaluating agents on [Terminal-Bench](https://github.com/laude-institute/terminal-bench) benchmarks. Each task runs in an isolated Docker container with the agent interacting via `execute_command` tool calls.

**Supported benchmarks:**
- `terminal-bench-1` - [Terminal-Bench 1.0](https://github.com/laude-institute/terminal-bench) 
- `terminal-bench-2` - [Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2)

## Components

**Environment:**
- `TerminalBenchEnv` - Docker-based environment based on [Harbor](https://github.com/laude-institute/harbor), with `execute_command` tool for shell interaction

## Files

- `terminal_bench_env.py` - Environment hook that creates `TerminalBenchEnv` instances

## Prerequisites

1. **Docker**: Ensure Docker is installed and running
2. **Dependencies**: Install additional requirements:
   ```bash
   pip install -r src/strands_env/environments/terminal_bench/requirements.txt
   ```

## Usage

```bash
# Terminal-Bench 1
strands-env eval run terminal-bench-1 \
    --env examples/eval/terminal_bench/terminal_bench_env.py \
    --base-url http://localhost:30000

# Terminal-Bench 2
strands-env eval run terminal-bench-2 \
    --env examples/eval/terminal_bench/terminal_bench_env.py \
    --base-url http://localhost:30000
```

See `strands-env eval run --help` for all CLI options.