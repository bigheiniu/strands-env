#!/usr/bin/env python3
"""AWM Eval Launcher - Submit AgentWorldModel evaluation jobs to Greenland.

Supports two LLM modes:
1. API mode: Use Bedrock models directly (no local serving needed)
2. Local model mode: Download S3/HF model, serve via vLLM/SGLang, run eval against local endpoint

Usage:
    # Bedrock (API mode)
    python launcher.py greenland \
        --job-name awm_sonnet \
        --artifact-path s3://shopqa-users/$USER/artifacts/ \
        --output-path s3://shopqa-users/$USER/awm_results/ \
        --model-id bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        --backend bedrock \
        --awm-data-s3 s3://shopqa-users/lyichuan/data/awm/all100_subsample/ \
        --initiative-id SFAI-shared-p5

    # Local SGLang model
    python launcher.py greenland \
        --job-name awm_local \
        --artifact-path s3://shopqa-users/$USER/artifacts/ \
        --output-path s3://shopqa-users/$USER/awm_results/ \
        --model-id hosted_vllm/local-model \
        --backend sglang \
        --model-path s3://shopqa-users/$USER/models/my-model \
        --serving-framework sglang \
        --tensor-parallel-size 8 \
        --awm-data-s3 s3://shopqa-users/lyichuan/data/awm/sonnet45_default_0325/part_1 \
        --initiative-id SFAI-shared-p5
"""

import argparse
import json
import os
import random
import re
import shutil
import string
import subprocess
import tempfile
from pathlib import Path

import s3fs  # noqa: F401 - ensures s3fs is available in the environment

# --- CONSTANTS ---
DEFAULT_REGION = "us-east-1"
AVAILABLE_INSTANCE_TYPES = [
    "p5.48xlarge",
    "p5en.48xlarge",
    "p4de.24xlarge",
    "p4d.24xlarge",
    "g5.4xlarge",
    "g5.16xlarge",
    "g6e.48xlarge",
]

AWS_ACCOUNT_ID = "684288478426"
ECR_REPO = f"{AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"

# Docker images per serving framework
DOCKER_IMAGES = {
    "vllm": f"{ECR_REPO}/neo-eval:vllm-public-015p1-lm_eval-20260206",
    "sglang": f"{AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/nemo-nile-runner:slime-nightly-dev-20260202c",
}

WORKDIR = "/workspace"

# Greenland roles
SHOPQA_PROD_GREENLAND_ROLE_ARN = "arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole"
SHOPQA_PROD_GREENLAND_ROLE_ARN = "arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole"
M5_PROD_GREENLAND_CROSS_ACCOUNT_ROLE_ARN = "arn:aws:iam::350694149704:role/M5BatchRoleProd"
M5_INITIATIVE_ID_SET = frozenset([
    "ArpM5Collab", "ConvoFM", "Lila", "LilaBase", "LilaE",
    "M5QU", "ModelingProspect", "TextProductization",
])

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
# strands-env project root
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


# =========================================================================
# CLI argument parsing
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AWM Eval Launcher - Submit AgentWorldModel evaluation jobs to Greenland",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("job_type", choices=["greenland"], help="Job type (only greenland supported)")

    # --- Required ---
    parser.add_argument("--job-name", required=True, help="Job name prefix")
    parser.add_argument("--output-path", required=True, help="S3 path for results")
    parser.add_argument("--artifact-path", required=True,
                        help="S3 path to store code snapshot")

    # --- AWM specific ---
    parser.add_argument("--awm-data-s3", required=True,
                        help="S3 path to AWM dataset partition (containing gen_tasks.jsonl, gen_verifier.jsonl, gen_envs.jsonl, databases/)")
    parser.add_argument("--strands-env-s3", default=None,
                        help="S3 path to pre-uploaded strands-env package. If not set, uploads from local project root.")

    # --- Model / backend ---
    parser.add_argument("--backend", choices=["sglang", "bedrock", "kimi"], default="sglang",
                        help="Model backend (default: sglang)")
    parser.add_argument("--model-id", default=None,
                        help="Model ID (e.g., us.anthropic.claude-sonnet-4-5-20250929-v1:0 for bedrock, auto-detected for sglang)")
    parser.add_argument("--base-url", default="http://localhost:30000",
                        help="Base URL for SGLang server (default: http://localhost:30000)")
    parser.add_argument("--aws-region", default="us-west-2",
                        help="AWS region for Bedrock (default: us-west-2)")
    parser.add_argument("--profile-name", default=None,
                        help="AWS profile name for Bedrock (default: None, uses default credentials)")
    parser.add_argument("--tool-parser", default=None,
                        help="Tool parser name (e.g., 'hermes', 'qwen_xml')")

    # --- Sampling params ---
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Maximum new tokens (default: 16384)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k sampling parameter")

    # --- Eval settings ---
    parser.add_argument("--n-samples-per-prompt", type=int, default=3,
                        help="Number of samples per prompt for pass@k (default: 3)")
    parser.add_argument("--max-concurrency", type=int, default=10,
                        help="Maximum concurrent evaluations (default: 10)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to N dataset samples (for testing)")
    parser.add_argument("--max-tool-iters", type=int, default=15,
                        help="Max tool iterations per episode (default: 15)")
    parser.add_argument("--max-tool-calls", type=int, default=30,
                        help="Max tool calls per episode (default: 30)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save results every N samples (default: 10)")

    # --- Local model serving ---
    parser.add_argument("--model-path", default=None,
                        help="Path to model: S3 URI or HuggingFace ID (triggers local serving mode)")
    parser.add_argument("--serving-framework", choices=["vllm", "sglang"], default="sglang",
                        help="Serving framework for local models (default: sglang)")
    parser.add_argument("--tensor-parallel-size", type=int, default=8,
                        help="Tensor parallel size (default: 8)")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="Port for vLLM/SGLang server (default: 8000)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="GPU memory utilization (default: 0.8)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max model context length for vLLM (default: auto)")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Context length for SGLang server (default: model default)")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Max sequences per iteration (default: 256)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"],
                        help="Data type (default: auto)")
    parser.add_argument("--quantization", default=None,
                        choices=["compressed-tensors", "fp8"],
                        help="Quantization method (default: None)")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer path (default: same as model-path)")
    parser.add_argument("--tool-call-parser", default="hermes",
                        help="Tool call parser for vLLM/SGLang server (default: hermes)")
    parser.add_argument("--reasoning-parser", default="deepseek_r1",
                        help="Reasoning parser (default: deepseek_r1)")

    # --- S3 sync ---
    parser.add_argument("--sync-interval", type=int, default=60,
                        help="Seconds between background S3 syncs (default: 60, 0 to disable)")

    # --- Greenland ---
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region for Greenland")
    parser.add_argument("--instance-type", default=None,
                        choices=AVAILABLE_INSTANCE_TYPES,
                        help="Instance type (auto-inferred from initiative if not set)")
    parser.add_argument("--initiative-id", default="SFAI-shared-p5",
                        choices=[
                            "GeneralEvaluationInitiative", "EvaluationP5NodesBOM",
                            "RufusPilotInitiative", "Neo-post-training",
                            "Rufus-shared", "SFAI-shared-p5", "SFAI-shared-p4de",
                            "Rufus-post-training",
                        ] + list(M5_INITIATIVE_ID_SET),
                        help="Greenland initiative ID (default: SFAI-shared-p5)")
    parser.add_argument("--is-production", action="store_true")
    parser.add_argument("--resume-from", default=None,
                        help="S3 path to previous results dir to resume from (copies results.jsonl before starting)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting job")

    args = parser.parse_args()

    # Validation
    if not args.output_path.startswith("s3://"):
        raise ValueError("--output-path must be an S3 URI")
    if not args.artifact_path.startswith("s3://"):
        raise ValueError("--artifact-path must be an S3 URI")
    if not args.awm_data_s3.startswith("s3://"):
        raise ValueError("--awm-data-s3 must be an S3 URI")

    print("========= AWM Eval Options =========")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=====================================")

    return args


# =========================================================================
# Shared helpers
# =========================================================================

def create_job_name(args: argparse.Namespace) -> str:
    suffix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', args.job_name)
    return f"{sanitized}_{suffix}"


def create_output_dir_name(args: argparse.Namespace) -> str:
    """Deterministic output directory name (no random suffix) for checkpoint resume."""
    return re.sub(r'[^a-zA-Z0-9_-]', '-', args.job_name)


def compress_items(source_items, output_filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        for item_path in source_items:
            item_name = os.path.basename(item_path)
            dest_path = os.path.join(temp_dir, item_name)
            if os.path.isdir(item_path):
                shutil.copytree(item_path, dest_path)
            elif os.path.isfile(item_path):
                shutil.copy2(item_path, dest_path)
        shutil.make_archive(output_filename, 'gztar', temp_dir)


PACK_ITEMS = ["src", "examples", "pyproject.toml", "README.md", "LICENSE"]


def pack_code(args: argparse.Namespace, job_name: str) -> str:
    """Pack the strands-env project and upload to S3."""
    code_snapshot_path = os.path.join(args.artifact_path, f"{job_name}.tar.gz")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_snapshot_path = os.path.join(temp_dir, job_name)
        items_to_pack = [
            os.path.join(str(PROJECT_ROOT), item) for item in PACK_ITEMS
            if os.path.exists(os.path.join(str(PROJECT_ROOT), item))
        ]
        compress_items(items_to_pack, temp_snapshot_path)
        local_tar = f'{temp_snapshot_path}.tar.gz'
        subprocess.run(
            ["aws", "s3", "cp", local_tar, code_snapshot_path, "--region", "us-east-1"],
            check=True,
        )
        print(f"Code snapshot uploaded to S3: {code_snapshot_path}")
    return code_snapshot_path


def _build_model_download_cmd(model_path: str, local_path: str) -> str:
    if not model_path.startswith("s3://"):
        return (
            f"curl -LsSf https://hf.co/cli/install.sh | bash && "
            f"hf download {model_path} --local-dir {local_path}"
        )
    return f"aws s3 cp {model_path} {local_path} --recursive --region us-west-2"


def _build_vllm_server_cmd(args: argparse.Namespace, model_local_path: str) -> str:
    tokenizer = args.tokenizer or model_local_path
    parts = [
        "python3 -m vllm.entrypoints.openai.api_server",
        f"--port {args.vllm_port}",
        f"--model {model_local_path}",
        f"--tokenizer {tokenizer}",
        "--served-model-name local-model",
        f"--tensor-parallel-size {args.tensor_parallel_size}",
        f"--gpu-memory-utilization {args.gpu_memory_utilization}",
        f"--max-num-seqs {args.max_num_seqs}",
        f"--dtype {args.dtype}",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
    ]
    if args.max_model_len:
        parts.append(f"--max-model-len {args.max_model_len}")
    if args.quantization:
        parts.append(f"--quantization {args.quantization}")
    return f"({' '.join(parts)} &)"


def _build_sglang_server_cmd(args: argparse.Namespace, model_local_path: str) -> str:
    return (
        f"(python3 -m sglang.launch_server "
        f"--model-path {model_local_path} "
        f"--host 0.0.0.0 "
        f"--tp {args.tensor_parallel_size} "
        f"--port {args.vllm_port} "
        f"--served-model-name local-model "
        f"--mem-fraction-static {args.gpu_memory_utilization} "
        f"--max-running-requests {args.max_num_seqs} "
        f"--dtype {args.dtype} "
        f"--trust-remote-code "
        f"{'--context-length ' + str(args.context_length) + ' ' if args.context_length else ''}"
        f"> {WORKDIR}/sglang_server.log 2>&1 &)"
    )


def _build_wait_for_server_cmd(port: int) -> str:
    return (
        f'echo "Waiting for model server to start..." && '
        f"_SERVER_READY=0 && "
        f"for i in $(seq 1 90); do "
        f"if curl -s http://localhost:{port}/health > /dev/null 2>&1 || "
        f"curl -s http://localhost:{port}/v1/models > /dev/null 2>&1; then "
        f'echo "Server is ready!"; _SERVER_READY=1; break; fi; '
        f'echo "Waiting... ($i/90)"; sleep 10; done && '
        f'if [ "$_SERVER_READY" -eq 0 ]; then '
        f'echo "ERROR: Server failed to start. Last 100 lines of log:" && '
        f"tail -100 {WORKDIR}/sglang_server.log 2>/dev/null && "
        f"exit 1; fi"
    )


def _build_local_serving_preamble(args: argparse.Namespace) -> list[str]:
    """Build commands to download model, start server, wait for readiness."""
    model_local_path = f"{WORKDIR}/model"
    cmds = [
        _build_model_download_cmd(args.model_path, model_local_path),
        "export OPENAI_API_KEY=dummy",
        f"export OPENAI_API_BASE=http://localhost:{args.vllm_port}/v1",
    ]
    if args.serving_framework == "vllm":
        cmds.append(_build_vllm_server_cmd(args, model_local_path))
    else:
        cmds.append(_build_sglang_server_cmd(args, model_local_path))
    cmds.append(_build_wait_for_server_cmd(args.vllm_port))
    return cmds


def build_s3_sync_commands(run_cmd: str, results_dir: str, output_path: str, sync_interval: int) -> str:
    results_dirname = os.path.basename(os.path.normpath(results_dir))
    s3_dest = f"{output_path.rstrip('/')}/{results_dirname}/"
    # Copy server log into results dir so it gets synced to S3
    copy_logs = (
        f"cp {WORKDIR}/sglang_server.log {results_dir} 2>/dev/null ; "
        f"cp {WORKDIR}/eval.log {results_dir} 2>/dev/null"
    )
    s3_cp = f"{copy_logs} ; aws s3 cp {results_dir} {s3_dest} --recursive --region us-east-1"
    # Tee eval stdout/stderr to a log file for debugging
    logged_run_cmd = f"{run_cmd} 2>&1 | tee {WORKDIR}/eval.log"
    if sync_interval <= 0:
        return f"{logged_run_cmd} ; {s3_cp}"
    return (
        f"(while true; do sleep {sync_interval}; {s3_cp}; done & _SYNC_PID=$! ; "
        f"{logged_run_cmd} ; "
        f"kill $_SYNC_PID 2>/dev/null ; "
        f"{s3_cp})"
    )


# =========================================================================
# AWM eval command builder
# =========================================================================

def build_awm_eval_cmd(args: argparse.Namespace, output_dir_name: str) -> str:
    """Build the strands-env eval run command for AWM benchmark."""
    awm_workdir = WORKDIR
    awm_data_dir = f"{WORKDIR}/awm_data"
    results_dir = f"{awm_workdir}/{output_dir_name}/"
    cmd_parts = []

    # 1. Download AWM dataset from S3
    cmd_parts.append(
        f"aws s3 cp {args.awm_data_s3} {awm_data_dir} --recursive --region us-east-1"
        f' && echo "AWM data downloaded to {awm_data_dir}"'
    )

    # 2. Restore previous checkpoint from S3 (for failure recovery)
    # The evaluator auto-resumes from results.jsonl if it exists.
    results_dirname = os.path.basename(os.path.normpath(results_dir))
    s3_results = f"{args.output_path.rstrip('/')}/{results_dirname}/"
    resume_source = args.resume_from.rstrip("/") + "/" if args.resume_from else s3_results
    cmd_parts.append(
        f"mkdir -p {results_dir}"
        f" && aws s3 cp {resume_source} {results_dir} --recursive --region us-east-1 2>/dev/null"
        f' && echo "Restored checkpoint from {resume_source} ($(wc -l < {results_dir}results.jsonl 2>/dev/null || echo 0) samples)"'
        f' || echo "No previous checkpoint found, starting fresh"'
    )

    # 3. Install strands-env
    # Set pretend version since .git is not included in the code snapshot.
    # Both generic and package-specific vars for maximum compatibility.
    scm_export = (
        "export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_STRANDS_ENV=0.0.0"
    )
    if args.strands_env_s3:
        cmd_parts.append(
            f"aws s3 cp {args.strands_env_s3} {awm_workdir} --recursive --region us-east-1"
            f" && cd {awm_workdir} && {scm_export} && pip install -e '.[dev]'"
        )
    else:
        # strands-env is packed in the code snapshot (project root)
        cmd_parts.append(
            f"cd {awm_workdir} && {scm_export} && pip install -e '.[dev]'"
        )

    # 4. Install AWM-specific dependencies
    cmd_parts.append(
        f"pip install -r {awm_workdir}/src/strands_env/environments/agentworldmodel/requirements.txt"
    )

    # 5. Local model serving (if applicable)
    if args.model_path:
        cmd_parts.extend(_build_local_serving_preamble(args))

    # 6. Build the strands-env eval run command
    eval_parts = [
        f"cd {awm_workdir}",
        f"AWM_DATA_DIR={awm_data_dir}",
        "strands-env eval run",
        "--evaluator examples.eval.agentworldmodel.awm_evaluator",
        "--env examples.eval.agentworldmodel.awm_env",
        f"--backend {args.backend}",
        f"--max-tokens {args.max_tokens}",
        f"--n-samples-per-prompt {args.n_samples_per_prompt}",
        f"--max-concurrency {args.max_concurrency}",
        f"--save-interval {args.save_interval}",
        f"-o {results_dir}",
    ]

    # Model-specific options
    if args.backend == "sglang":
        if args.model_path:
            eval_parts.append(f"--base-url http://localhost:{args.vllm_port}")
            # Use local model path as tokenizer source (model_id may not be a valid HF name)
            model_local_path = f"{WORKDIR}/model"
            eval_parts.append(f"--tokenizer-path {model_local_path}")
        else:
            eval_parts.append(f"--base-url {args.base_url}")
        if args.tool_parser:
            eval_parts.append(f"--tool-parser {args.tool_parser}")
    elif args.backend == "bedrock":
        if args.model_id:
            eval_parts.append(f"--model-id {args.model_id}")
        eval_parts.append(f"--region {args.aws_region}")
        if args.profile_name:
            eval_parts.append(f"--profile-name {args.profile_name}")

    if args.model_id and args.backend != "bedrock":
        eval_parts.append(f"--model-id {args.model_id}")

    # Sampling params
    if args.temperature is not None:
        eval_parts.append(f"--temperature {args.temperature}")
    if args.top_p is not None:
        eval_parts.append(f"--top-p {args.top_p}")
    if args.top_k is not None:
        eval_parts.append(f"--top-k {args.top_k}")

    # Env config
    env_config = {"max_tool_iters": args.max_tool_iters, "max_tool_calls": args.max_tool_calls}
    eval_parts.append(f"--env-config '{json.dumps(env_config)}'")

    if args.max_samples:
        eval_parts.append(f"--max-samples {args.max_samples}")

    # Combine: "cd ... && AWM_DATA_DIR=... strands-env eval run ..."
    cd_part = eval_parts[0]
    env_var = eval_parts[1]
    rest = " \\\n    ".join(eval_parts[2:])
    run_cmd = f"{cd_part} && {env_var} {rest}"

    sync_and_run = build_s3_sync_commands(
        run_cmd=run_cmd,
        results_dir=results_dir,
        output_path=args.output_path,
        sync_interval=args.sync_interval,
    )
    cmd_parts.append(sync_and_run)
    return " && ".join(cmd_parts)


# =========================================================================
# Greenland submission
# =========================================================================

def try_get_greenland_instance_type_and_region(initiative: str, instance_type=None):
    mapping = {
        "EvaluationP5NodesBOM": ("p5.48xlarge", "ap-south-1"),
        "GeneralEvaluationInitiative": ("p4d.24xlarge", "us-east-2"),
        "Neo-post-training": ("p5.48xlarge", "ap-south-1"),
        "SFAI-shared-p5": ("p5.48xlarge", "us-west-2"),
        "SFAI-shared-p4de": ("p4de.24xlarge", "us-east-1"),
        "Rufus-post-training": ("p5en.48xlarge", "us-west-2"),
    }
    if initiative in mapping:
        default_instance, region = mapping[initiative]
        return instance_type or default_instance, region
    if instance_type is not None:
        if instance_type.startswith("p5"):
            return instance_type, "ap-south-1"
        elif instance_type.startswith("p4"):
            return instance_type, "us-east-2"
        return instance_type, "us-east-1"
    raise ValueError(
        f"Cannot infer instance type for initiative '{initiative}'. Set --instance-type explicitly."
    )


def launch_greenland_job(
    user_cmd: str,
    code_snapshot_path: str,
    job_name: str,
    initiative_id: str,
    region: str,
    instance_type: str,
    is_production: bool = False,
    docker_image_tag: str = None,
    dry_run: bool = False,
):
    env = os.environ.copy()
    env["ECR_DOCKER_IMAGE"] = docker_image_tag or DOCKER_IMAGES["sglang"]
    env["CODE_SNAPSHOT"] = code_snapshot_path
    env["RUNCMD"] = user_cmd
    env["JOB_NAME"] = job_name
    env["WORKDIR"] = WORKDIR
    env["INITIATIVE_ID"] = initiative_id
    env["IS_PRODUCTION"] = str(is_production).lower()

    instance_type, region = try_get_greenland_instance_type_and_region(
        initiative_id, instance_type
    )
    env["INSTANCE_TYPE"] = instance_type
    env["GREENLAND_REGION"] = region


    env["ROLE"] = SHOPQA_PROD_GREENLAND_ROLE_ARN

    print("=" * 60)
    print("AWM Eval - Submitting to Greenland")
    print("=" * 60)
    print(f"  Job name:       {job_name}")
    print(f"  Code snapshot:  {code_snapshot_path}")
    print(f"  Docker image:   {env['ECR_DOCKER_IMAGE']}")
    print(f"  Instance type:  {instance_type}")
    print(f"  Region:         {region}")
    print(f"  Initiative:     {initiative_id}")
    print("-" * 60)
    print("User commands:")
    print(user_cmd)
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Skipping job submission")
        return

    greenland_script = Path("/Volumes/workplace/TauGL/scripts/aws_greenland_common.sh")
    if not greenland_script.exists():
        # Fallback: check if it's bundled alongside this script
        alt = SCRIPT_DIR / "aws_greenland_common.sh"
        if alt.exists():
            greenland_script = alt
        else:
            raise FileNotFoundError(
                f"Greenland submission script not found at {greenland_script} or {alt}. "
                f"Copy aws_greenland_common.sh next to this launcher or update the path."
            )
    subprocess.run(["/bin/bash", str(greenland_script)], env=env)


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    job_name = create_job_name(args)
    output_dir_name = create_output_dir_name(args)

    # Build eval command
    eval_cmd = build_awm_eval_cmd(args, output_dir_name)

    # Pack and upload code
    code_snapshot_path = pack_code(args, job_name)

    # Select Docker image
    if args.model_path:
        docker_image = DOCKER_IMAGES.get(args.serving_framework)
    else:
        # API mode: use sglang image as base (has Python + pip)
        docker_image = None

    # Launch
    launch_greenland_job(
        user_cmd=eval_cmd,
        code_snapshot_path=code_snapshot_path,
        job_name=job_name,
        initiative_id=args.initiative_id,
        region=args.region,
        instance_type=args.instance_type,
        is_production=args.is_production,
        docker_image_tag=docker_image,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
