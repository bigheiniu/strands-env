"""Shared fixtures for integration tests.

All tests in this directory require a running SGLang server.

Configuration (priority: CLI > env var > default):
    pytest --sglang-base-url=http://localhost:30000 --sglang-model-id=Qwen/Qwen3-4B-Instruct-2507
    SGLANG_BASE_URL=http://... SGLANG_MODEL_ID=... pytest tests/integration/
"""

import pytest
from strands_sglang import SGLangClient
from transformers import AutoTokenizer

from strands_env.core.models import DEFAULT_SAMPLING_PARAMS, sglang_model_factory

# Mark all tests in this directory as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def sglang_base_url(request):
    """Get SGLang server URL from pytest config."""
    return request.config.getoption("--sglang-base-url")


@pytest.fixture(scope="session")
def sglang_model_id(request):
    """Get model ID from pytest config."""
    return request.config.getoption("--sglang-model-id")


@pytest.fixture(scope="session")
def tokenizer(sglang_model_id):
    """Load tokenizer for the configured model."""
    return AutoTokenizer.from_pretrained(sglang_model_id)


@pytest.fixture(scope="session")
def sglang_client(sglang_base_url):
    """Shared SGLang client for connection pooling."""
    return SGLangClient(sglang_base_url)


@pytest.fixture
def model_factory(tokenizer, sglang_client, sglang_model_id):
    """Model factory for Environment integration tests."""
    return sglang_model_factory(
        model_id=sglang_model_id,
        tokenizer=tokenizer,
        client=sglang_client,
        sampling_params=DEFAULT_SAMPLING_PARAMS,
    )
