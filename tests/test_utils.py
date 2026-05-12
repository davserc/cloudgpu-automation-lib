import pytest

from services.vast.gpu_manager.utils import extract_gpu_model_number


@pytest.mark.parametrize(
    "name,expected",
    [
        ("RTX 3090", 3090),
        ("GTX 1080 Ti", 1080),
        ("A100", 100),
        ("H100", 100),
        ("Tesla V100", 100),
        ("RTX 4090", 4090),
        ("RTX_3090", 3090),
        ("no numbers here", None),
        ("", None),
    ],
)
def test_extract_gpu_model_number(name, expected):
    assert extract_gpu_model_number(name) == expected
