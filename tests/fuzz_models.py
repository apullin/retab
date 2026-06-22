"""Lightweight model fuzzer for the Rust TOSA pipeline.

Generates random quantized models from supported ops and runs them through the
full TOSA INT pipeline (export -> quantize -> partition -> lower -> reference
model comparison). Each model is seeded for reproducibility: a failing seed
can be re-run in isolation to debug.

Usage (standalone, inside Docker with ARM SDK + Vela):
    RETAB_ENABLE_PATCH=1 python tests/fuzz_models.py [num_seeds]

Usage (pytest):
    RETAB_ENABLE_PATCH=1 pytest tests/fuzz_models.py -v
"""

import random
import sys
import time
from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Random model generation
# ---------------------------------------------------------------------------

# Channels choices that stay small enough for fast quantized inference
_CHANNEL_CHOICES = [8, 16, 32, 64]


class RandomModel(nn.Module):
    """A randomly constructed CNN from supported TOSA ops.

    The model always ends with AdaptiveAvgPool2d(1) -> Flatten -> Linear(10)
    so it produces a fixed-size output regardless of the random body.
    """

    def __init__(self, seed: int):
        super().__init__()
        rng = random.Random(seed)

        layers: List[nn.Module] = []
        in_c = 3  # RGB input
        spatial = 32  # assume 32x32 input

        num_layers = rng.randint(2, 6)
        for _ in range(num_layers):
            op = rng.choice(["conv", "conv", "conv_bn", "pool", "activation"])

            if op == "conv" and spatial >= 3:
                out_c = rng.choice(_CHANNEL_CHOICES)
                k = rng.choice([1, 3])
                layers.append(nn.Conv2d(in_c, out_c, k, padding=k // 2))
                layers.append(_random_activation(rng))
                in_c = out_c

            elif op == "conv_bn" and spatial >= 3:
                out_c = rng.choice(_CHANNEL_CHOICES)
                k = rng.choice([1, 3])
                layers.append(nn.Conv2d(in_c, out_c, k, padding=k // 2))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(_random_activation(rng))
                in_c = out_c

            elif op == "pool" and spatial >= 4:
                pool_type = rng.choice(["max", "avg"])
                if pool_type == "max":
                    layers.append(nn.MaxPool2d(2))
                else:
                    layers.append(nn.AvgPool2d(2))
                spatial //= 2

            elif op == "activation":
                layers.append(_random_activation(rng))

        # Fixed tail: collapse spatial dims and classify
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_c, 10))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RandomResidualModel(nn.Module):
    """A randomly constructed CNN with one residual (add) connection."""

    def __init__(self, seed: int):
        super().__init__()
        rng = random.Random(seed)

        mid_c = rng.choice(_CHANNEL_CHOICES)
        k = rng.choice([1, 3])
        self.stem = nn.Sequential(
            nn.Conv2d(3, mid_c, k, padding=k // 2),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
        )

        # Residual block: conv -> bn -> relu -> conv -> bn, then add input
        self.res_conv1 = nn.Conv2d(mid_c, mid_c, 3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(mid_c)
        self.res_conv2 = nn.Conv2d(mid_c, mid_c, 3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(mid_c)

        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_c, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        residual = out
        out = torch.relu(self.res_bn1(self.res_conv1(out)))
        out = self.res_bn2(self.res_conv2(out))
        out = torch.relu(out + residual)
        return self.tail(out)


def _random_activation(rng: random.Random) -> nn.Module:
    return rng.choice([
        nn.ReLU(),
        nn.ReLU6(),
        nn.Sigmoid(),
        nn.Tanh(),
    ])


def _make_model(seed: int) -> nn.Module:
    """Pick a model class based on the seed."""
    rng = random.Random(seed)
    cls = rng.choice([RandomModel, RandomModel, RandomModel, RandomResidualModel])
    return cls(seed)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def run_pipeline(model: nn.Module, test_data: tuple):
    """Run the full TOSA INT quantized pipeline on a model."""
    from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT

    pipeline = TosaPipelineINT[torch.Tensor](
        model,
        test_data,
        aten_op=[],
        exir_op=[],
        # Relax tolerances slightly for random architectures
        atol=1e-02,
        rtol=1e-02,
        qtol=2,
        frobenius_threshold=0.2,
        cosine_threshold=0.85,
    )
    pipeline.run()


def test_fuzz_model_seed(seed: int = 42):
    """Fuzz a single model. Callable as a pytest test or standalone."""
    model = _make_model(seed)
    model.eval()
    test_data = (torch.randn(1, 3, 32, 32),)
    run_pipeline(model, test_data)


# Parametrized pytest entrypoint: runs seeds 0..49
def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        num = int(metafunc.config.getoption("--num-seeds", default=50))
        metafunc.parametrize("seed", range(num))


def test_fuzz(seed: int):
    test_fuzz_model_seed(seed)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    passed = 0
    failed = 0
    failures: List[dict] = []

    print(f"Fuzzing {num_seeds} random models through TOSA INT pipeline\n")

    for seed in range(num_seeds):
        t0 = time.time()
        try:
            test_fuzz_model_seed(seed)
            elapsed = time.time() - t0
            passed += 1
            print(f"  seed={seed:4d}  PASS  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            err_line = str(e).split("\n")[0][:120]
            failures.append({"seed": seed, "error": err_line})
            print(f"  seed={seed:4d}  FAIL  ({elapsed:.1f}s)  {err_line}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {num_seeds}")

    if failures:
        print(f"\nFailing seeds (re-run with: python {__file__} <seed>):")
        for f in failures:
            print(f"  seed={f['seed']:4d}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
