"""Smoke test for src/model.py. Run from the repo root:   python -m src.check_model

Checks, for each model size: it builds, the output has the same shape as the input (also for image sizes
that are not divisible by 4), gradients reach every parameter, and prints the parameter count.
If a GPU is usable it also prints the peak GPU memory of one training-like step.
"""
import torch

from src.model import CONFIGS, build_model, count_parameters
from src.utils import set_seed


def main():
    set_seed()
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    for name in CONFIGS:
        model = build_model(name)
        for shape in [(1, 3, 64, 64), (1, 3, 70, 90)]:  # second one is NOT divisible by 4
            x = torch.rand(*shape)
            y = model(x)
            assert y.shape == x.shape, f"{name}: output {tuple(y.shape)} != input {tuple(x.shape)}"
        y.mean().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"{name}: no gradient for {missing[:3]}"
        n = count_parameters(model)
        print(f"{name:>5}: OK   parameters = {n:,} ({n / 1e6:.3f} M)")

    if torch.cuda.is_available():
        for name in CONFIGS:
            model = x = None
            try:
                model = build_model(name).cuda()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.rand(4, 3, 128, 128, device="cuda")  # the default training configuration
                model(x).mean().backward()
                print(f"{name:>5}: GPU peak memory, batch 4 x 128x128 forward+backward: "
                      f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
            except torch.OutOfMemoryError:
                print(f"{name:>5}: OUT OF MEMORY at batch 4 x 128x128 -> use a smaller batch or patch size")
            finally:
                del model, x
                torch.cuda.empty_cache()
    else:
        print("GPU not available to PyTorch: the GPU memory check was skipped.")


if __name__ == "__main__":
    main()
