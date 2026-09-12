def resolve_device() -> str:
    try:
        import torch

        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass

        try:
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
        except Exception:
            pass

    except Exception:
        pass

    return "cpu"