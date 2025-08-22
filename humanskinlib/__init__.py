import torch


def get_current_gpu():
    # We use map_location to ensure it works on CPU if CUDA is not available
    # --- Recommended Replacement ---
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device