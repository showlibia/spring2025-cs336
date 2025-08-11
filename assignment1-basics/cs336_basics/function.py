import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    v_max = torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x - v_max) / torch.sum(torch.exp(x - v_max), dim=dim, keepdim=True)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
        inputs: (batch_size, vocab_size)
        targets: (batch_size)
    """
    log_probs = torch.log_softmax(inputs, dim=-1)
    return -torch.sum(log_probs[torch.arange(inputs.size(0)), targets]) / inputs.size(0)
