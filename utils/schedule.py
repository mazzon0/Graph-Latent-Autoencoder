def warmup_progress(epoch: int, delay_epochs: int = 0, ramp_epochs: int = 0) -> float:
    """
    Progress of the sparsity schedule, from 0.0 (dense graph) to 1.0 (target sparsity).
    It stays at 0 for `delay_epochs` epochs, then grows linearly over `ramp_epochs` epochs.
    Used by both the loss (band bounds) and the model (gating thresholds), so they always agree.
    """
    if epoch < delay_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - delay_epochs) / ramp_epochs)
