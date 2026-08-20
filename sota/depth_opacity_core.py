"""Pure selection rules for the frozen depth-opacity ceiling."""


SCALES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.375, 0.25)
SELECTION_VIEWS = 32
PSNR_GATE_DB = 0.5


def evenly_spaced_indices(count, maximum):
    if count < 1 or maximum < 1:
        raise ValueError("count and maximum must be positive")
    chosen = min(count, maximum)
    return [index * count // chosen for index in range(chosen)]


def choose_scale(mean_psnr_by_scale):
    if set(mean_psnr_by_scale) != set(SCALES):
        raise ValueError("selection results do not match the locked scale grid")
    return max(SCALES, key=lambda scale: (mean_psnr_by_scale[scale], scale))
