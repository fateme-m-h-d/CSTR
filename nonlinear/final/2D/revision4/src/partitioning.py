import numpy as np


def geometric_edges_1d(
    xmin,
    xmax,
    n_regions,
    ratio,
    shrink_toward="upper",
):
    """
    Build geometrically shrinking segment lengths.

    Example:
        ratio = 0.95
        relative lengths = 1, 0.95, 0.95^2, ...

    The lengths are normalized so that the segments exactly span
    [xmin, xmax].

    shrink_toward="upper":
        segments become smaller as x increases.

    shrink_toward="lower":
        segments become smaller as x decreases.
    """

    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")

    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must satisfy 0 < ratio <= 1")

    if shrink_toward not in {"upper", "lower"}:
        raise ValueError(
            "shrink_toward must be 'upper' or 'lower'"
        )

    weights = ratio ** np.arange(
        n_regions,
        dtype=float,
    )

    if shrink_toward == "lower":
        weights = weights[::-1]

    lengths = (
        (xmax - xmin)
        * weights
        / weights.sum()
    )

    edges = np.concatenate(
        (
            [xmin],
            xmin + np.cumsum(lengths),
        )
    )

    # Avoid tiny roundoff at final endpoint
    edges[-1] = xmax

    return edges


def geometric_edges_nd(
    bounds,
    n_regions,
    ratios,
    shrink_toward=None,
):
    """
    Dimension-general geometric partition.

    Parameters
    ----------
    bounds:
        [(xmin_0, xmax_0),
         (xmin_1, xmax_1),
         ...]

    n_regions:
        [n0, n1, ...]

    ratios:
        [r0, r1, ...]

    shrink_toward:
        ["upper", "lower", ...]

    One ratio is used per input dimension.
    """

    d = len(bounds)

    if (
        len(n_regions) != d
        or len(ratios) != d
    ):
        raise ValueError(
            "bounds, n_regions, and ratios "
            "must have the same length"
        )

    if shrink_toward is None:
        shrink_toward = ["upper"] * d

    if len(shrink_toward) != d:
        raise ValueError(
            "shrink_toward must contain "
            "one value per dimension"
        )

    edges = []

    for j in range(d):
        edges_j = geometric_edges_1d(
            xmin=bounds[j][0],
            xmax=bounds[j][1],
            n_regions=n_regions[j],
            ratio=ratios[j],
            shrink_toward=shrink_toward[j],
        )

        edges.append(edges_j)

    return edges