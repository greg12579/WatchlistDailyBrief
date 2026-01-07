"""Ranking and scoring for triggered items."""

from watchbrief.features.triggers import TriggerResult


def compute_score(result: TriggerResult) -> float:
    """Compute ranking score for a trigger result.

    Formula (from BuildPlan):
    score = 0
        + abs(price_z) * 2
        + max(0, volume_multiple - 1) * 1
        + abs(rel_vs_spy_z) * 1.5
        + abs(rel_vs_sector_z) * 1.5 (if exists)

    Higher scores indicate more significant moves.
    """
    score = 0.0

    # Price z-score contribution (weight: 2)
    score += abs(result.price_z) * 2.0

    # Volume multiple contribution (weight: 1, only excess above 1x)
    score += max(0, result.volume_multiple - 1) * 1.0

    # Relative vs SPY contribution (weight: 1.5)
    score += abs(result.rel_vs_spy_z) * 1.5

    # Relative vs sector contribution (weight: 1.5, if available)
    if result.rel_vs_sector_z is not None:
        score += abs(result.rel_vs_sector_z) * 1.5

    return score


def rank_and_select(
    results: list[TriggerResult],
    max_items: int = 5,
) -> list[tuple[TriggerResult, float, int]]:
    """Rank triggered results and select top items.

    Args:
        results: List of TriggerResult objects (should be pre-filtered to triggered only)
        max_items: Maximum number of items to return

    Returns:
        List of tuples: (TriggerResult, score, rank)
        Sorted by score descending, limited to max_items.
    """
    # Filter to only triggered items
    triggered = [r for r in results if r.triggered]

    # Compute scores
    scored = [(r, compute_score(r)) for r in triggered]

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top items and add rank
    top_items = scored[:max_items]
    ranked = [(r, score, rank + 1) for rank, (r, score) in enumerate(top_items)]

    return ranked
