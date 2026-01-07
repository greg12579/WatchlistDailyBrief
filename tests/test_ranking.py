"""Tests for ranking functionality."""

import pytest
from watchbrief.features.ranking import (
    compute_score,
    compute_base_score,
    compute_enhanced_score,
    compute_evidence_boost,
    compute_drift_penalty,
    rank_and_select,
    rank_and_select_enhanced,
    ScoringConfig,
    EnhancedScore,
)
from watchbrief.features.triggers import TriggerResult
from watchbrief.features.attribution import (
    AttributionContext,
    AttributionCategory,
    AttributionHint,
    CatalystChecks,
    CatalystCheckStatus,
    CoverageTier,
)
from watchbrief.llm.explain import Explanation, Driver


def create_trigger_result(
    ticker: str,
    price_z: float = 0.0,
    volume_multiple: float = 1.0,
    rel_vs_spy_z: float = 0.0,
    rel_vs_sector_z: float = None,
    triggered: bool = True,
) -> TriggerResult:
    """Create a TriggerResult for testing."""
    return TriggerResult(
        ticker=ticker,
        last_close=100.0,
        pct_change_1d=price_z * 0.5,  # Approximate
        pct_change_5d=0.0,
        volume_last=1_000_000 * volume_multiple,
        volume_avg=1_000_000,
        volume_multiple=volume_multiple,
        price_z=price_z,
        rel_vs_spy_z=rel_vs_spy_z,
        rel_vs_sector_z=rel_vs_sector_z,
        triggered=triggered,
        triggered_reasons=["test"],
        label="MONITOR",
    )


class TestComputeScore:
    def test_score_formula(self):
        """Test the scoring formula matches spec."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=2.0,      # contributes 4.0 (2.0 * 2)
            volume_multiple=2.0,  # contributes 1.0 ((2.0-1) * 1)
            rel_vs_spy_z=1.0,     # contributes 1.5 (1.0 * 1.5)
            rel_vs_sector_z=1.0,  # contributes 1.5 (1.0 * 1.5)
        )

        score = compute_score(result)
        expected = 4.0 + 1.0 + 1.5 + 1.5
        assert abs(score - expected) < 0.01

    def test_score_without_sector(self):
        """Test scoring without sector data."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=2.0,
            volume_multiple=1.5,
            rel_vs_spy_z=1.0,
            rel_vs_sector_z=None,
        )

        score = compute_score(result)
        expected = 4.0 + 0.5 + 1.5  # No sector contribution
        assert abs(score - expected) < 0.01

    def test_negative_price_z_uses_absolute(self):
        """Test that negative z-scores use absolute value."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=-2.0,
            volume_multiple=1.0,
            rel_vs_spy_z=-1.0,
        )

        score = compute_score(result)
        expected = 4.0 + 0.0 + 1.5
        assert abs(score - expected) < 0.01

    def test_volume_below_1_contributes_zero(self):
        """Test that volume < 1x contributes nothing."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=1.0,
            volume_multiple=0.5,
            rel_vs_spy_z=0.0,
        )

        score = compute_score(result)
        expected = 2.0 + 0.0 + 0.0
        assert abs(score - expected) < 0.01


class TestRankAndSelect:
    def test_ranks_by_score_descending(self):
        """Test that results are ranked by score descending."""
        results = [
            create_trigger_result("LOW", price_z=1.0),
            create_trigger_result("HIGH", price_z=3.0),
            create_trigger_result("MED", price_z=2.0),
        ]

        ranked = rank_and_select(results, max_items=5)

        assert len(ranked) == 3
        assert ranked[0][0].ticker == "HIGH"
        assert ranked[1][0].ticker == "MED"
        assert ranked[2][0].ticker == "LOW"

    def test_limits_to_max_items(self):
        """Test that selection is limited to max_items."""
        results = [
            create_trigger_result(f"TICK{i}", price_z=float(i))
            for i in range(10)
        ]

        ranked = rank_and_select(results, max_items=3)

        assert len(ranked) == 3

    def test_filters_untriggered(self):
        """Test that untriggered items are filtered out."""
        results = [
            create_trigger_result("TRIG", price_z=2.0, triggered=True),
            create_trigger_result("NOTRIG", price_z=3.0, triggered=False),
        ]

        ranked = rank_and_select(results, max_items=5)

        assert len(ranked) == 1
        assert ranked[0][0].ticker == "TRIG"

    def test_includes_rank_number(self):
        """Test that rank numbers are correct."""
        results = [
            create_trigger_result("A", price_z=3.0),
            create_trigger_result("B", price_z=2.0),
            create_trigger_result("C", price_z=1.0),
        ]

        ranked = rank_and_select(results, max_items=5)

        assert ranked[0][2] == 1  # First item has rank 1
        assert ranked[1][2] == 2
        assert ranked[2][2] == 3

    def test_deterministic_ranking(self):
        """Test that ranking is deterministic."""
        results = [
            create_trigger_result("A", price_z=2.0),
            create_trigger_result("B", price_z=2.0),  # Same score
        ]

        ranked1 = rank_and_select(results, max_items=5)
        ranked2 = rank_and_select(results, max_items=5)

        # Order should be consistent
        assert ranked1[0][0].ticker == ranked2[0][0].ticker
        assert ranked1[1][0].ticker == ranked2[1][0].ticker


def create_attribution_context(
    ticker: str,
    result: TriggerResult,
    has_company_catalyst: bool = False,
    has_sector_peer: bool = False,
    news_checked: bool = True,
    sec_checked: bool = True,
) -> AttributionContext:
    """Create an AttributionContext for testing."""
    hints = []

    if has_company_catalyst:
        hints.append(AttributionHint(
            category=AttributionCategory.COMPANY,
            evidence=["Earnings release", "8-K filed"],
            strength=0.9,
        ))

    if has_sector_peer:
        hints.append(AttributionHint(
            category=AttributionCategory.SECTOR_PEER,
            evidence=["XLF +2.1%", "JPM +1.8%"],
            strength=0.6,
        ))

    if not hints:
        hints.append(AttributionHint(
            category=AttributionCategory.UNATTRIBUTED,
            evidence=["No clear catalyst"],
            strength=0.1,
        ))

    return AttributionContext(
        ticker=ticker,
        trigger_result=result,
        attribution_hints=hints,
        catalyst_checks=CatalystChecks(
            earnings_calendar=CatalystCheckStatus.CHECKED,
            news_feed=CatalystCheckStatus.CHECKED if news_checked else CatalystCheckStatus.NOT_AVAILABLE,
            sec_filings=CatalystCheckStatus.CHECKED if sec_checked else CatalystCheckStatus.NOT_AVAILABLE,
        ),
    )


def create_explanation(confidence: str = "Medium") -> Explanation:
    """Create an Explanation for testing."""
    return Explanation(
        drivers=[Driver(rank=1, text="Test driver", weight_pct=100)],
        confidence=confidence,
        why_it_matters="Test explanation",
    )


class TestEvidenceBoost:
    """Tests for evidence-based score boosts."""

    def test_company_catalyst_boost(self):
        """Test that company catalysts add boost."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result, has_company_catalyst=True)
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, None, config)

        assert boost == config.company_catalyst_boost
        assert any("company catalyst" in r for r in reasons)

    def test_sector_peer_corroboration_boost(self):
        """Test that sector/peer correlation adds boost."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result, has_sector_peer=True)
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, None, config)

        assert boost == config.sector_peer_corroboration_boost
        assert any("sector/peer" in r for r in reasons)

    def test_high_confidence_boost(self):
        """Test that high confidence adds boost."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result)
        explanation = create_explanation(confidence="High")
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, explanation, config)

        assert boost == config.high_confidence_boost
        assert any("high confidence" in r for r in reasons)

    def test_combined_boosts(self):
        """Test that multiple boosts stack."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result, has_company_catalyst=True, has_sector_peer=True)
        explanation = create_explanation(confidence="High")
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, explanation, config)

        expected = config.company_catalyst_boost + config.sector_peer_corroboration_boost + config.high_confidence_boost
        assert boost == expected
        assert len(reasons) == 3

    def test_no_boost_without_evidence(self):
        """Test that no boost when no evidence."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result)
        explanation = create_explanation(confidence="Low")
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, explanation, config)

        assert boost == 0.0
        assert len(reasons) == 0


class TestDriftPenalty:
    """Tests for low-volume drift penalty."""

    def test_no_penalty_with_normal_volume(self):
        """Test no penalty when volume is normal."""
        result = create_trigger_result("TEST", price_z=2.0, volume_multiple=1.2)
        context = create_attribution_context("TEST", result)
        config = ScoringConfig()

        penalty, reasons = compute_drift_penalty(result, context, config)

        assert penalty == 0.0
        assert len(reasons) == 0

    def test_penalty_with_low_volume_no_catalyst(self):
        """Test penalty when low volume and no catalyst."""
        result = create_trigger_result("TEST", price_z=2.0, volume_multiple=0.5)
        context = create_attribution_context("TEST", result, has_company_catalyst=False)
        config = ScoringConfig()

        penalty, reasons = compute_drift_penalty(result, context, config)

        assert penalty == config.drift_penalty  # Should be negative
        assert any("drift" in r for r in reasons)

    def test_no_penalty_with_low_volume_and_catalyst(self):
        """Test no penalty when low volume but has catalyst."""
        result = create_trigger_result("TEST", price_z=2.0, volume_multiple=0.5)
        context = create_attribution_context("TEST", result, has_company_catalyst=True)
        config = ScoringConfig()

        penalty, reasons = compute_drift_penalty(result, context, config)

        assert penalty == 0.0

    def test_no_penalty_if_checks_not_done(self):
        """Test no penalty if we didn't check for catalysts."""
        result = create_trigger_result("TEST", price_z=2.0, volume_multiple=0.5)
        context = create_attribution_context("TEST", result, news_checked=False, sec_checked=False)
        config = ScoringConfig()

        penalty, reasons = compute_drift_penalty(result, context, config)

        assert penalty == 0.0


class TestEnhancedScore:
    """Tests for complete enhanced scoring."""

    def test_enhanced_score_includes_base(self):
        """Test that enhanced score includes base score."""
        result = create_trigger_result("TEST", price_z=2.0, volume_multiple=2.0, rel_vs_spy_z=1.0)
        config = ScoringConfig()

        enhanced = compute_enhanced_score(result, config=config, skip_cooldown=True)

        assert enhanced.base_score == compute_base_score(result)
        assert enhanced.final_score >= enhanced.base_score - 1.0  # May have slight penalty

    def test_enhanced_score_with_context(self):
        """Test that enhanced score uses context for adjustments."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result, has_company_catalyst=True)
        config = ScoringConfig()

        enhanced = compute_enhanced_score(result, context=context, config=config, skip_cooldown=True)

        assert enhanced.evidence_boost > 0
        assert enhanced.final_score > enhanced.base_score

    def test_enhanced_score_adjustment_reasons(self):
        """Test that adjustment reasons are tracked."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result, has_company_catalyst=True, has_sector_peer=True)
        config = ScoringConfig()

        enhanced = compute_enhanced_score(result, context=context, config=config, skip_cooldown=True)

        assert len(enhanced.adjustment_reasons) >= 2


class TestRankAndSelectEnhanced:
    """Tests for enhanced ranking with evidence."""

    def test_evidence_affects_ranking(self):
        """Test that evidence boosts can reorder rankings."""
        # Item A: Higher base score, no catalyst
        result_a = create_trigger_result("A", price_z=2.5, volume_multiple=1.0)
        context_a = create_attribution_context("A", result_a, has_company_catalyst=False)

        # Item B: Lower base score, but has catalyst
        result_b = create_trigger_result("B", price_z=2.0, volume_multiple=1.0)
        context_b = create_attribution_context("B", result_b, has_company_catalyst=True)

        items = [
            (result_a, context_a, None),
            (result_b, context_b, None),
        ]

        # With enhanced scoring, B might rank higher due to catalyst boost
        ranked = rank_and_select_enhanced(items, max_items=5)

        # A has base: 5.0 (2.5*2), B has base: 4.0 (2.0*2) + 1.5 catalyst = 5.5
        # So B should rank higher
        assert ranked[0].result.ticker == "B"
        assert ranked[1].result.ticker == "A"

    def test_drift_penalty_affects_ranking(self):
        """Test that drift penalty can demote items."""
        # Item A: Low volume drift
        result_a = create_trigger_result("A", price_z=2.0, volume_multiple=0.5)
        context_a = create_attribution_context("A", result_a, has_company_catalyst=False)

        # Item B: Normal volume
        result_b = create_trigger_result("B", price_z=1.8, volume_multiple=1.2)
        context_b = create_attribution_context("B", result_b, has_company_catalyst=False)

        items = [
            (result_a, context_a, None),
            (result_b, context_b, None),
        ]

        ranked = rank_and_select_enhanced(items, max_items=5)

        # A has base: 4.0 but gets -0.5 drift = 3.5
        # B has base: 3.6 + 0.2 volume = 3.8
        # So B should rank higher
        assert ranked[0].result.ticker == "B"

    def test_ranked_item_has_score_breakdown(self):
        """Test that RankedItem includes full score breakdown."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context("TEST", result, has_company_catalyst=True)

        items = [(result, context, None)]
        ranked = rank_and_select_enhanced(items, max_items=5)

        assert len(ranked) == 1
        item = ranked[0]

        assert item.enhanced_score.base_score > 0
        assert item.enhanced_score.evidence_boost > 0
        assert item.rank == 1


class TestCoverageTier:
    """Tests for coverage tier computation."""

    def test_full_coverage(self):
        """Test that news + SEC checked = Full coverage."""
        checks = CatalystChecks(
            earnings_calendar=CatalystCheckStatus.CHECKED,
            news_feed=CatalystCheckStatus.CHECKED,
            sec_filings=CatalystCheckStatus.CHECKED,
        )
        assert checks.coverage == CoverageTier.FULL

    def test_partial_coverage_news_only(self):
        """Test that only news checked = Partial coverage."""
        checks = CatalystChecks(
            earnings_calendar=CatalystCheckStatus.CHECKED,
            news_feed=CatalystCheckStatus.CHECKED,
            sec_filings=CatalystCheckStatus.NOT_AVAILABLE,
        )
        assert checks.coverage == CoverageTier.PARTIAL

    def test_partial_coverage_sec_only(self):
        """Test that only SEC checked = Partial coverage."""
        checks = CatalystChecks(
            earnings_calendar=CatalystCheckStatus.CHECKED,
            news_feed=CatalystCheckStatus.NOT_AVAILABLE,
            sec_filings=CatalystCheckStatus.CHECKED,
        )
        assert checks.coverage == CoverageTier.PARTIAL

    def test_none_coverage(self):
        """Test that neither news nor SEC checked = None coverage."""
        checks = CatalystChecks(
            earnings_calendar=CatalystCheckStatus.CHECKED,
            news_feed=CatalystCheckStatus.NOT_AVAILABLE,
            sec_filings=CatalystCheckStatus.NOT_AVAILABLE,
        )
        assert checks.coverage == CoverageTier.NONE


class TestConfidenceBoostGatedByCoverage:
    """Tests for confidence boost requiring Full coverage."""

    def test_high_confidence_boost_with_full_coverage(self):
        """Test that high confidence boost is applied when coverage is Full."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context(
            "TEST", result,
            news_checked=True,
            sec_checked=True,  # Full coverage
        )
        explanation = create_explanation(confidence="High")
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, explanation, config)

        assert boost == config.high_confidence_boost
        assert any("high confidence" in r for r in reasons)

    def test_no_high_confidence_boost_with_partial_coverage(self):
        """Test that high confidence boost is NOT applied with Partial coverage."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context(
            "TEST", result,
            news_checked=True,
            sec_checked=False,  # Partial coverage
        )
        explanation = create_explanation(confidence="High")
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, explanation, config)

        # No boost for confidence when coverage is not Full
        assert boost == 0.0
        assert not any("high confidence" in r for r in reasons)

    def test_no_high_confidence_boost_with_none_coverage(self):
        """Test that high confidence boost is NOT applied with None coverage."""
        result = create_trigger_result("TEST", price_z=2.0)
        context = create_attribution_context(
            "TEST", result,
            news_checked=False,
            sec_checked=False,  # None coverage
        )
        explanation = create_explanation(confidence="High")
        config = ScoringConfig()

        boost, reasons = compute_evidence_boost(context, explanation, config)

        # No boost for confidence when coverage is not Full
        assert boost == 0.0


class TestActionabilityWithCatalyst:
    """Tests for tightened actionability rules with catalyst detection."""

    def test_actionable_via_path_a_big_move_plus_confirmation(self):
        """Test Path A: Big price move + relative/volume confirmation."""
        from watchbrief.features.triggers import compute_actionability_label

        # price_z >= 2.0 AND rel_vs_spy_z >= 1.5
        label = compute_actionability_label(
            price_z=2.5,
            volume_multiple=1.0,
            rel_vs_spy_z=1.8,
            rel_vs_sector_z=None,
            triggered=True,
            has_company_catalyst=False,
        )
        assert label == "ACTIONABLE"

    def test_actionable_via_path_a_big_move_plus_volume(self):
        """Test Path A: Big price move + high volume."""
        from watchbrief.features.triggers import compute_actionability_label

        # price_z >= 2.0 AND volume >= 2.0
        label = compute_actionability_label(
            price_z=2.2,
            volume_multiple=2.5,
            rel_vs_spy_z=0.5,
            rel_vs_sector_z=None,
            triggered=True,
            has_company_catalyst=False,
        )
        assert label == "ACTIONABLE"

    def test_actionable_via_path_b_catalyst_with_notable_move(self):
        """Test Path B: Company catalyst + notable move."""
        from watchbrief.features.triggers import compute_actionability_label

        # has_company_catalyst AND price_z >= 1.5
        label = compute_actionability_label(
            price_z=1.6,
            volume_multiple=1.0,
            rel_vs_spy_z=0.5,
            rel_vs_sector_z=None,
            triggered=True,
            has_company_catalyst=True,
        )
        assert label == "ACTIONABLE"

    def test_actionable_via_path_b_catalyst_with_volume(self):
        """Test Path B: Company catalyst + volume."""
        from watchbrief.features.triggers import compute_actionability_label

        # has_company_catalyst AND volume >= 1.5
        label = compute_actionability_label(
            price_z=1.0,
            volume_multiple=1.8,
            rel_vs_spy_z=0.5,
            rel_vs_sector_z=None,
            triggered=True,
            has_company_catalyst=True,
        )
        assert label == "ACTIONABLE"

    def test_monitor_when_triggered_but_not_actionable(self):
        """Test MONITOR when triggered but neither path satisfied."""
        from watchbrief.features.triggers import compute_actionability_label

        # Triggered but: price_z < 2.0, no catalyst, no confirmation
        label = compute_actionability_label(
            price_z=1.8,
            volume_multiple=1.2,
            rel_vs_spy_z=0.5,
            rel_vs_sector_z=None,
            triggered=True,
            has_company_catalyst=False,
        )
        assert label == "MONITOR"

    def test_monitor_big_move_without_confirmation(self):
        """Test MONITOR when big move but no confirmation."""
        from watchbrief.features.triggers import compute_actionability_label

        # Big price_z but: no relative move, no volume, no catalyst
        label = compute_actionability_label(
            price_z=2.5,
            volume_multiple=1.0,
            rel_vs_spy_z=0.5,  # Below 1.5 threshold
            rel_vs_sector_z=0.5,  # Below 1.5 threshold
            triggered=True,
            has_company_catalyst=False,
        )
        assert label == "MONITOR"

    def test_ignore_when_not_triggered(self):
        """Test IGNORE when not triggered."""
        from watchbrief.features.triggers import compute_actionability_label

        label = compute_actionability_label(
            price_z=3.0,
            volume_multiple=3.0,
            rel_vs_spy_z=2.0,
            rel_vs_sector_z=2.0,
            triggered=False,  # Not triggered
            has_company_catalyst=True,
        )
        assert label == "IGNORE"
