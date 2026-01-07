"""News processing pipeline for PM-grade 'Why' attribution.

Implements the news_processing_spec_for_claude.md specification:
1. Fetch headlines
2. Filter low-signal headlines
3. Cluster and de-duplicate
4. Classify event types
5. Score relevance
6. Select evidence for attribution
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional
import re


class EventType(str, Enum):
    """Event type taxonomy from spec."""

    # Company-specific
    EARNINGS = "EARNINGS"
    GUIDANCE = "GUIDANCE"
    MNA = "MNA"  # M&A
    FINANCING = "FINANCING"
    REGULATORY_LEGAL = "REGULATORY_LEGAL"
    PRODUCT = "PRODUCT"
    MANAGEMENT = "MANAGEMENT"
    CONTRACT_CUSTOMER = "CONTRACT_CUSTOMER"
    INSIDER_OWNERSHIP = "INSIDER_OWNERSHIP"

    # Non-company
    SECTOR = "SECTOR"
    MACRO = "MACRO"
    ANALYST_ACTION = "ANALYST_ACTION"
    OTHER = "OTHER"
    LOW_SIGNAL = "LOW_SIGNAL"


# Source authority rankings (higher = more authoritative)
SOURCE_AUTHORITY = {
    # Tier 1: Wire services and major financial news
    "reuters": 1.0,
    "bloomberg": 1.0,
    "dow jones": 1.0,
    "associated press": 0.95,
    "wall street journal": 0.95,
    "wsj": 0.95,
    "financial times": 0.95,

    # Tier 2: Official sources
    "pr newswire": 0.90,
    "business wire": 0.90,
    "globenewswire": 0.90,
    "sec edgar": 0.95,
    "sec": 0.95,

    # Tier 3: Major business news
    "cnbc": 0.80,
    "yahoo finance": 0.75,
    "marketwatch": 0.75,
    "barrons": 0.80,
    "seeking alpha": 0.65,
    "benzinga": 0.65,
    "investorplace": 0.55,
    "motley fool": 0.50,

    # Tier 4: Lower quality / aggregators
    "zacks": 0.45,
    "thefly": 0.60,
    "tipranks": 0.55,

    # Default
    "unknown": 0.40,
}


@dataclass
class NewsCluster:
    """A cluster of related news headlines."""

    cluster_id: int
    event_type: EventType
    event_confidence: float
    relevance_score: float
    headline: str
    source: str
    published_at: datetime
    url: Optional[str]
    weak_evidence: bool = False
    cluster_size: int = 1
    keywords_matched: list[str] = field(default_factory=list)
    relevance_rationale: list[str] = field(default_factory=list)


@dataclass
class NewsEvidence:
    """Structured news evidence payload for attribution."""

    checked: bool
    lookback_hours: int
    top_clusters: list[NewsCluster] = field(default_factory=list)
    no_company_specific_catalyst_found: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "checked": self.checked,
            "lookback_hours": self.lookback_hours,
            "top_clusters": [
                {
                    "event_type": c.event_type.value,
                    "event_confidence": c.event_confidence,
                    "relevance_score": c.relevance_score,
                    "headline": c.headline,
                    "source": c.source,
                    "published_at": c.published_at.isoformat(),
                    "url": c.url,
                    "weak_evidence": c.weak_evidence,
                    "cluster_size": c.cluster_size,
                    "keywords_matched": c.keywords_matched,
                }
                for c in self.top_clusters
            ],
            "no_company_specific_catalyst_found": self.no_company_specific_catalyst_found,
        }


# Event classification patterns
EVENT_PATTERNS = {
    EventType.EARNINGS: [
        r"\bq[1-4]\s*(20\d{2})?\s*(results?|earnings?)\b",
        r"\b(beats?|misses?|tops?)\s*(estimates?|expectations?|consensus)\b",
        r"\b(earnings?|eps|revenue)\s*(beat|miss|surprise)\b",
        r"\breports?\s*(q[1-4]|quarterly|annual)\b",
        r"\b(fiscal|fy)\s*(20\d{2})?\s*results?\b",
    ],
    EventType.GUIDANCE: [
        r"\b(raises?|lowers?|cuts?|withdraws?|reaffirms?)\s*(guidance|outlook|forecast)\b",
        r"\b(guidance|outlook)\s*(raised?|lowered?|cut|withdrawn?)\b",
        r"\bpre-?announc",
        r"\bwarns?\s*(on|of|about)?\s*(earnings?|revenue|sales)\b",
    ],
    EventType.MNA: [
        r"\b(acquires?|acquiring|acquisition|acquired)\b",
        r"\b(to\s+buy|buying|bought)\b",
        r"\b(merger|merge|merging)\b",
        r"\b(in\s+talks?|exploring\s+(sale|options))\b",
        r"\b(takeover|take\s*over)\b",
        r"\b(deal|transaction)\s*(value|worth|at)\b",
    ],
    EventType.FINANCING: [
        r"\b(offering|offer)\s*(priced?|of|at)\b",
        r"\batm\s*(offering|program)\b",
        r"\b(convertible|convert)\s*(notes?|bonds?|offering)\b",
        r"\b(debt|bond|notes?)\s*(offering|issue|priced?)\b",
        r"\b(equity|stock|share)\s*(offering|sale)\b",
        r"\braises?\s*\$[\d.]+[mb]\b",
        r"\bipo\b",
        r"\bsecondary\s*(offering|sale)\b",
    ],
    EventType.REGULATORY_LEGAL: [
        r"\b(fda|sec|doj|ftc|eu|cma)\s*(approves?|clears?|blocks?|sues?|investigat|probes?)\b",
        r"\b(lawsuit|litigation|sued|suing)\b",
        r"\b(settles?|settlement)\b",
        r"\b(regulatory|antitrust)\s*(approval|clearance|review)\b",
        r"\b(patent|infringement)\b",
    ],
    EventType.PRODUCT: [
        r"\b(launches?|launching|launch(ed)?)\s*(new|product|service)\b",
        r"\b(recalls?|recalling)\b",
        r"\b(outage|disruption|downtime)\b",
        r"\b(new\s+product|product\s+launch)\b",
    ],
    EventType.MANAGEMENT: [
        r"\b(ceo|cfo|coo|cto|chairman)\s*(steps?\s*down|resigns?|retires?|appointed|named|departs?)\b",
        r"\b(appoints?|names?|hires?)\s*(new\s+)?(ceo|cfo|coo|cto|chairman)\b",
        r"\b(executive|management)\s*(shakeup|changes?|departure)\b",
    ],
    EventType.CONTRACT_CUSTOMER: [
        r"\b(wins?|awarded?|secures?|lands?)\s*(contract|deal|order)\b",
        r"\b(contract|deal|order)\s*(worth|valued?\s*at)\s*\$",
        r"\b(loses?|lost)\s*(contract|customer|client)\b",
        r"\b(partnership|partner)\s*(with|announced)\b",
    ],
    EventType.INSIDER_OWNERSHIP: [
        r"\b(activist|stake|position)\s*(takes?|discloses?|increases?|reduces?)\b",
        r"\b(buyback|repurchase)\b",
        r"\b(insider|director)\s*(buys?|sells?|purchase|sale)\b",
        r"\bform\s*4\b",
        r"\b13[df]\b",
    ],
    EventType.ANALYST_ACTION: [
        r"\b(upgrades?|downgrades?)\s*(to|from)?\s*(buy|sell|hold|neutral|outperform)\b",
        r"\b(price\s*target|pt)\s*(raised?|lowered?|cut|to\s*\$)\b",
        r"\b(initiates?|starts?)\s*(coverage|at)\b",
        r"\b(reiterat|maintains?)\s*(buy|sell|hold)\b",
    ],
    EventType.SECTOR: [
        r"\b(sector|industry)\s*(rallies?|falls?|drops?|surges?)\b",
        r"\b(oil|energy|tech|bank)\s*(stocks?|sector)\s*(up|down|fall|rise)\b",
    ],
    EventType.MACRO: [
        r"\b(fed|federal\s*reserve|interest\s*rates?|inflation)\b",
        r"\b(tariff|trade\s*war|sanctions?)\b",
        r"\b(recession|economic\s*(slowdown|growth))\b",
    ],
}

# Low-signal headline patterns to filter/downrank
LOW_SIGNAL_PATTERNS = [
    r"\b(top\s*movers?|biggest\s*(gainers?|losers?)|stocks?\s*to\s*watch)\b",
    r"\bwhy\s+.+\s+(stock|shares?)\s+(is|are)\s+(up|down|moving|falling|rising)\b",
    r"\b(stock|shares?)\s+(jumps?|falls?|drops?|surges?|plunges?)\s+(on\s+)?(heavy\s+)?volume\b",
    r"\b(technical\s*analysis|breaks?\s*(resistance|support)|trading\s*at)\b",
    r"\b(what\s*happened|here'?s?\s*why|breaking\s*down)\b",
    r"\bmarket\s*(wrap|update|close|open)\b",
    r"\b(best|worst)\s*performing\s*stocks?\b",
    r"\bmomentum\s*stocks?\b",
]


def get_source_authority(source: str) -> float:
    """Get authority score for a news source."""
    source_lower = source.lower()
    for key, score in SOURCE_AUTHORITY.items():
        if key in source_lower:
            return score
    return SOURCE_AUTHORITY["unknown"]


def is_low_signal_headline(headline: str) -> bool:
    """Check if headline is low-signal (recycled 'why it moved' content)."""
    headline_lower = headline.lower()
    for pattern in LOW_SIGNAL_PATTERNS:
        if re.search(pattern, headline_lower):
            return True
    return False


def classify_event_type(headline: str, source: str) -> tuple[EventType, float, list[str]]:
    """Classify headline into event type using keyword patterns.

    Returns:
        Tuple of (event_type, confidence, keywords_matched)
    """
    headline_lower = headline.lower()
    matches = []

    # Check each event type's patterns
    for event_type, patterns in EVENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, headline_lower):
                matches.append((event_type, pattern))

    if not matches:
        # Check if it's low signal
        if is_low_signal_headline(headline):
            return EventType.LOW_SIGNAL, 0.3, []
        return EventType.OTHER, 0.2, []

    # Get the most specific match (company-specific > sector > macro)
    company_specific_types = {
        EventType.EARNINGS, EventType.GUIDANCE, EventType.MNA,
        EventType.FINANCING, EventType.REGULATORY_LEGAL, EventType.PRODUCT,
        EventType.MANAGEMENT, EventType.CONTRACT_CUSTOMER, EventType.INSIDER_OWNERSHIP,
    }

    company_matches = [(t, p) for t, p in matches if t in company_specific_types]

    if company_matches:
        # Prioritize company-specific
        event_type, pattern = company_matches[0]
        confidence = 0.85 if len(company_matches) > 1 else 0.75
    else:
        event_type, pattern = matches[0]
        confidence = 0.65

    keywords = list(set(p for _, p in matches))
    return event_type, confidence, keywords


def compute_headline_similarity(h1: str, h2: str) -> float:
    """Compute similarity between two headlines.

    Uses token overlap + sequence matching for robustness.
    """
    # Normalize
    h1_lower = h1.lower()
    h2_lower = h2.lower()

    # Token overlap (Jaccard)
    tokens1 = set(re.findall(r'\b\w+\b', h1_lower))
    tokens2 = set(re.findall(r'\b\w+\b', h2_lower))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    jaccard = intersection / union if union > 0 else 0.0

    # Sequence similarity
    seq_ratio = SequenceMatcher(None, h1_lower, h2_lower).ratio()

    # Weighted combination
    return 0.4 * jaccard + 0.6 * seq_ratio


def cluster_headlines(
    headlines: list[dict],
    similarity_threshold: float = 0.65,
) -> list[list[dict]]:
    """Cluster similar headlines together.

    Args:
        headlines: List of headline dicts with 'headline', 'source', 'published_at', 'url'
        similarity_threshold: Minimum similarity to cluster together

    Returns:
        List of clusters, each cluster is a list of headline dicts
    """
    if not headlines:
        return []

    # Sort by published_at (earliest first for representative selection)
    sorted_headlines = sorted(headlines, key=lambda h: h.get('published_at', datetime.min))

    clusters = []
    used = set()

    for i, h1 in enumerate(sorted_headlines):
        if i in used:
            continue

        cluster = [h1]
        used.add(i)

        for j, h2 in enumerate(sorted_headlines[i+1:], start=i+1):
            if j in used:
                continue

            similarity = compute_headline_similarity(
                h1.get('headline', ''),
                h2.get('headline', '')
            )

            if similarity >= similarity_threshold:
                cluster.append(h2)
                used.add(j)

        clusters.append(cluster)

    return clusters


def select_cluster_representative(cluster: list[dict]) -> dict:
    """Select the best representative from a cluster.

    Priority:
    1. Most authoritative source
    2. Earliest timestamp
    3. Richest headline
    """
    if len(cluster) == 1:
        return cluster[0]

    # Score each item
    scored = []
    for item in cluster:
        source_score = get_source_authority(item.get('source', 'unknown'))
        # Earlier is better (invert for sorting)
        time_score = 1.0  # Will use published_at for tie-breaking
        # Longer headline = richer content
        richness_score = min(len(item.get('headline', '')) / 100, 1.0)

        total_score = source_score * 0.6 + richness_score * 0.4
        scored.append((total_score, item))

    # Sort by score descending, then by published_at ascending
    scored.sort(key=lambda x: (-x[0], x[1].get('published_at', datetime.max)))

    return scored[0][1]


def compute_relevance_score(
    headline: str,
    source: str,
    published_at: datetime,
    trigger_time: datetime,
    event_type: EventType,
    cluster_size: int,
) -> tuple[float, list[str]]:
    """Compute relevance score for a news cluster.

    Returns:
        Tuple of (score, rationale_list)
    """
    score = 0.0
    rationale = []

    # Company-specific types are more relevant
    company_specific_types = {
        EventType.EARNINGS, EventType.GUIDANCE, EventType.MNA,
        EventType.FINANCING, EventType.REGULATORY_LEGAL, EventType.PRODUCT,
        EventType.MANAGEMENT, EventType.CONTRACT_CUSTOMER, EventType.INSIDER_OWNERSHIP,
    }

    # 1. Recency scoring
    hours_ago = (trigger_time - published_at).total_seconds() / 3600
    if hours_ago <= 12:
        score += 0.35
        rationale.append(f"Very recent ({hours_ago:.0f}h ago)")
    elif hours_ago <= 24:
        score += 0.25
        rationale.append(f"Recent ({hours_ago:.0f}h ago)")
    elif hours_ago <= 48:
        score += 0.15
        rationale.append(f"Within 48h")
    else:
        score -= 0.10
        rationale.append(f"Older news ({hours_ago:.0f}h ago)")

    # 2. Specificity (company-specific vs sector/macro)
    if event_type in company_specific_types:
        score += 0.30
        rationale.append(f"Company-specific ({event_type.value})")
    elif event_type == EventType.ANALYST_ACTION:
        score += 0.20
        rationale.append("Analyst action")
    elif event_type in {EventType.SECTOR, EventType.MACRO}:
        score += 0.05
        rationale.append(f"Broader context ({event_type.value})")

    # 3. Source authority
    authority = get_source_authority(source)
    if authority >= 0.85:
        score += 0.20
        rationale.append(f"Authoritative source ({source})")
    elif authority >= 0.70:
        score += 0.10
        rationale.append(f"Reputable source ({source})")

    # 4. Cluster size (weak proxy for story importance)
    if cluster_size >= 5:
        score += 0.15
        rationale.append(f"Widely covered ({cluster_size} sources)")
    elif cluster_size >= 3:
        score += 0.10
        rationale.append(f"Multiple sources ({cluster_size})")

    # 5. Penalize low-signal
    if event_type == EventType.LOW_SIGNAL:
        score -= 0.25
        rationale.append("Low-signal content")

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return score, rationale


def process_news_for_attribution(
    headlines: list[dict],
    trigger_time: datetime,
    lookback_hours: int = 72,
    relevance_threshold: float = 0.55,
    max_clusters: int = 3,
) -> NewsEvidence:
    """Process news headlines into structured evidence for attribution.

    Args:
        headlines: List of dicts with 'headline', 'source', 'published_at', 'url'
        trigger_time: When the alert triggered
        lookback_hours: How far back we looked for news
        relevance_threshold: Minimum relevance score to include
        max_clusters: Maximum clusters to return

    Returns:
        NewsEvidence object ready for attribution
    """
    if not headlines:
        return NewsEvidence(
            checked=True,
            lookback_hours=lookback_hours,
            top_clusters=[],
            no_company_specific_catalyst_found=True,
        )

    # Step 1: Filter low-signal headlines (but keep for fallback)
    primary_headlines = []
    secondary_headlines = []

    for h in headlines:
        headline_text = h.get('headline', '')
        if is_low_signal_headline(headline_text):
            secondary_headlines.append(h)
        else:
            primary_headlines.append(h)

    # Use primary headlines if available, otherwise fallback to secondary
    working_headlines = primary_headlines if primary_headlines else secondary_headlines

    # Step 2: Cluster headlines
    clusters = cluster_headlines(working_headlines)

    # Step 3: Process each cluster
    processed_clusters = []

    for cluster_id, cluster in enumerate(clusters):
        # Select representative
        rep = select_cluster_representative(cluster)

        headline_text = rep.get('headline', '')
        source = rep.get('source', 'Unknown')
        published_at = rep.get('published_at', trigger_time)
        url = rep.get('url')

        # Classify event type
        event_type, event_confidence, keywords = classify_event_type(headline_text, source)

        # Compute relevance
        relevance_score, rationale = compute_relevance_score(
            headline_text,
            source,
            published_at,
            trigger_time,
            event_type,
            len(cluster),
        )

        processed_clusters.append(NewsCluster(
            cluster_id=cluster_id,
            event_type=event_type,
            event_confidence=event_confidence,
            relevance_score=relevance_score,
            headline=headline_text,
            source=source,
            published_at=published_at,
            url=url,
            weak_evidence=relevance_score < relevance_threshold,
            cluster_size=len(cluster),
            keywords_matched=keywords,
            relevance_rationale=rationale,
        ))

    # Step 4: Sort by relevance and select top clusters
    processed_clusters.sort(key=lambda c: c.relevance_score, reverse=True)

    # Filter by threshold
    strong_clusters = [c for c in processed_clusters if c.relevance_score >= relevance_threshold]

    if strong_clusters:
        top_clusters = strong_clusters[:max_clusters]
        no_catalyst = False
    else:
        # Fallback: keep most recent as weak evidence
        if processed_clusters:
            fallback = processed_clusters[0]
            fallback.weak_evidence = True
            top_clusters = [fallback]
            no_catalyst = True
        else:
            top_clusters = []
            no_catalyst = True

    # Check if any company-specific catalyst found
    company_specific_types = {
        EventType.EARNINGS, EventType.GUIDANCE, EventType.MNA,
        EventType.FINANCING, EventType.REGULATORY_LEGAL, EventType.PRODUCT,
        EventType.MANAGEMENT, EventType.CONTRACT_CUSTOMER, EventType.INSIDER_OWNERSHIP,
    }

    has_company_catalyst = any(
        c.event_type in company_specific_types and not c.weak_evidence
        for c in top_clusters
    )

    return NewsEvidence(
        checked=True,
        lookback_hours=lookback_hours,
        top_clusters=top_clusters,
        no_company_specific_catalyst_found=not has_company_catalyst,
    )
