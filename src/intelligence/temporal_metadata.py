"""
Temporal Intelligence Layer — Problem 3 Fix

Every node and edge in the knowledge graph gains temporal awareness.
The system can now reason about recency, volatility, and change patterns,
enabling insights like:
    "This path is unstable and recently modified — high regression risk"

Temporal Properties (added to every node/edge):
    introduced_at       - When the code artifact first appeared
    last_modified_at    - When it was last changed
    change_frequency    - Changes per time window (rolling)
    volatility_score    - Composite instability metric [0, 1]

Volatility Score Formula:
    volatility = w1 * recency_factor + w2 * frequency_factor + w3 * churn_factor
    where:
        recency_factor  = exp(-days_since_last_change / half_life)
        frequency_factor = min(1, changes_in_window / high_change_threshold)
        churn_factor    = lines_changed / total_lines (if available)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Weights for volatility score components
_W_RECENCY = 0.40
_W_FREQUENCY = 0.35
_W_CHURN = 0.25

# Recency half-life: after this many days, recency contribution drops to 50%
_RECENCY_HALF_LIFE_DAYS = 14.0

# High-change threshold: changes per 30-day window
_HIGH_CHANGE_THRESHOLD = 10

# Default analysis window (days)
_DEFAULT_WINDOW_DAYS = 90


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TemporalMetadata:
    """Temporal metadata for a single code node or edge."""
    introduced_at: Optional[datetime] = None
    last_modified_at: Optional[datetime] = None
    change_count: int = 0
    change_frequency: float = 0.0  # changes per 30-day period
    lines_added: int = 0
    lines_removed: int = 0
    total_lines: int = 0
    volatility_score: float = 0.0
    contributors: List[str] = field(default_factory=list)
    commit_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "introduced_at": self.introduced_at.isoformat() if self.introduced_at else None,
            "last_modified_at": self.last_modified_at.isoformat() if self.last_modified_at else None,
            "change_count": self.change_count,
            "change_frequency": round(self.change_frequency, 4),
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "total_lines": self.total_lines,
            "volatility_score": round(self.volatility_score, 4),
            "contributors_count": len(self.contributors),
        }

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to flat properties suitable for Neo4j node/edge update."""
        props: Dict[str, Any] = {
            "change_count": self.change_count,
            "change_frequency": round(self.change_frequency, 4),
            "volatility_score": round(self.volatility_score, 4),
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "total_lines": self.total_lines,
            "contributors_count": len(self.contributors),
        }
        if self.introduced_at:
            props["introduced_at"] = self.introduced_at.isoformat()
        if self.last_modified_at:
            props["last_modified_at"] = self.last_modified_at.isoformat()
        return props


@dataclass
class ChangeEvent:
    """Represents a single change to a code artifact (from git log)."""
    commit_hash: str
    timestamp: datetime
    author: str
    file_path: str
    lines_added: int = 0
    lines_removed: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# Volatility computation
# ---------------------------------------------------------------------------

def compute_volatility_score(
    last_modified_at: Optional[datetime],
    change_count: int,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    lines_changed: int = 0,
    total_lines: int = 0,
    now: Optional[datetime] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute volatility score for a code artifact.

    Returns:
        (score, breakdown) where score ∈ [0, 1] and breakdown shows factors.
    """
    now = now or datetime.now(timezone.utc)
    breakdown: Dict[str, float] = {}

    # 1. Recency factor
    if last_modified_at:
        if last_modified_at.tzinfo is None:
            last_modified_at = last_modified_at.replace(tzinfo=timezone.utc)
        days_since = max(0, (now - last_modified_at).total_seconds() / 86400)
        recency = math.exp(-days_since * math.log(2) / _RECENCY_HALF_LIFE_DAYS)
    else:
        recency = 0.0
    breakdown["recency_factor"] = round(recency, 4)

    # 2. Frequency factor (changes per window, normalized)
    frequency = min(1.0, change_count / max(1, _HIGH_CHANGE_THRESHOLD))
    breakdown["frequency_factor"] = round(frequency, 4)

    # 3. Churn factor (proportion of code changed)
    if total_lines > 0 and lines_changed > 0:
        churn = min(1.0, lines_changed / total_lines)
    else:
        churn = 0.0
    breakdown["churn_factor"] = round(churn, 4)

    # Weighted sum
    score = (
        _W_RECENCY * recency
        + _W_FREQUENCY * frequency
        + _W_CHURN * churn
    )
    score = max(0.0, min(1.0, score))
    breakdown["volatility_score"] = round(score, 4)

    return score, breakdown


# ---------------------------------------------------------------------------
# Temporal analyzer
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """
    Analyzes git history to produce temporal metadata for code nodes.

    Usage:
        analyzer = TemporalAnalyzer()
        events = analyzer.parse_git_log(git_log_output, repo_name)
        metadata_map = analyzer.compute_metadata(events)
        # metadata_map: Dict[file_path, TemporalMetadata]
    """

    def __init__(self, window_days: int = _DEFAULT_WINDOW_DAYS):
        self._window_days = window_days
        self._stats = {"files_analyzed": 0, "events_processed": 0}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def compute_metadata(
        self,
        events: List[ChangeEvent],
        now: Optional[datetime] = None,
    ) -> Dict[str, TemporalMetadata]:
        """
        Aggregate change events into per-file temporal metadata.

        Args:
            events: List of change events from git history.
            now:    Reference time for recency computation.

        Returns:
            Dict mapping file_path → TemporalMetadata.
        """
        now = now or datetime.now(timezone.utc)
        window_start = now - timedelta(days=self._window_days)

        # Group events by file
        by_file: Dict[str, List[ChangeEvent]] = {}
        for ev in events:
            by_file.setdefault(ev.file_path, []).append(ev)
            self._stats["events_processed"] += 1

        result: Dict[str, TemporalMetadata] = {}

        for file_path, file_events in by_file.items():
            sorted_events = sorted(file_events, key=lambda e: e.timestamp)

            introduced = sorted_events[0].timestamp
            last_modified = sorted_events[-1].timestamp

            # Count changes within the analysis window
            window_events = [e for e in sorted_events if e.timestamp >= window_start]
            window_days = max(1, self._window_days)
            change_freq = len(window_events) / (window_days / 30)  # per 30-day period

            total_added = sum(e.lines_added for e in sorted_events)
            total_removed = sum(e.lines_removed for e in sorted_events)
            lines_changed = total_added + total_removed

            contributors = list({e.author for e in sorted_events})
            commit_hashes = list({e.commit_hash for e in sorted_events})

            # Estimate total lines (last known + net changes is imprecise; use lines_added as proxy)
            total_lines = max(total_added, 1)

            volatility, _ = compute_volatility_score(
                last_modified_at=last_modified,
                change_count=len(window_events),
                window_days=self._window_days,
                lines_changed=lines_changed,
                total_lines=total_lines,
                now=now,
            )

            meta = TemporalMetadata(
                introduced_at=introduced,
                last_modified_at=last_modified,
                change_count=len(sorted_events),
                change_frequency=change_freq,
                lines_added=total_added,
                lines_removed=total_removed,
                total_lines=total_lines,
                volatility_score=volatility,
                contributors=contributors,
                commit_hashes=commit_hashes[-20:],  # Keep last 20 for reference
            )
            result[file_path] = meta
            self._stats["files_analyzed"] += 1

        return result

    def parse_git_log_numstat(
        self,
        git_output: str,
        repository_name: str = "",
    ) -> List[ChangeEvent]:
        """
        Parse output from: git log --numstat --format='COMMIT:%H|%aI|%ae|%s'

        Returns list of ChangeEvent objects.
        """
        events: List[ChangeEvent] = []
        current_commit: Optional[Dict[str, Any]] = None

        for line in git_output.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("COMMIT:"):
                parts = line[7:].split("|", 3)
                if len(parts) >= 3:
                    try:
                        ts = datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
                    except ValueError:
                        ts = datetime.now(timezone.utc)
                    current_commit = {
                        "hash": parts[0],
                        "timestamp": ts,
                        "author": parts[2],
                        "message": parts[3] if len(parts) > 3 else "",
                    }
            elif current_commit and "\t" in line:
                parts = line.split("\t", 2)
                if len(parts) == 3:
                    try:
                        added = int(parts[0]) if parts[0] != "-" else 0
                        removed = int(parts[1]) if parts[1] != "-" else 0
                    except ValueError:
                        added, removed = 0, 0
                    events.append(ChangeEvent(
                        commit_hash=current_commit["hash"],
                        timestamp=current_commit["timestamp"],
                        author=current_commit["author"],
                        file_path=parts[2],
                        lines_added=added,
                        lines_removed=removed,
                        message=current_commit["message"],
                    ))

        return events


def classify_temporal_risk(
    volatility_score: float,
    last_modified_days_ago: float,
) -> Tuple[str, str]:
    """
    Classify temporal risk level for human-readable output.

    Returns:
        (risk_level, explanation)
    """
    if volatility_score >= 0.7 and last_modified_days_ago <= 7:
        return ("critical", "Recently modified and highly volatile — active regression risk")
    if volatility_score >= 0.5 and last_modified_days_ago <= 30:
        return ("high", "Volatile and recently changed — regression likely if modified further")
    if volatility_score >= 0.3:
        return ("medium", "Moderate volatility — requires careful testing if changed")
    if volatility_score >= 0.1:
        return ("low", "Stable code with infrequent changes")
    return ("minimal", "Very stable — rarely changed, low regression risk")


# Singleton
temporal_analyzer = TemporalAnalyzer()
