"""analysis package: evaluation and analysis (pipeline.md section 11 / scaf.md sections 10-12; rulings C1/C12)."""

from analysis.accuracy import calculate_accuracy
from analysis.confidence import extract_confidence, extract_interpretation
from analysis.embedding import EmbeddingService, HashEmbedder
from analysis.emergence import calculate_emergence
from analysis.judge import LLMEquivalenceJudge
from analysis.pipeline import run_analysis
from analysis.stability import calculate_stability, step_distances
from analysis.statistics import run_statistics, trajectory_similarity

__all__ = [
    "EmbeddingService",
    "HashEmbedder",
    "LLMEquivalenceJudge",
    "calculate_accuracy",
    "calculate_emergence",
    "calculate_stability",
    "extract_confidence",
    "extract_interpretation",
    "run_analysis",
    "run_statistics",
    "step_distances",
    "trajectory_similarity",
]
