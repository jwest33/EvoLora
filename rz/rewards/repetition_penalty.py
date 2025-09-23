"""Repetition penalty for encouraging diversity in problem generation"""
import numpy as np
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering


class RepetitionPenalty:
    """Compute repetition penalty using BLEU-based clustering

    This penalty discourages the Challenger from generating similar problems
    within a batch, promoting diversity in the curriculum.
    """

    def __init__(self, tau_bleu: float = 0.5):
        """Initialize repetition penalty calculator

        Args:
            tau_bleu: BLEU threshold for clustering (default 0.5)
        """
        self.tau_bleu = tau_bleu
        self.name = "repetition"
        self.smoothing = SmoothingFunction()

    def compute(self, problems: List[Dict]) -> List[float]:
        """Compute repetition penalties for a batch of problems

        Args:
            problems: List of generated problems

        Returns:
            List of repetition penalties (higher = more repetition)
        """
        if len(problems) <= 1:
            return [0.0] * len(problems)

        # Extract questions for comparison
        questions = [p.get("question", "") for p in problems]

        # Compute pairwise BLEU distances
        distance_matrix = self._compute_distance_matrix(questions)

        # Perform clustering
        clusters = self._cluster_problems(distance_matrix)

        # Compute penalties based on cluster sizes
        penalties = self._compute_cluster_penalties(clusters, len(problems))

        return penalties

    def _compute_distance_matrix(self, questions: List[str]) -> np.ndarray:
        """Compute pairwise BLEU distance matrix

        Args:
            questions: List of question strings

        Returns:
            Distance matrix where d[i,j] = 1 - BLEU(q_i, q_j)
        """
        n = len(questions)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # Tokenize questions
                ref = questions[i].lower().split()
                hyp = questions[j].lower().split()

                # Compute BLEU score with smoothing
                try:
                    bleu = sentence_bleu(
                        [ref],
                        hyp,
                        smoothing_function=self.smoothing.method1
                    )
                except:
                    bleu = 0.0

                # Convert to distance
                distance = 1.0 - bleu
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    def _cluster_problems(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster problems based on BLEU distances

        Args:
            distance_matrix: Pairwise distance matrix

        Returns:
            Array of cluster labels
        """
        if distance_matrix.shape[0] == 0:
            return np.array([])

        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.tau_bleu,
            metric='precomputed',
            linkage='average'
        )

        try:
            cluster_labels = clustering.fit_predict(distance_matrix)
        except:
            # Fallback if clustering fails
            cluster_labels = np.arange(distance_matrix.shape[0])

        return cluster_labels

    def _compute_cluster_penalties(
        self,
        cluster_labels: np.ndarray,
        batch_size: int
    ) -> List[float]:
        """Compute penalties based on cluster sizes

        Args:
            cluster_labels: Cluster assignment for each problem
            batch_size: Total number of problems

        Returns:
            List of penalties proportional to cluster size
        """
        if len(cluster_labels) == 0:
            return []

        penalties = []
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        cluster_size_map = dict(zip(unique_clusters, cluster_counts))

        for label in cluster_labels:
            cluster_size = cluster_size_map[label]
            # Penalty proportional to relative cluster size
            penalty = cluster_size / batch_size
            penalties.append(penalty)

        return penalties

    def get_diversity_score(self, problems: List[Dict]) -> float:
        """Compute overall diversity score for a batch

        Args:
            problems: List of generated problems

        Returns:
            Diversity score (0 = all identical, 1 = all unique)
        """
        if len(problems) <= 1:
            return 1.0

        penalties = self.compute(problems)
        avg_penalty = np.mean(penalties)

        # Convert penalty to diversity score
        diversity = 1.0 - avg_penalty
        return max(0.0, min(1.0, diversity))
