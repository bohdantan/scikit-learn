"""
The :mod:`sklearn.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""


from ._dist_metrics import DistanceMetric

from .pairwise import pairwise_distances_chunked
