import numpy as np
import scipy.spatial
import scipy.stats
import warnings

# The functions in this file are used to measure the level of compositionality
# So far this is implemented by a TRE proxy and topological similarity


def compositionality_metrics(compositional_representation, messages, samples=5000):
    """
    Approximates Topological Similarity and TRE score from https://arxiv.org/abs/1902.07181
    Args:
        compositional_representation (np.array): one-hot encoded compositional, size N*C
        messages (np.array): one-hot encoded messages, size N*M
        samples (int, optional): default 5000 - number of pairs to sample
    Returns:
        topological_similarity (float): correlation between similarity of pairs in representation/messages
        tre_score (float): L1 distance for similarity pairs between representation/messages
                           Note: This is only an approximation of the orinal score,
                                 it correlates heavily with the original TRE, but does
                                 not have the same magnitude.
                                 The full implementation is not yet implemented.
    """
    assert compositional_representation.shape[0] == messages.shape[0]

    sim_representation = np.zeros(samples)
    sim_messages = np.zeros(samples)

    for i in range(samples):
        rnd = np.random.choice(len(messages), 2, replace=False)
        s1, s2 = rnd[0], rnd[1]

        sim_representation[i] = scipy.spatial.distance.hamming(
            compositional_representation[s1], compositional_representation[s2]
        )

        sim_messages[i] = scipy.spatial.distance.cosine(messages[s1], messages[s2])

    # If either metric has a standard deviation of 0 then the Pearson R will be NaN
    if sim_messages.std() == 0.0 or sim_representation.std() == 0.0:
        warnings.warn(
            "Standard deviation of 0.0 for passed parameter in compositionality_metrics"
        )
        topographic_similarity = 0
    else:
        topographic_similarity = scipy.stats.pearsonr(sim_messages, sim_representation)[
            0
        ]
    tre_score = np.linalg.norm(sim_representation - sim_messages, ord=1)

    return (topographic_similarity, tre_score)

