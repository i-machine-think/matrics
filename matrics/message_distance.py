import itertools
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from .utils import one_hot

# The functions in this file are used to compare different languages with each other
# Doing so allows us to mesure the similarity between languages,
# and thus determine whether multiple agents are indeed speaking the same language.


def message_distance(messages):
    """
    Returns an averge message distance from a set of different messages.
    This is useful to determine whether a group of A agents are speaking
    the same language. If they are then for each example the edit distance
    will be 0, since they will generate the same message.
    Additionaly the average number of perfect_matches (messages that are identical)
    between pairs of agents are returned.
    Args:
        messages (ndarray, ints): N messages of length L from A agents, shape: N*A*L
    Returns:
        tot_dist (float): average edit distance between all messages
                          in all possible pairs of agent.
        perfect_matches (float): average number of perfect matches in all messages,
                                 and all possible pairs of agents
    """
    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    encoded_messages = one_hot(messages).reshape(N, A, -1).astype(float)
    tot_dist = 0
    perfect_matches = 0
    for c in combinations:
        diff = np.sum(
            np.abs(encoded_messages[:, c[0], :] - encoded_messages[:, c[1], :]), axis=1
        )
        perfect_matches += np.count_nonzero(diff == 0)
        tot_dist += np.sum(diff)

    # average over number of number of combinations and examples
    tot_dist /= N * len(combinations)
    perfect_matches /= N * len(combinations)

    return (tot_dist, perfect_matches)


def jaccard_similarity(messages):
    """
    Calculates average jaccard similarity between all pairs of agents.
    Args:
        messages (ndarray, ints): N messages of length L from A agents, shape: N*A*L
    Returns:
        score (float): average jaccard similarity between all pairs of agents.
    """
    N, A = messages.shape[0], messages.shape[1]
    combinations = list(itertools.combinations(range(A), 2))
    score = 0.0
    for c in combinations:
        score += jaccard_similarity_score(messages[:, c[0], :], messages[:, c[1], :])

    # average over number of combinations
    score /= len(combinations)

    return score


def kl_divergence(messages, eps=1e-6):
    """
    Calculates average KL divergence between all pairs of agents.
    Args:
        messages (ndarray, ints): N messages of length L from A agents, shape: N*A*L
    Returns:
        score (float): average pair-wise KL divergence
    """
    N, A = messages.shape[0], messages.shape[1]

    vocab_size = messages.max() + 1

    score = 0.0
    count = 1
    for i in range(A):
        for j in range(A):
            if j == i:
                continue
            else:
                # this is unnefficient - unfortunately bincount and entropy do not acce
                for m in range(N):
                    dist1 = np.bincount(messages[m, i, :], minlength=vocab_size) + eps
                    dist2 = np.bincount(messages[m, j, :], minlength=vocab_size) + eps
                    score += scipy.stats.entropy(dist1, dist2)
                    count += 1
    # average over number of combinations
    score /= count

    return score
