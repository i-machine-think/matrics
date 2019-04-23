import numpy as np
import scipy.spatial
import scipy.stats


def representation_similarity_analysis(inputs, hidden_S, hidden_R, samples=5000):
    """
    Calculates RSA scores of two agents (ρS/R), and of each agent with the
    input (ρS/I and ρR/I), where S refers to Sender,R to Receiver,I to input.
    Args:
        inputs (np.array): input features, size N*F
        hidden_S (np.array): encoded/latent representation in sender, size N*S
        hidden_R (np.array): encoded/latent representation in receiver, size N*R
        samples (int, optional): default 5000 - number of pairs to sample
    Returns:
        rsa_sr (float): correlation between similarity of pairs in Sender/Receiver (ρS/R)
        rsa_si (float): correlation between similarity of pairs in Sender/Input (ρS/I)
        rsa_ri (float): correlation between similarity of pairs in Receiver/Input (ρR/I)
    """
    assert inputs.shape[0] == hidden_S.shape[0] and inputs.shape[0] == hidden_R.shape[0]

    sim_input_features = np.zeros(samples)
    sim_hidden_S = np.zeros(samples)
    sim_hidden_R = np.zeros(samples)

    for i in range(samples):
        rnd = np.random.choice(len(inputs), 2, replace=False)
        s1, s2 = rnd[0], rnd[1]

        sim_hidden_S[i] = scipy.spatial.distance.cosine(inputs[s1], inputs[s2])
        sim_hidden_S[i] = scipy.spatial.distance.cosine(hidden_S[s1], hidden_S[s2])
        sim_hidden_R[i] = scipy.spatial.distance.cosine(hidden_R[s1], hidden_R[s2])

    rsa_sr = scipy.stats.pearsonr(sim_hidden_S, sim_hidden_R)[0]
    rsa_si = scipy.stats.pearsonr(sim_hidden_S, sim_image_features)[0]
    rsa_ri = scipy.stats.pearsonr(sim_hidden_R, sim_image_features)[0]

    return (rsa_sr, rsa_si, rsa_ri)
