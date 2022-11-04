from markov import MarkovModel

def intial_probabilities(n: list[list[int]], k: int) -> float:
    """
    Computes the initial porbabilities for k in a list of observed sequences
    """
    sequences = len(n)
    initial_states = []
    for i in range(sequences):
        seq = n[i]
        initial_states.append(seq[0])
    count = 0
    for i in range(len(initial_states)):
        if initial_states[i] == k:
            count += 1
    initial_prob = count/len(initial_states)
    return initial_prob

def transitions_probabilities(n: list[list[int]], k: int, h: int) -> float:
    """
    Computes the probability of transitioning from k to h, from a list of observed sequences
    """
    count_of_k = 0
    count_of_trans = 0
    sequences = len(n)
    for i in range(sequences):
        sequence = n[i]
        for j in range(len(sequence)):
            if sequence[j] == k:
                count_of_k += 1
    if count_of_k == 0:
        return 0
    for i in range(sequences):
        sequence = n[i]
        for j in range(1,len(sequence)):
            if sequence[j] == h and sequence[j-1] == k:
                count_of_trans += 1
    trans_prob = count_of_trans/count_of_k
    return trans_prob

def estimate_parameters(n: list[list[int]]) -> MarkovModel:
    sequences = len(n)
    possible_outcomes = []
    for i in range(sequences):
        sequence = n[i]
        for j in sequence: 
            if j not in possible_outcomes:
                possible_outcomes.append(j)
    possible_outcomes = sorted(possible_outcomes)
    init_probs = []
    trans = []
    for outcome1 in possible_outcomes:
        init_probs.append(intial_probabilities(n, outcome1))
        trans_outcome = []
        for outcome2 in possible_outcomes:
            trans_outcome.append(transitions_probabilities(n, outcome1, outcome2))
        trans.append(trans_outcome)
    return MarkovModel(init_probs, trans)
