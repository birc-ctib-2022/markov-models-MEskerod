"""Testing Markov models."""

from markov import (MarkovModel, likelihood, log_likelihood)
from estimating_parameters import (intial_probabilities, transitions_probabilities, estimate_parameters)

import pytest
from math import (log)

### Tests for markov.py
def create_markov() -> MarkovModel:
    SUNNY = 0
    CLOUDY = 1
    init_probs = [0.3, 0.7]
    transition_from_SUNNY = [0.5, 0.5]
    transition_from_CLOUDY = [0.2, 0.8] 
    transition_probs = [transition_from_SUNNY, transition_from_CLOUDY]
    return MarkovModel(init_probs, transition_probs)

def test_empty():
    mm = create_markov()
    assert pytest.approx(likelihood([], mm)) == 1

def test_initial_prob():
    mm = create_markov()
    assert pytest.approx(likelihood([0],mm)) == 0.3
    assert pytest.approx(likelihood([1], mm)) == 0.7

def test_sequence():
    mm = create_markov()
    assert pytest.approx(likelihood([0,0], mm)) == 0.3 * 0.5 
    assert pytest.approx(likelihood([0,1],mm)) == 0.3 * 0.5 
    assert pytest.approx(likelihood([1, 1, 0, 1], mm)) == 0.7 * 0.8 * 0.2 * 0.5

def test_empty_log(): 
    mm = create_markov
    assert pytest.approx(log_likelihood([], mm)) == log(1)

def test_initial_prob_log():
    mm = create_markov()
    assert pytest.approx(log_likelihood([0],mm)) == log(0.3)
    assert pytest.approx(log_likelihood([1], mm)) == log(0.7)

def test_sequence_log():
    mm = create_markov()
    assert pytest.approx(log_likelihood([0,0], mm)) == sum(map(log,[0.3, 0.5])) 
    assert pytest.approx(log_likelihood([0,1],mm)) == sum(map(log, [0.3, 0.5])) 
    assert pytest.approx(log_likelihood([1, 1, 0, 1], mm)) == sum(map(log, [0.7, 0.8, 0.2, 0.5]))

def test_long_sequences_log(): 
    mm = create_markov()
    assert pytest.approx(log_likelihood(50*[0],mm)) == sum(map(log, [0.3] + 49*[0.5]))
    assert pytest.approx(log_likelihood(50*[0,1], mm)) == sum(map(log, [0.3]+49*[0.5, 0.2]+[0.5]))
    assert pytest.approx(log_likelihood(40*[1,0,0], mm)) == sum(map(log, [0.7]+39*[0.2,0.5,0.5]+[0.2,0.5]))

### Tests for estimating_parameters.py 
def make_set(): 
    SUNNY = 0
    CLOUDY = 1
    week1 = [0, 0, 0, 0, 0, 0, 0]
    week2 = [0, 1, 1, 1, 1, 1, 1]
    week3 = [0, 1, 0, 1, 0, 1, 0]
    wheather = [week1, week2, week3]
    return wheather

def test_initial_probs(): 
    n = make_set()
    assert pytest.approx(intial_probabilities(n, 0)) == 1
    assert pytest.approx(intial_probabilities(n, 1)) == 0

def test_trans_probs(): 
    n = make_set()
    assert pytest.approx(transitions_probabilities(n, 0, 0)) == 6/12
    assert pytest.approx(transitions_probabilities(n, 0, 1)) == 4/12
    assert pytest.approx(transitions_probabilities(n, 1, 0)) == 3/9
    assert pytest.approx(transitions_probabilities(n, 1, 1)) == 5/9

def test_estimating_markov_model(): 
    n = make_set() 
    mm = estimate_parameters(n)

    expected_int = [1,0]
    expected_trans0 = [6/12, 4/12]
    expected_trans1 = [3/9, 5/9]

    for i in range(len(expected_int)):
        assert mm.init_probs[i] == expected_int[i]
    
    for i in range(len(expected_trans0)):
        observed = mm.trans[0]
        assert observed[i] == expected_trans0[i]
    
    for i in range(len(expected_trans1)): 
        observed = mm.trans[1]
        assert observed[i] == expected_trans1[i]