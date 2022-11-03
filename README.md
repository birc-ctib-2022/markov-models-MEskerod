# Markov models

For a sequence of observations

```
X = [...]
```


the joint probability of a Markov model is

```
P(X)=pi[X[0]] * T[X[0],X[1]] * T[X[1],X[2]] * ...
```

**Exercise:** Write a function that computes the joint probability of a sequence, `X` given parameters `T` and `pi`. There is a template waiting for you in `src/markov.py`.


A Markov model for sequences of sunny and cloudy days can look like this:

```python
# States
SUNNY = 0
CLOUDY = 1

X = [SUNNY, SUNNY, CLOUDY, SUNNY]

init_probs = [0.1, 0.9]  # it is almost always cloudy
transitions_from_SUNNY = [0.3, 0.7]
transitions_from_CLOUDY = [0.4, 0.6]
transition_probs = [
    transitions_from_SUNNY,
    transitions_from_CLOUDY
]
M = MarkovModel(init_probs, transition_probs)
```


Arbitrarily, I have specified that you have only 10% chance of starting with a sunny day, if today is sunny then tomorrow is only sunny with 30% probability, whereas if it is cloudy you have 40%. Not realistic, I know, but it is the computations rather than the statistical accuracy you should focus on.

**Exercise:** Compute the joint probability for longer sequences. You will discover that the result gets smaller the longer the sequence—not surprisingly since we are multiplying numbers numerically smaller than one. This is a problem when computers only have a finite resolution because they store them as floating point numbers. You can, however, compute the log of the probability instead. Implement that.

The probability distribution that tells you the probability of *k* heads out of *n* tosses, `P(k;n,𝜃)= binom(n,k)𝜃**k * (1-𝜃)**(n-k)` is called the *binomial distribution* (from the binomial coefficient, I think). If you have more than a binary outcome, say you roll dice instead of toss coins, the mathematics is very similar, and the distribution is called a *multinomial distribution*. If you see `n[i]` outcomes of state *i*, in *n* experiments, then the maximum likelihood parameter for the probability of seeing result *i* in a single experiment is `n[i]/n`. For the starting probabilities in a Markov model, 𝜋, you have a multinomial distribution. If you see *n* runs of the Markov model, and `n[i]` of the start in state *i*, then the maximum likelihood parameter for 𝜋 has `𝜋[i]=n[i]/n`.

At first glance, the transition probabilities look different, because we have two variables in play. The transition parameter, *T*, is not a multinomial distribution. That is because it is a set of conditional probabilities, row *k* is the conditional probability `P(X[i]|X[i-1]=k)`. Each of the rows are multinomial distributions, and you can estimate the paramers as such. Take one row at a time, say `T[k,-]`, then look at all the transitions out of *k*—those are the states you are conditioning on—and count how many times you move to each state. If you move from *k* to *h* `n[k,h]` times, and you move out of a *k* state `n[k]` times, then `T[k,h]=n[k,h]/n[k]`.

**Exercise:** Write a program that, given a set of sequences, estimates the parameters for a Markov model.