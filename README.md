Demonstrates how difficult it is to find the right parameters to have an ELO contest where a user converges on their true value.

Scenario:

We have 10 levels, these are represented by 
- level 1 == ELOs 1..1000,
- level 2 == ELOs 1001..2000,
- ...
- level 10 == ELOs 9001..10000

We want a user of unknown starting ELO to converge on their true ELO. We consider a user converged if
1. Their guessed ELO is within +- 500 of their true ELO
2. THeir variance is less than 500

By running a simulation with varying starting parameters, we can determine which are the best at convergence.

Starting parameters include:
- starting variance,
- variance decay,
- range of competitors from guessed elo

A few results for best params are:

| Avg Rounds before convergence   | Competitor ELO Delta | Starting Variance | Variance Decay % |
|---------------------------------|----------------------|-------------------|------------------|
| 12.1                            | 7690                 | 2250              | 0.0635           |
| 14.2                            | 1060                 | 3860              | 0.117            |
| 15.5                            | 3040                 | 3720              | 0.115            |

This shows we need on average 12-16 rounds to be reasonably confident of convergence.

