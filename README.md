A sandbox for running ELO contests where a user converges on their true value.

Scenario:

We have 10 levels, these are represented by 
- level 1 == ELOs 1..1000,
- level 2 == ELOs 1001..2000,
- ...
- level 10 == ELOs 9001..10000

We want a user of unknown starting ELO to converge on their true ELO. We consider a user converged if
1. Their guessed ELO is within +- 500 of their true ELO
2. Their variance is less than 500

By running a simulation with varying starting parameters, we can determine which are the best at convergence.

Starting parameters include:
- starting variance,
- variance decay,

Since we might have an idea of the true ELO, we can include a guess of the true ELO.

Current results show convergence can be reliably achieved with around 6 rounds.