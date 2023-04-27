# Poisoning the Search Space in Neural Architecture Search.

This repository contains the poisoning experiments carried out on the [ENAS algorithm](https://github.com/rusbridger/ENAS-Experiments).

| Group                   | Poisoning Set                                                |
| ----------------------- | ------------------------------------------------------------ |
| Baseline                | P0 = {}                                                      |
| Identity                | P1 = {Identity}                                              |
| Gaussian Noise          | P2 = {Gaussian(sigma=2.0) **or** Gaussian(sigma=10.0)}       |
| Dropout                 | P3 = {Dropout(p=0.9) **or** Dropout(p=1.0)}                  |
| Transposed Convolutions | P4 = {3x3 TransposedConv **and** 5x5 Transposed Conv}        |
| Stretched Convolutions  | P5 = {Conv(k=3, p=50, d=50)}                                 |
| Grouped Operations      | P6 = (P1 **or** {Conv(k=3, p=54, d=54)}) + P2 + P3 + P4 + P5 |

See individual experiment files for branches and probabilistic pick functions.
