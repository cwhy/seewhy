# At 10M tokens DUST trails backprop by 0.25 nats; at 100K nothing clears the floor

At 10M training tokens DUST with 256 draws reaches **5.232** (paper 5.231), 0.25 nats behind backprop at
**4.984** (paper 4.989). EGGROLL with 256 passes reaches **7.128** (paper 6.906), above a
token-frequency floor of 6.978. At 100K tokens, which is six updates, no method gets below its floor of 7.027, the paper's included.
These are the budgets either side of the 1M-token ladder of report 4.

## 10M tokens

![Test loss at 10M tokens](https://media.tanh.xyz/seewhy/26-10-08/ladder_10M_a.svg)

| run | ours | paper | ours − paper | s/update |
|---|---:|---:|---:|---:|
| backprop | 4.984 ± 0.008 | 4.989 | -0.005 | 0.13 |
| DUST, 256 draws | 5.232 (n=1) | 5.231 | +0.001 | 7.6 |
| EGGROLL, 256 passes | 7.128 (n=1) | 6.906 | +0.222 | 8.1 |

Backprop has three seeds; DUST and EGGROLL one each, because a 610-update run at 256 draws takes about 80 minutes. EGGROLL's
settings are its 1M-token winner, not tuned at this budget, where the paper tunes every cell. DUST uses the authors' released
10M settings. DUST's distance to backprop here, 0.248, is the paper's (0.242) to within
0.006. The EGGROLL–DUST gap is 1.90 nats here against
1.67, wider by 0.22.
DUST at 256 draws is 0.25 nats from backprop here, as in the paper, where DUST stays above backprop at 10M
tokens at every population (its best, 5.049 at 16,384 draws, is 0.060 above). We ran 256 draws only at this budget.

## 100K tokens

![Test loss at 100K tokens](https://media.tanh.xyz/seewhy/26-10-08/ladder_100K_a.svg)

Six updates. Every method is tuned on the paper's grids (seed 42, winner repeated, grid extended one step where a winner sat on an edge).

| population | DUST (ours) | DUST (paper) | EGGROLL (ours) | EGGROLL (paper) | gap, ours | gap, paper |
|---:|---|---:|---|---:|---:|---:|
| 64 | 7.264 ± 0.006 | 7.240 | 8.276 ± 0.006 | 7.961 | 1.01 | 0.72 |
| 256 | 7.235 ± 0.114 | 7.165 | 8.135 ± 0.028 | 7.605 | 0.90 | 0.44 |
| 1,024 | 7.186 ± 0.032 | 7.151 | 7.804 ± 0.009 | 7.386 | 0.62 | 0.24 |

Backprop here is 7.371 ± 0.184 (paper 7.216), and the unigram floor from the 100K prefix is 7.027. Nothing is below that floor, ours or the paper's:
six updates teach a model how often each token occurs and little else. DUST is ahead of EGGROLL at every population here, by more than in the paper.
Backprop's spread over three seeds, 0.184, is 7 times what it is at 1M tokens, so we do not rank DUST against backprop at this budget.

## What this changes

- The 10M-token DUST number reproduces the paper's, and so does its distance to backprop: DUST at small population is not backprop-competitive at 10M.
- The paper's own caveat applies to us a fortiori: the 100K-token column is a measurement of who learns token frequencies fastest.
- Not tested: tuning EGGROLL at 10M tokens, a second and third seed at 10M, and any population above 256 at this budget.

Code: `experiments11.py`, `experiments12.py`, `experiments13.py`.
