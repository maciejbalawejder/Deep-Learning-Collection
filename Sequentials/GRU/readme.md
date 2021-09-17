# LSTM football predictions project
---

### __Features:__
- [x] Team rating 
- [x] Team offensive style
- [x] Team defensive style
- [x] Overall team ELO team rating 
- [x] Odds
- [x] xG - quality of goalscoring chances (attack form)
- [x] xGa - ability to prevent scoring chances (defense form) 
- [x] Ppda - passes allowed per defensive action 
- [x] Manager candance time
- [x] Bets

### __Models:__
- GRU -> many2one

### __Data Preparation:__
I use ELO ratings equation to calculate team rating based on their last results, so with each match the rating is updated.

Rn = Ro + K(W-We)
We = 1/(10**(-dr/400)+1)
Where,
- Rn -> new rating
- Ro -> old rating
- K -> constant weight according to the relevance of the game in my case I used 50, if dr is 1(K = 1.5K), if dr = 2(K = 1.75K), if dr > 3 dr = (1.75K + (dr-3)/8)
- dr -> goals difference  
- W -> game result( Win = 1, Draw = 0.5, Loss = 0) 
- We -> calculated expected result

|Units| Optimizer | Lr | Dropout | Batch | Epochs | Layers | Accuracy |
|:---:|:---------:|:--:|:-------:|:-----:|:------:|:------:|:--------:|
| 512 |    Adam   |1e-4|    .5   |   1   |   50   |   2    |    65%   |
| 512 |    Adam   |2e-4|    .5   |   1   |   200  |   2    |    86%   |
| 512 |    Adam   |2e-4|    .5   |  30   |   150  |   2    |    85%   |
| 512 |    Adam   |2e-4|    .5   |  50   |   150  |   2    |    85%   |
| 512 |    Adam   |2e-4|    .3   |  30   |   150  |   3    |    85%   |
| 1024|    Adam   |2e-4|    .5   |  30   |   150  |   2    |    81%   |
| 1024|    Adam   |1e-4|    .5   |  30   |   150  |   2    |    81%   |

