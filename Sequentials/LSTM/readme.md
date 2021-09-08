# LSTM football predictions project
---

### __Features:__
- [x] Team rating 
- [x] Team offensive style
- [x] Team defensive style
- [x] Overall team ELO team rating 
- [x]  Odds
- [x] xG - quality of goalscoring chances (attack form)
- [x] xGa - ability to prevent scoring chances (defense form) 
- [x] Ppda - passes allowed per defensive action 
- [x] Strike
- [x] Manager candance time
- Sentiment analysis
- [x] Current standing
- [x] Bets

### __Models:__
- GRU -> many2one

### __Testing:__
- Cross-validation

### __Data Preparation:__
I use ELO ratings equation to calculate team rating based on their last results, so with each match the rating is updated.

Rn = Ro + K(W-We)
We = 1/(10**(-dr/400)+1)
Where,
- Rn -> new rating
- Ro -> old rating
- K -> constant weight according to the relevance of the game in my case I used 50, if df is 1(K = 1.5K), if df = 2(K = 1.75K), if df > 3 df = (1.75K + (df-3)/8)
- dr -> goals difference  
- W -> game result( Win = 1, Draw = 0.5, Loss = 0) 
- We -> calculated expected resulti
