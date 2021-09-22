# Football predictions project
In this project I created an GRU architecture to predict the outcome(Win, Draw, Lose) of the football game.
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

### __Model:__
- [x] GRU -> many2one

### __Data Preparation:__
    I use ELO ratings equation to calculate team rating based on their last results, so with each match the rating is updated.

    Rn = Ro + K(W-We)
    We = 1/(10**(-dr/400)+1)
    Where,
    - Rn -> new rating
    - Ro -> old rating
    - K -> constant weight according to the relevance of the game in my case I used 50, if dr is 1(K = 1.5K), if dr = 2(K = 1.75K), if dr > 3 dr = (1.75K + (dr-    3)/8)
    - dr -> goals difference  
    - W -> game result( Win = 1, Draw = 0.5, Loss = 0) 
    - We -> calculated expected result

