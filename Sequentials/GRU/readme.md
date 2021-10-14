# Football predictions project
In this project I created an __GRU__ architecture to predict the outcome(Win, Draw, Lose) of the Premier League game. I recommend checking out my [medium article](https://medium.com/nerd-for-tech/premier-league-predictions-using-artificial-intelligence-7421dddc8778) about the topic.

### 1) __Data Preparation:__
The final dataset contains the combined features listed below from Premier League season 19/20 and 20/21. The format of sequential data is [sample, time step, sequence]. Each sample contains the features of the home team, away team, and odds from last 5 games. To make the form more visible I created the rating column for each team with the initial value taken from Premier League Fantasy dataset. The rating is updated with each game, based on the ELO rating formula.

    Rn = Ro + K(W-We)
    We = 1/(10**(-dr/400)+1)
    Where,
    - Rn -> new rating
    - Ro -> old rating
    - K -> constant weight according to the relevance of the game in my case I used 50, if dr is 1(K = 1.5K), if dr = 2(K = 1.75K), if dr > 3 dr = (1.75K + (dr-    3)/8)
    - dr -> goals difference  
    - W -> game result( Win = 1, Draw = 0.5, Loss = 0) 
    - We -> calculated expected result

### 2) __Extracted Features:__
- __Attacking style(AS)__  -  taken from FIFA21 dataset 
- __Defensive style(DS)__  -  taken from FIFA21 dataset
- __Rating home (SOH)__  -  initial ratings is from Premier League Fantasy dataset updated with each game using ELO rating equation
- __Rating away (SOA)__  -  same as above but for away games 
- __xG__  -  quality of goalscoring chances (attack form)
- __xGa__  -  ability to prevent scoring chances (defense form) 
- __Ppda__  -  passes allowed per defensive action 
- __Mtime__  -  managers time calculated from the day he started to the date of the game
- __Odds__  -  bets on outcome of the game home win, draw, away win 

### 3) __Model Architecture:__

| Hyperparameter | Value |
|:--------------:|:-----:|
| Batch          | 16    |
| Optimizer      | Adam  |
| Learning rate  | 0.0001|
| Dropout rate   |  0.2  | 
| Epochs         | 70    | 

![Architecture](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/Sequentials/GRU/figures/Architecture.png)

### 4) __Results__:
- Accuracy on testing dataset : 92%
![Results](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/Sequentials/GRU/figures/Loss92.png)

### 5) __Improvements__:
- Live line-ups 
- Live form of the players
- Game schedule - add Champions League and other Tournaments
- Weather conditions
- Referee
- Relationship between teams
- Sentiment score

### 6) __Datasets links__:
- https://www.kaggle.com/cashncarry/fifa-21-players-teams-full-database?select=teams_fifa21.csv
- http://github.com/vaastav/Fantasy-Premier-League
- https://www.football-data.co.uk/englandm.php
- https://fbref.com/en/
