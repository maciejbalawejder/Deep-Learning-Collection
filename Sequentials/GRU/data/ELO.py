import numpy as np

# Equation -> https://www.eloratings.net/about

def ELO_rating(rating_t1, rating_t2, score_t1, score_t2, info):
    # t1 -> team 1
    # t2 -> team2
    # info "h" -> home, "a" -> away
    K = 50
    goal_diff = score_t1 - score_t2
    dr = rating_t1 - rating_t2
    dr += 100

    if info == 'a':
        goal_diff = -goal_diff
        dr = -dr + 100

    if goal_diff == 2:
        K += K * 0.5
    elif goal_diff == 3:
        K += K*0.75
    elif goal_diff > 3:
        K+= (0.75 + (goal_diff - 3)/8) * K

    if goal_diff > 0:
        W = 1
    elif goal_diff < 0:
        W = 0
    else:
        W = 0.5

    We = 1/(np.power(10,-dr/400)+1)

    return rating_t1 + K * (W-We)
