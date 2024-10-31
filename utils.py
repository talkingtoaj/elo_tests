from typing import Union
WINNER = 1
LOSER = 0
DRAW = 0.5


def update(competitor_list, variance_decrease_pc:float=0.1):
    """
        <competitor_list>: a list of competitor objects. 
          These can be any model, but require the following attributes to be pre-set:
          score: ELO score
          variance
          position: must be from WINNER, LOSER, DRAW

        returns competitor_list with updated score and variance values
    """

    num_competitors = len(competitor_list)
    if len(competitor_list) < 1:
        raise ValueError("At least one competitor is required")
    if len(competitor_list) == 1:
        promote_single(winner=competitor_list[0], variance_decrease_pc=variance_decrease_pc)
    if variance_decrease_pc > 1.0 or variance_decrease_pc < 0.0:
        raise ValueError("variance_decrease_pc must be between 0 and 1.0")

    R = {}  # define dictionary
    K = {}
    sum = 0

    for idx, competitor in enumerate(competitor_list):
        # R(1) = 10**(1elo/400)
        # R(2) = 10**(2elo/400)
        R[idx] = 10**(competitor.score/400)
        sum = sum + R[idx]       
        K[idx] = competitor.variance

    # score = 1 for win, 0 for loss, middle players get 0.5/(number of middle players)
    if num_competitors > 2:
        neutral_score = 0.5/(num_competitors-2)
    else:
        neutral_score = DRAW

    for idx, competitor in enumerate(competitor_list):
        result = competitor.position
        if competitor.position == DRAW:
            result = neutral_score
        competitor.score = competitor.score + K[idx] * (result -(R[idx] / sum))
        competitor.variance = competitor.variance * (1-variance_decrease_pc) # decrease K-variance after a contest
    return competitor_list

def promote_single (winner, variance_decrease_pc:float=0.1, competitor_elo=2000):
    """ Given a single competitor selected as suitable, with no losers, ELO and variances are updated against fictional rival"""    
    loser = Competitor(elo_score=2000, variance=0, position=LOSER)
    competitors = [winner, loser]
    return update(competitors)[0]


class Competitor:
    """ If the element you want to elo_compete does not have score and variance attributes you can set, you can use this wrapper class """
    # Unfortunately this currently does not allow us to calculate multiple competitions (e.g. this competitor coming 2nd out of 3 competitors. 
    # That will require two competitions until a wrapper is built)
    def __init__(self, elo_score:Union[int,float], variance:Union[int,float], position, obj=None):
        if position not in [WINNER, LOSER, DRAW]:
            raise ValueError (f"Invalid value for position {position}")
        self.score = elo_score
        self.variance = variance
        self.position = position
        self.obj = obj
