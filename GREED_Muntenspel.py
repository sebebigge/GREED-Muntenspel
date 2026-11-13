"""
@author: Sebastian Bigge
@date:   2026-11-13
"""

# Imports
import random


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################

def greedy_aux(A: list) -> int:
    if A[0] > A[-1]:
        return A.pop(0)
    return A.pop()


def greedy(A: list) -> tuple[int, int]:
    player1 = 0
    player2 = 0
    while len(A) > 0:
        player1 += greedy_aux(A)
        if A:
            player2 += greedy_aux(A)
    return player1, player2


def dynamic_programming(A: list) -> tuple[int, int]:
    player1 = 0
    player2 = 0
    n = len(A)
    gain = dict()
    for i in range(2, n):
        for j in range(0, n - i):
            # remainder is zero -> player 1's turn
            if i % 2 == 0:
                gain[(j, j + i)] = max(gain[(j, j + i - 1)] + A[j + i - 1], 
                                   gain[(j + 1, j + i)] + A[j])
            else:
                pass


"""
Player 1: brute force
Player 2: greedy 
"""
def brute_force_greedy(A: list, i: int, j: int) -> int:
    # convert from 1,...,n to 0,...,n-1
    i = i - 1; j = j - 1
    length = j - i + 1
    if length == 1:
        return A[i]
    if length == 2:
        return max(A[i:j+1])
    if length == 3:
        return max(A[i] + min(A[i+1:j+1]), A[j] + min(A[i:j]))
    # first case: DP chooses A[i]
    (i1, j1) = (i + 2, j) if A[i + 1] > A[j] else (i + 1, j - 1)
    # second case: DP chooses A[j]
    (i2, j2) = (i + 1, j - 1) if A[i] > A[j - 1] else (i, j - 2)
    # convert back to 1,...,n
    return max(A[i] + brute_force_greedy(A, i1 + 1, j1 + 1),
               A[j] + brute_force_greedy(A, i2 + 1, j2 + 1))

n = 6
sample = random.sample(range(1,n+1), n)
print(sample)
print(brute_force_greedy(sample, 1, n))

####################################################################################
####################################  HELPERS  #####################################
####################################################################################



####################################################################################
####################################  PLOTS  #######################################
####################################################################################



####################################################################################
####################################  TESTS  #######################################
####################################################################################
