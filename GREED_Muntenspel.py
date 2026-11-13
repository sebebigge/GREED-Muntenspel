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


"""
Both players use a greedy approach to maximize their gain
"""
def greedy(A: list) -> tuple[int, int]:
    player1 = 0
    player2 = 0
    while len(A) > 0:
        player1 += greedy_aux(A)
        if A:
            player2 += greedy_aux(A)
    return player1, player2


"""
Both players use dynamic programming (bottom-up) to maximize their gain
"""
def dynamic_programming(A: list) -> dict[tuple[int, int], int]:
    n = len(A); gain = {}
    partial_sum = [0] * (n + 1)
    for i, a in enumerate(A):
        partial_sum[i + 1] = partial_sum[i] + a
    for l in range(2, n + 1):
        for k in range(n - l + 1):
            j = k + l
            if l == 2:
                gain[(k, j)] = max(A[k], A[k + 1])
            else:
                total = partial_sum[j] - partial_sum[k]
                gain[(k, j)] = total - min(gain[(k + 1, j)], gain[(k, j - 1)])
    return gain


"""
One player uses a dynamic programming approach (bottom-up) while the other applies a greedy approach
"""
def dynamic_greedy(A: list) -> dict[tuple[int, int], int]:
    n = len(A); gain = {}
    for l in range(2, n + 1):
        for k in range(n - l + 1):
            j = k + l
            if l == 2:
                gain[(k, j)] = max(A[k], A[k + 1])
            elif l == 3:
                gain[(k, j)] = max(A[k] + min(A[k + 1:j]), A[j - 1] + min(A[k:j - 1]))
            else:
                (k1, j1) = (k + 2, j) if A[k + 1] > A[j - 1] else (k + 1, j - 1)
                (k2, j2) = (k + 1, j - 1) if A[k] > A[j - 2] else (k, j - 2)
                gain[(k, j)] = max(A[k] + gain[(k1, j1)], A[j - 1] + gain[(k2, j2)])
    return gain


"""
Player 1: brute force (instead of dynamic programming)
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
print(dynamic_greedy(sample))

####################################################################################
####################################  HELPERS  #####################################
####################################################################################



####################################################################################
####################################  PLOTS  #######################################
####################################################################################



####################################################################################
####################################  TESTS  #######################################
####################################################################################
