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
def greedy(A: list) -> int:
    player1 = 0
    while len(A) > 0:
        player1 += greedy_aux(A)
        if A:
            greedy_aux(A)
    return player1


"""
Both players use dynamic programming (bottom-up) to maximize their gain
"""
def dynamic_programming(A: list) -> int:
    n = len(A); gain = {}
    partial_sum = [0] * (n + 1)
    for i, a in enumerate(A):
        partial_sum[i + 1] = partial_sum[i] + a
    for l in range(2, n + 1):
        for k in range(1, n - l + 2):
            j = k + l - 1
            if l == 2:
                gain[(k, j)] = max(A[k - 1], A[k])
            else:
                total = partial_sum[j] - partial_sum[k - 1]
                gain[(k, j)] = total - min(gain[(k + 1, j)], gain[(k, j - 1)])
    return gain[(1,n)]


"""
The first player uses a dynamic programming approach (bottom-up) while the second applies a greedy approach
"""
def dp_first_vs_greedy(A: list) -> int:
    n = len(A); gain = {}
    for l in range(2, n + 1):
        for k in range(1, n - l + 2):
            j = k + l - 1
            if l == 2:
                gain[(k, j)] = max(A[k - 1], A[k])
            elif l == 3:
                gain[(k, j)] = max(A[k - 1] + min(A[k:j]), A[j - 1] + min(A[k - 1:j - 1]))
            else:
                (k1, j1) = (k + 2, j) if A[k] > A[j - 1] else (k + 1, j - 1)
                (k2, j2) = (k + 1, j - 1) if A[k - 1] > A[j - 2] else (k, j - 2)
                gain[(k, j)] = max(A[k - 1] + gain[(k1, j1)], A[j - 1] + gain[(k2, j2)])
    return gain[(1,n)]


"""
The first player uses a greedy approach while the second applies a dynamic programming (bottom-up) approach
"""
def greedy_first_vs_dp(A: list) -> int:
    return sum(A) - dp_first_vs_greedy(A[1:] if A[0] > A[-1] else A[:-1])


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

n = 200
sample = random.sample(range(1,n+1), n)
#print(sample)
print(f"Dynamic programming: {dynamic_programming(sample)}")
print(f"DP first vs greedy: {dp_first_vs_greedy(sample)}")
print(f"Greedy first vs DP: {greedy_first_vs_dp(sample)}")
print(f"Greedy: {greedy(sample.copy())}")

####################################################################################
####################################  HELPERS  #####################################
####################################################################################



####################################################################################
####################################  PLOTS  #######################################
####################################################################################



####################################################################################
####################################  TESTS  #######################################
####################################################################################

