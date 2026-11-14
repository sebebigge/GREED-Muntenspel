def dynamic_programming_dict(A: list) -> int:
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
    return gain[(1, n)]