"""
@author: Sebastian Bigge
@date:   2026-11-14
"""

# Imports
import os
import random
import numpy as np
import matplotlib.pyplot as plt


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################

# -------------------------  efficient implementations  -----------------------------

# GenAI
"""
Both players use a greedy approach to maximize their gain
"""
def greedy(A: list) -> int:
    lo, hi = 0, len(A) - 1
    player1 = 0
    first = True
    while lo <= hi:
        if A[lo] > A[hi]:
            v = A[lo]; lo += 1
        else:
            v = A[hi]; hi -= 1
        if first:
            player1 += v
        first = not first
    return player1


# GenAI
"""
Both players use dynamic programming (bottom-up) to maximize their gain

Bottom-up over the window length l. Only the previous length is kept (O(n) memory), window
sums come from prefix sums, and all windows of one length are computed at once with NumPy.
prev[k] = gain of the player to move on the window A[k : k + l - 1]
"""
def dynamic_programming(A: list) -> int:
    n = len(A)
    if n == 0:
        return 0
    a = np.asarray(A, dtype=np.int64)
    partial_sum = np.concatenate(([0], np.cumsum(a)))
    prev = a.copy()                                          # windows of length 1
    for l in range(2, n + 1):
        m = n - l + 1                                        # number of windows of length l
        prev = partial_sum[l:] - partial_sum[:m] - np.minimum(prev[:m], prev[1:m + 1])
    return int(prev[0])


# GenAI
"""
The first player uses a dynamic programming approach (bottom-up) while the second applies a greedy approach

The greedy reply is forced, so a window of length l only depends on windows of length l - 2:
two layers are kept (O(n) memory). layer[k] = gain of the DP player on the window A[k : k + length]
"""
def dp_first_vs_greedy(A: list) -> int:
    n = len(A)
    if n == 0:
        return 0
    a = np.asarray(A, dtype=np.int64)
    two_back = np.zeros(n + 1, dtype=np.int64)               # windows of length 0
    one_back = a.copy()                                      # windows of length 1
    for l in range(2, n + 1):
        m = n - l + 1
        # take the left end A[k]: greedy then removes the larger end of A[k+1 .. k+l-1]
        left = a[:m] + np.where(a[1:m + 1] > a[l - 1:], two_back[2:m + 2], two_back[1:m + 1])
        # take the right end A[k+l-1]: greedy then removes the larger end of A[k .. k+l-2]
        right = a[l - 1:] + np.where(a[:m] > a[l - 2:l - 2 + m], two_back[1:m + 1], two_back[:m])
        two_back, one_back = one_back, np.maximum(left, right)
    return int(one_back[0])


# GenAI
"""
The first player uses a greedy approach while the second applies a dynamic programming (bottom-up) approach
"""
def greedy_first_vs_dp(A: list) -> int:
    if not A:
        return 0
    return sum(A) - dp_first_vs_greedy(A[1:] if A[0] > A[-1] else A[:-1])


# -------------------------  original implementations  -----------------------------

def greedy_aux(A: list) -> int:
    if A[0] > A[-1]:
        return A.pop(0)
    return A.pop()


def greedy_pop(A: list) -> int:
    A = A.copy()
    player1 = 0
    while len(A) > 0:
        player1 += greedy_aux(A)
        if A:
            greedy_aux(A)
    return player1


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


def dp_first_vs_greedy_dict(A: list) -> int:
    n = len(A); gain = {}
    if n <= 1:
        return sum(A)
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
    return gain[(1, n)]


def greedy_first_vs_dp_dict(A: list) -> int:
    if not A:
        return 0
    return sum(A) - dp_first_vs_greedy_dict(A[1:] if A[0] > A[-1] else A[:-1])


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


####################################################################################
####################################  HELPERS  #####################################
####################################################################################

# GenAI
# name -> function taking a list and returning the gain of player 1
STRATEGIES = {
    "greedy":              greedy,
    "dynamic_programming": dynamic_programming,
    "dp_first_vs_greedy":  dp_first_vs_greedy,
    "greedy_first_vs_dp":  greedy_first_vs_dp,
}

# GenAI
COLORS = {
    "greedy":              "tab:orange",
    "dynamic_programming": "tab:blue",
    "dp_first_vs_greedy":  "tab:green",
    "greedy_first_vs_dp":  "tab:red",
}

# GenAI
# (a, b, description): a - b
DIFFERENCES = [
    ("dp_first_vs_greedy", "greedy",
     "DP-first vs greedy  minus  greedy vs greedy\n(gain of switching to DP against a greedy opponent)"),
    ("greedy", "greedy_first_vs_dp",
     "greedy vs greedy  minus  greedy-first vs DP\n(loss of greedy when the opponent switches to DP)"),
    ("dynamic_programming", "greedy_first_vs_dp",
     "DP vs DP  minus  greedy-first vs DP\n(gain of switching to DP as first player, optimal opponent)"),
    ("dp_first_vs_greedy", "dynamic_programming",
     "DP-first vs greedy  minus  DP vs DP\n(gain from facing a greedy instead of an optimal opponent)"),
]

# GenAI
# (dp strategy, greedy strategy, description): the two strategies of player 1 that face the SAME
# opponent, so that the comparison "greedy versus DP" only differs in the strategy of player 1
LOSS_PAIRS = [
    ("dp_first_vs_greedy",  "greedy",             "Opponent plays greedy"),
    ("dynamic_programming", "greedy_first_vs_dp", "Opponent plays DP"),
]


####################################################################################
####################################  ORDERING  ####################################
####################################################################################

def _require_even_n(n: int) -> None:
    if n < 2 or n % 2:
        raise ValueError(f"n must be a positive even integer, got {n}")


def _random(n: int, rng: random.Random) -> list:
    _require_even_n(n)
    return rng.sample(range(1, n + 1), n)


def _sorted(n: int, rng: random.Random) -> list:
    _require_even_n(n)
    return sorted(range(1, n + 1))


def _alternating(n: int, rng: random.Random) -> list:
    _require_even_n(n)
    h = (n + 1) // 2
    high = list(range(n - h + 1, n + 1))
    rng.shuffle(high)
    low = list(range(1, n - h + 1))
    rng.shuffle(low)
    A = [0] * n
    A[0::2] = high
    A[1::2] = low
    return A


# GenAI
def _alternating_noisy(noise: float):
    """Alternating arrangement followed by round(noise * n) random swaps."""
    def generate(n: int, rng: random.Random) -> list:
        A = _alternating(n, rng)
        for _ in range(int(round(noise * n))):
            i, j = rng.randrange(n), rng.randrange(n)
            A[i], A[j] = A[j], A[i]
        return A
    return generate


# GenAI
"""
name -> (generator, deterministic). A deterministic arrangement gives the same array
for every trial, so only one trial is run for it.
"""
ARRANGEMENTS = {
    "random":           (_random, False),
    "sorted":            (_sorted, True),
    "alternating":      (_alternating, False),
    "alternating_1pct": (_alternating_noisy(0.01), False),
    "alternating_5pct": (_alternating_noisy(0.05), False),
    "alternating_20pct": (_alternating_noisy(0.20), False),
}

# GenAI
ARRANGEMENT_COLORS = dict(zip(ARRANGEMENTS, plt.cm.tab10.colors))


# GenAI
def make_array(n: int, arrangement: str, rng: random.Random) -> list:
    return ARRANGEMENTS[arrangement][0](n, rng)


# GenAI
def log_spaced_sizes(n_min: int, n_max: int, points: int) -> list:
    """Return unique, even integer sizes on a logarithmic scale, within the given bounds."""
    if n_min < 1 or n_max < n_min:
        raise ValueError("expected 1 <= n_min <= n_max")
    if points < 1:
        raise ValueError("points must be >= 1")
    first_even = n_min + n_min % 2
    last_even = n_max - n_max % 2
    if first_even > last_even:
        raise ValueError("the range must contain at least one positive even size")
    sizes = np.logspace(np.log10(first_even), np.log10(last_even), points)
    sizes = np.clip(2 * np.round(sizes / 2), first_even, last_even)
    return np.unique(sizes.astype(int)).tolist()


####################################################################################
####################################  PLOTS  #######################################
####################################################################################

# GenAI
def _finish(fig, name: str, save_dir, show: bool) -> None:
    for ax in fig.axes:
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis="both", labelsize=14)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 2.5))
            if line.get_marker() != "None":
                line.set_markersize(max(line.get_markersize(), 6))
        legend = ax.get_legend()
        if legend:
            for legend_text in legend.get_texts():
                label = legend_text.get_text()
                if label:
                    legend_text.set_text(label[:1].upper() + label[1:])
                legend_text.set_fontsize(14)
            for line in legend.get_lines():
                line.set_linewidth(max(line.get_linewidth(), 2.5))
                if line.get_marker() != "None":
                    line.set_markersize(max(line.get_markersize(), 6))
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


# GenAI
def _label(name: str) -> str:
    label = name.replace("_", " ")
    return label[:1].upper() + label[1:]


# GenAI
def _mean_ci(y: np.ndarray):
    """Mean over the trials (axis 1) and the half-width of its 95% confidence interval."""
    mean = y.mean(axis=1)
    trials = y.shape[1]
    if trials < 2:
        return mean, np.zeros_like(mean)
    return mean, 1.96 * y.std(axis=1, ddof=1) / np.sqrt(trials)


# GenAI
def _relative_loss(res: dict, dp_name: str, greedy_name: str) -> np.ndarray:
    """Relative loss (%) of player 1 when using greedy instead of DP against the same opponent,
    per size and per trial: 100 * (gain_DP - gain_greedy) / gain_DP."""
    dp, greedy_gain = res["gain"][dp_name], res["gain"][greedy_name]
    return 100 * (dp - greedy_gain) / dp


# GenAI
def _slug(text: str) -> str:
    return text.lower().replace(" ", "_")


# GenAI
def _common_ylim(arrays, pad: float = 0.05):
    """y-limits that fit all given arrays, so that separate figures stay comparable."""
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    margin = pad * ((hi - lo) or 1.0)
    return lo - margin, hi + margin


# -------------------------  plots for ONE arrangement  -----------------------------
# Every plot function creates one figure per graph; the file name says what it shows.

# GenAI
def plot_gain_share(res: dict, save_dir=None, show: bool = True) -> None:
    """Mean gain as a fraction of the total n(n+1)/2: removes the n^2 growth."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in res["gain"].items():
        ax.plot(res["sizes"], (g / res["total"][:, None]).mean(axis=1), marker="o", ms=4,
                color=COLORS[name], label=_label(name))
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="Input size n", ylabel="Mean gain / total")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _finish(fig, "gain_share_vs_n", save_dir, show)


def plot_mean_std(res: dict, save_dir=None, show: bool = True) -> None:
    """Plot mean gain / total with a band of plus or minus one standard deviation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = res["sizes"]
    for name, g in res["gain"].items():
        y = g / res["total"][:, None]
        mean = y.mean(axis=1)
        std = y.std(axis=1, ddof=1) if y.shape[1] > 1 else np.zeros_like(mean)
        ax.plot(x, mean, marker="o", ms=4, color=COLORS[name], label=_label(name))
        ax.fill_between(x, mean - std, mean + std, color=COLORS[name], alpha=0.2)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="Input size n", ylabel="Mean gain / total")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _finish(fig, "mean_gain_share_std_vs_n", save_dir, show)


# GenAI
def plot_range(res: dict, save_dir=None, show: bool = True) -> None:
    """One figure per strategy: min-max band, interquartile band and mean of gain / total
    against n. All figures use the same y-range so that they stay comparable."""
    x = res["sizes"]
    shares = {name: g / res["total"][:, None] for name, g in res["gain"].items()}
    ylim = _common_ylim(list(shares.values()) + [[0.5]])
    for name, y in shares.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        q25, q75 = np.percentile(y, [25, 75], axis=1)
        ax.fill_between(x, y.min(axis=1), y.max(axis=1), color=COLORS[name], alpha=0.15, label="Min - max")
        ax.fill_between(x, q25, q75, color=COLORS[name], alpha=0.35, label="Interquartile range (25%-75%)")
        ax.plot(x, y.mean(axis=1), color=COLORS[name], marker="o", ms=3, label="Mean")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set(xlabel="Input size n", ylabel="Gain / total")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _finish(fig, f"range_vs_n_{name}", save_dir, show)


# GenAI
def _boxplot_gain(res: dict, i: int, save_dir=None, show: bool = True) -> None:
    """Box plot of gain / total of the four match-ups for size index i."""
    n = int(res["sizes"][i])
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([res["gain"][s][i] / res["total"][i] for s in STRATEGIES], patch_artist=True,
                    tick_labels=[_label(s) for s in STRATEGIES])
    for patch, name in zip(bp["boxes"], STRATEGIES):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.5)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set(ylabel="Gain / total")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, f"boxplot_gain_n{n}", save_dir, show)


# GenAI
def _boxplot_loss(res: dict, i: int, save_dir=None, show: bool = True) -> None:
    """Box plot of the relative loss (%) of greedy compared to DP for size index i."""
    n = int(res["sizes"][i])
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([_relative_loss(res, d, g)[i] for d, g, _ in LOSS_PAIRS], patch_artist=True,
                    tick_labels=[title for _, _, title in LOSS_PAIRS])
    for patch, (_, greedy_name, _) in zip(bp["boxes"], LOSS_PAIRS):
        patch.set_facecolor(COLORS[greedy_name]); patch.set_alpha(0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set(ylabel="Relative loss of greedy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, f"boxplot_relative_loss_n{n}", save_dir, show)


# GenAI
def plot_boxplots(res: dict, n_values=None, save_dir=None, show: bool = True) -> None:
    """One box plot figure (gain / total) per selected size (default: smallest, middle, largest)."""
    sizes = list(res["sizes"])
    if n_values is None:
        n_values = sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
    for n in n_values:
        _boxplot_gain(res, sizes.index(n), save_dir, show)


# GenAI
def plot_boxplot_large_n(res: dict, n: int = 100, save_dir=None, show: bool = True) -> None:
    """Two figures for one large n (the size in res closest to n): box plots of the gain /
    total of the four match-ups and of the relative loss (%) of greedy compared to DP.
    Shows the variation per game."""
    i = int(np.argmin(np.abs(np.asarray(res["sizes"]) - n)))
    _boxplot_gain(res, i, save_dir, show)
    _boxplot_loss(res, i, save_dir, show)


# GenAI
def plot_win_probability(res: dict, save_dir=None, show: bool = True) -> None:
    """Fraction of trials in which player 1 gets more than half of the total."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in res["gain"].items():
        wins = (2 * g > res["total"][:, None]).mean(axis=1)
        ax.plot(res["sizes"], wins, marker="o", ms=4, color=COLORS[name], label=_label(name))
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="Input size n", ylabel="Probability player 1 wins", ylim=(-0.05, 1.05))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _finish(fig, "win_probability_vs_n", save_dir, show)


# GenAI
def plot_greedy_vs_dp(res: dict, save_dir=None, show: bool = True) -> None:
    """Main plots: one figure per opponent (greedy, DP) with the mean gain / total of player 1
    using greedy versus DP. Band = 95% confidence interval of the mean."""
    x = res["sizes"]
    curves = {name: _mean_ci(res["gain"][name] / res["total"][:, None])
              for dp_name, greedy_name, _ in LOSS_PAIRS for name in (greedy_name, dp_name)}
    ylim = _common_ylim([m - h for m, h in curves.values()] +
                        [m + h for m, h in curves.values()] + [[0.5]])
    for dp_name, greedy_name, title in LOSS_PAIRS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, label in ((greedy_name, "Player 1: greedy"), (dp_name, "Player 1: DP")):
            mean, half = curves[name]
            ax.plot(x, mean, marker="o", ms=4, color=COLORS[name], label=label)
            ax.fill_between(x, mean - half, mean + half, color=COLORS[name], alpha=0.2)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set(xlabel="Input size n", ylabel="Mean gain / total")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _finish(fig, f"greedy_vs_dp_{_slug(title)}", save_dir, show)


# GenAI
def plot_relative_loss(res: dict, save_dir=None, show: bool = True) -> None:
    """Relative loss (%) of greedy compared to DP, per opponent, against n.
    Band = 95% confidence interval of the mean."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = res["sizes"]
    for dp_name, greedy_name, title in LOSS_PAIRS:
        mean, half = _mean_ci(_relative_loss(res, dp_name, greedy_name))
        ax.plot(x, mean, marker="o", ms=4, color=COLORS[greedy_name], label=title)
        ax.fill_between(x, mean - half, mean + half, color=COLORS[greedy_name], alpha=0.2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="Input size n", ylabel="Relative loss of greedy (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _finish(fig, "relative_loss_greedy_vs_n", save_dir, show)


# GenAI
def plot_all(res: dict, save_dir=None, show: bool = True, box_n: int = 100) -> None:
    plot_greedy_vs_dp(res, save_dir=save_dir, show=show)
    plot_relative_loss(res, save_dir=save_dir, show=show)
    plot_boxplot_large_n(res, n=box_n, save_dir=save_dir, show=show)
    plot_gain_share(res, save_dir=save_dir, show=show)
    plot_range(res, save_dir=save_dir, show=show)
    plot_boxplots(res, save_dir=save_dir, show=show)
    plot_win_probability(res, save_dir=save_dir, show=show)
    plot_mean_std(res, save_dir=save_dir, show=show)


# ------------------------  plots COMPARING arrangements  ---------------------------

# GenAI
def plot_arrangements_share(all_res: dict, save_dir=None, show: bool = True) -> None:
    """One figure per strategy; one line per arrangement: mean gain / total against n.
    All figures use the same y-range so that they stay comparable."""
    shares = {s: {arr: (res["gain"][s] / res["total"][:, None]).mean(axis=1)
                  for arr, res in all_res.items()} for s in STRATEGIES}
    ylim = _common_ylim([y for d in shares.values() for y in d.values()] + [[0.5]])
    for strategy, curves in shares.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for arr, y in curves.items():
            ax.plot(all_res[arr]["sizes"], y, marker="o", ms=3, color=ARRANGEMENT_COLORS[arr], label=_label(arr))
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set(xlabel="Input size n", ylabel="Mean gain / total")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left")
        _finish(fig, f"arrangements_share_{strategy}", save_dir, show)


# GenAI
def plot_arrangements_differences(all_res: dict, save_dir=None, show: bool = True) -> None:
    """One figure per strategy pair; one line per arrangement: mean difference (% of total).
    Shows where the choice of strategy matters (difference != 0)."""
    for a, b, _ in DIFFERENCES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for arr, res in all_res.items():
            y = ((res["gain"][a] - res["gain"][b]) / res["total"][:, None] * 100).mean(axis=1)
            ax.plot(res["sizes"], y, marker="o", ms=3, color=ARRANGEMENT_COLORS[arr], label=_label(arr))
        ax.axhline(0, color="k", lw=1)
        ax.set_xscale("log")
        ax.set(xlabel="Input size n", ylabel="Mean difference (% of total)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _finish(fig, f"arrangements_difference_{a}_minus_{b}", save_dir, show)


# GenAI
def plot_arrangements_relative_loss(all_res: dict, save_dir=None, show: bool = True) -> None:
    """One figure per opponent; one line per arrangement: mean relative loss (%) of greedy
    compared to DP against n. Both figures use the same y-range."""
    losses = {title: {arr: _relative_loss(res, dp_name, greedy_name).mean(axis=1)
                      for arr, res in all_res.items()} for dp_name, greedy_name, title in LOSS_PAIRS}
    ylim = _common_ylim([y for d in losses.values() for y in d.values()] + [[0]])
    for title, curves in losses.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for arr, y in curves.items():
            ax.plot(all_res[arr]["sizes"], y, marker="o", ms=3, color=ARRANGEMENT_COLORS[arr], label=_label(arr))
        ax.axhline(0, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set(xlabel="Input size n", ylabel="Mean relative loss of greedy (%)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")
        _finish(fig, f"arrangements_relative_loss_{_slug(title)}", save_dir, show)


# GenAI
def plot_noise_sweep(sweeps: dict, save_dir=None, show: bool = True) -> None:
    """One figure per sweep: gain / total of every strategy against the amount of disturbance of
    the alternating arrangement (number of random swaps = noise * n). Same y-range for all."""
    shares = {key: {name: g / sw["total"] for name, g in sw["gain"].items()} for key, sw in sweeps.items()}
    ylim = _common_ylim([y for d in shares.values() for y in d.values()] + [[0.5]])
    for key, sw in sweeps.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        x = sw["noise"]
        for name, y in shares[key].items():
            ax.plot(x, y.mean(axis=1), marker="o", ms=4, color=COLORS[name], label=_label(name))
            ax.fill_between(x, y.min(axis=1), y.max(axis=1), color=COLORS[name], alpha=0.12)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xscale("symlog", linthresh=x[x > 0].min())
        ax.set_xlim(left=0)
        ax.set_ylim(ylim)
        ax.set(xlabel="Random swaps / n   (0 = perfectly alternating)",
               ylabel="Gain / total   (mean, min-max band)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _finish(fig, f"noise_sweep_n{sw['n']}", save_dir, show)


####################################################################################
####################################  TESTS  #######################################
####################################################################################

# GenAI
def check_gains(g: dict, total: int) -> None:
    """Relations that must hold for every input (player 1's gain in each match-up)."""
    greedy_g, dp_g = g["greedy"], g["dynamic_programming"]
    dp_first, greedy_first = g["dp_first_vs_greedy"], g["greedy_first_vs_dp"]
    assert all(0 <= v <= total for v in g.values()), f"gain out of range: {g}"
    assert dp_first >= dp_g >= greedy_first, f"ordering violated: {g}"
    assert dp_first >= greedy_g >= greedy_first, f"ordering violated: {g}"


# GenAI
def test_consistency(A: list) -> None:
    check_gains({name: f(A) for name, f in STRATEGIES.items()}, sum(A))


# GenAI
def test_equivalence(max_n: int = 60, trials: int = 200, seed: int = 0) -> None:
    """Fast versions must equal the dict versions on even-sized lists, including ties."""
    max_even = max_n - max_n % 2
    if max_even < 4:
        raise ValueError("max_n must be >= 4")
    rng = random.Random(seed)
    for _ in range(trials):
        n = rng.randrange(4, max_even + 1, 2)
        for A in (_random(n, rng), [rng.randint(1, 5) for _ in range(n)]):
            assert dynamic_programming(A) == dynamic_programming_dict(A), A
            assert dp_first_vs_greedy(A) == dp_first_vs_greedy_dict(A), A


# GenAI
def test_arrangements(trials: int = 20, seed: int = 0) -> None:
    """Every arrangement is a permutation of 1..n; test only even input sizes."""
    rng = random.Random(seed)
    assert make_array(10, "sorted", rng) == list(range(1, 11))
    for name in ARRANGEMENTS:
        for n in (2, 10, 50):
            assert sorted(make_array(n, name, rng)) == list(range(1, n + 1)), (name, n)
    for n in (10, 50, 200):
        for _ in range(trials):
            A = _alternating(n, rng)
            g = {name: f(A) for name, f in STRATEGIES.items()}
            assert len(set(g.values())) == 1, f"strategies differ on alternating n={n}: {g}"
            assert g["greedy"] == sum(range(n // 2 + 1, n + 1))
    assert all(n % 2 == 0 for n in log_spaced_sizes(3, 51, 10))
    try:
        make_array(3, "random", rng)
    except ValueError:
        pass
    else:
        raise AssertionError("odd array sizes must be rejected")


def test_all() -> None:
    test_equivalence()
    test_arrangements()
    assert greedy_first_vs_dp([]) == 0
    assert greedy_first_vs_dp_dict([]) == 0
    print("all tests passed")


####################################################################################
#################################  EXPERIMENTS  ####################################
####################################################################################

# GenAI
def run_experiments(sizes: list, trials: int = 20, seed: int = 0, check: bool = True,
                    arrangement: str = "random") -> dict:
    """
    For every size n and every trial one array is generated with the given arrangement and
    ALL strategies are run on that same array (paired samples), so that differences between
    strategies are meaningful. Deterministic arrangements are run once (trials = 1).

    Returns {"arrangement", "sizes", "total", "gain"} where gain[name] is an array of
    shape (len(sizes), trials).
    """
    if not sizes or any(n < 2 or n % 2 for n in sizes):
        raise ValueError("sizes must contain only positive even values")
    if ARRANGEMENTS[arrangement][1]:
        trials = 1
    rng = random.Random(seed)
    gain = {name: np.zeros((len(sizes), trials)) for name in STRATEGIES}
    for i, n in enumerate(sizes):
        for t in range(trials):
            A = make_array(n, arrangement, rng)
            g = {name: f(A) for name, f in STRATEGIES.items()}
            if check:
                check_gains(g, n * (n + 1) // 2)
            for name, v in g.items():
                gain[name][i, t] = v
        print(f"[{arrangement}] n = {n:>6} done ({trials} trials)")
    return {"arrangement": arrangement, "sizes": np.array(sizes),
            "total": np.array([n * (n + 1) // 2 for n in sizes]), "gain": gain}


# GenAI
def run_all_arrangements(sizes: list, trials: int = 20, seed: int = 0, arrangements=None) -> dict:
    return {arr: run_experiments(sizes, trials, seed, arrangement=arr)
            for arr in (arrangements or ARRANGEMENTS)}


# GenAI
def run_noise_sweep(n: int, noise_levels: list, trials: int = 20, seed: int = 0) -> dict:
    """Fixed n; the alternating arrangement is disturbed by round(noise * n) random swaps."""
    _require_even_n(n)
    rng = random.Random(seed)
    gain = {name: np.zeros((len(noise_levels), trials)) for name in STRATEGIES}
    for i, noise in enumerate(noise_levels):
        generate = _alternating_noisy(noise)
        for t in range(trials):
            A = generate(n, rng)
            g = {name: f(A) for name, f in STRATEGIES.items()}
            check_gains(g, n * (n + 1) // 2)
            for name, v in g.items():
                gain[name][i, t] = v
        print(f"[noise sweep n = {n}] noise = {noise} done")
    return {"n": n, "noise": np.array(noise_levels, dtype=float),
            "total": n * (n + 1) // 2, "gain": gain}


# GenAI
def run_full_analysis(n_min: int = 4, n_max: int = 5000, runs: int = 30,
                      nb_points: int = 10, seed: int = 0, show: bool = False,
                      folder: str = "Graphs", box_n: int = 100) -> None:
    """Run tests and the arrangement experiments."""
    folder = os.path.join(folder, f"N{n_max}RUNS{runs}")
    test_all()

    sizes = log_spaced_sizes(n_min, n_max, nb_points)
    all_res = run_all_arrangements(sizes, runs, seed)
    skip_individual_plots = {
        "alternating", "alternating_1pct", "alternating_5pct",
        "alternating_20pct", "sorted",
    }
    for arr, res in all_res.items():
        if arr not in skip_individual_plots:
            plot_all(res, save_dir=os.path.join(folder, arr), show=show, box_n=box_n)
    plot_arrangements_share(all_res, save_dir=folder, show=show)
    plot_arrangements_differences(all_res, save_dir=folder, show=show)
    plot_arrangements_relative_loss(all_res, save_dir=folder, show=show)


if __name__ == "__main__":
    for size in [10_000]:
        for run in [1_000]:
            run_full_analysis(n_min=4, n_max=size, runs=run, nb_points=10,
                            seed=0, show=False, folder="Graphs")