from enum import Enum

import numpy as np
from .std import PairwiseVariance


def p_bernstein(eps, n1, n2, sigma1, sigma2, abs_max: float = 0.1):
    k = n2 / (n1 + n2)

    def exponent(eps, n, sigma, k, M):
        a = sigma ** 2
        b = (M * k * eps) / 3
        a += 1e-10  # add some small positive value to avoid dividing by 0 in some rare cases.
        b += 1e-10
        return -(n * (k * eps) ** 2) / (2 * (a + b))

    e1 = exponent(eps, n1, sigma1, k, abs_max)
    e2 = exponent(eps, n2, sigma2, 1 - k, abs_max)
    if np.any(np.isnan(e1)) or np.any(np.isnan(e2)):
        print("nan value in exponent")
    res = 2 * (np.exp(e1) + np.exp(e2))
    return res


class SplitType(Enum):
    All = 0
    Equidistant = 1


class AdaptiveWindow:
    def __init__(self,
                 delta: float,
                 delta_warning: float,
                 max_size: int = np.infty,
                 split_type: SplitType = SplitType.Equidistant,
                 bonferroni: bool = False,
                 n_splits: int = 20):
        """
        The data structure containing the aggregates
        :param delta: The allowed rate of false alarms
        :param max_size: The maximum size of the _window
        :param split_type: equidistant (ed) or every possible points (all)
        :param bonferroni: Flag if Bonferroni correction should be used
        :param n_splits: Number of evaluated time points
        """
        self.w = []
        self.delta = delta
        self._delta_warning = delta_warning
        self.bonferroni = bonferroni
        self.t_star = 0
        self.n_seen_items = 0
        self._best_split_candidate = 0
        self.min_p_value = 1.0
        self._argmin_p_value = 0
        self.n_splits = n_splits
        self.variance_tracker = PairwiseVariance(max_size=max_size)
        self.max_size = max_size
        self.split_type: SplitType = split_type
        self.min_window_size = 60
        self.logger = None
        self._cut_indices = []

    def __len__(self):
        return len(self.w)

    def grow(self, new_item):
        """
        Grows the adaptive _window by one instance
        :param new_item: Tuple (loss, reconstruction, original)
        :return: nothing
        """
        loss, data = new_item[0], (new_item[1], new_item[2])
        self.w.append(data)
        self.variance_tracker.update(loss)
        self.n_seen_items += 1
        if len(self.w) > self.max_size:
            self.w = self.w[-self.max_size:]
        self._update_cut_indices()

    def has_change(self):
        """
        Performs change and warning detection.
        Result can be obtained from t_star
        :return: (has_change, has_warning, change/warning point)
        """
        return self._bernstein_cd()

    def reset(self):
        """
        Drop all data up to the last change point
        :return:
        """
        self.w = []
        self.variance_tracker.reset()
        self._argmin_p_value = False
        self.min_p_value = 1.0
        self._update_cut_indices()

    def data(self):
        """
        :return: All observations in the _window
        """
        return np.array([item[-1] for item in self.w])

    def reconstructions(self):
        return np.array([item[0] for item in self.w])

    def data_new(self):
        return np.array([item[-1] for item in self.w[self._cut_index(offset=1):]])

    def _bernstein_cd(self):
        """
        Change detection using the Bernstein method
        :return: change detected, change point
        """
        if len(self.variance_tracker) <= self.min_window_size:
            return False, False, None
        aggregates = [self.variance_tracker.pairwise_aggregate(i) for i in self._cut_indices]
        info = np.array([[aggregate.mean(), aggregate.std(), aggregate.n()] for aggregate in aggregates])
        pairwise_means, sigma, pairwise_n = info[:, 0], info[:, 1], info[:, 2]
        pairwise_n = np.array([aggregate.n() for aggregate in aggregates])
        epsilon = np.array([np.abs(m2 - m1) for (m1, m2) in pairwise_means])
        delta_empirical = p_bernstein(eps=epsilon,
                                      n1=pairwise_n[:, 0], n2=pairwise_n[:, 1],
                                      sigma1=sigma[:, 0], sigma2=sigma[:, 1])
        self.min_p_value = np.min(delta_empirical)
        self._argmin_p_value = np.argmin(delta_empirical)
        d = self._delta_bonferroni(self.delta) if self.bonferroni else self.delta
        d_warning = self._delta_bonferroni(self._delta_warning) if self.bonferroni else self._delta_warning
        has_change = self.min_p_value < d
        has_warning = self.min_p_value < d_warning
        self.t_star = self._cut_index()
        if has_change:
            return has_change, has_warning, self.n_seen_items
        else:
            return False, False, None

    def most_recent_loss(self):
        p_aggregate = self.variance_tracker.pairwise_aggregate(len(self.variance_tracker.aggregates) - 1)
        _, loss = p_aggregate.mean()
        return loss

    def _update_cut_indices(self):
        if len(self.variance_tracker) <= self.min_window_size:
            return
        k_min = int(self.min_window_size / 2)
        k_max = len(self.variance_tracker) - k_min
        if self.split_type == SplitType.Equidistant:
            interval = k_max - k_min
            n_points = self.n_splits
            if interval < n_points:
                self._cut_indices = np.arange(k_min, k_max + 1)
            else:
                dist = int(interval / n_points)
                cut_indices = (k_min + k_max) - np.arange(k_min, k_max + 1, dist)[:n_points]
                self._cut_indices = cut_indices
        elif self.split_type == SplitType.All:
            self._cut_indices = list(range(k_min, k_max, 1))
        else:
            raise ValueError

    def _cut_index(self, offset=0):
        index_out_of_bounds = self._argmin_p_value + offset < 0
        index_out_of_bounds = index_out_of_bounds or self._argmin_p_value + offset >= len(self._cut_indices)
        if index_out_of_bounds:
            offset = 0
        return self._cut_indices[self._argmin_p_value + offset]

    def _delta_bonferroni(self, d: float):
        return d / len(self._cut_indices)
