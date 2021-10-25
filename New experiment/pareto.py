import numpy as np

class BoundedVolumes:
    @classmethod
    def empty(cls, dim):
        """
        Returns an empty bounded volume (hypercube).
        :param dim: dimension of the volume
        :param dtype: dtype of the coordinates
        :return: an empty :class:`.BoundedVolumes`
        """
        setup_arr = np.zeros((0, dim), dtype=np.int32)
        return cls(setup_arr.copy(), setup_arr.copy())

    def __init__(self, lb, ub):
        """
        Construct bounded volumes.
        :param lb: the lowerbounds of the volumes
        :param ub: the upperbounds of the volumes
        """
        super(BoundedVolumes, self).__init__()
        assert np.all(lb.shape == ub.shape)
        self.lb = np.atleast_2d(lb)
        self.ub = np.atleast_2d(ub)

    def append(self, lb, ub):
        """
        Add new bounded volumes.
        :param lb: the lowerbounds of the volumes
        :param ub: the upperbounds of the volumes
        """
        self.lb = np.vstack((self.lb, lb))
        self.ub = np.vstack((self.ub, ub))

    def clear(self):
        """
        Clears all stored bounded volumes
        """
        outdim = self.lb.shape[1]
        self.lb = np.zeros((0, outdim))
        self.ub = np.zeros((0, outdim))

    def size(self):
        """
        :return: volume of each bounded volume
        """
        return np.prod(self.ub - self.lb, axis=1)

def non_dominated_sort(objectives):
    """
    Computes the non-dominated set for a set of data points
    :param objectives: data points
    :return: tuple of the non-dominated set and the degree of dominance,
        dominances gives the number of dominating points for each data point
    """
    extended = np.tile(objectives, (objectives.shape[0], 1, 1))
    dominance = np.sum(np.logical_and(np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
                                      np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)

    return objectives[dominance == 0], dominance

class Pareto:
    def __init__(self, Y, threshold=0):
        """
        Construct a Pareto set.
        Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
        The latter is needed for certain multiobjective acquisition functions.
        E.g., the :class:`~.acquisition.HVProbabilityOfImprovement`.
        :param Y: output data points, size N x R
        :param threshold: approximation threshold for the generic divide and conquer strategy
            (default 0: exact calculation)
        """
        self.threshold = threshold
        self.Y = Y

        # Setup data structures
        self.bounds = BoundedVolumes.empty(Y.shape[1])
        self.front = np.zeros((0, Y.shape[1]))

        # Initialize
        self.update()

    @staticmethod
    def _is_test_required(smaller):
        """
        Tests if a point augments or dominates the Pareto set.
        :param smaller: a boolean ndarray storing test point < Pareto front
        :return: True if the test point dominates or augments the Pareto front (boolean)
        """
        # if and only if the test point is at least in one dimension smaller for every point in the Pareto set
        idx_dom_augm = np.any(smaller, axis=1)
        is_dom_augm = np.all(idx_dom_augm)

        return is_dom_augm

    def _update_front(self):
        """
        Calculate the non-dominated set of points based on the latest data.
        The stored Pareto set is sorted on the first objective in ascending order.
        :return: boolean, whether the Pareto set has actually changed since the last iteration
        """
        current = self.front
        pf, _ = non_dominated_sort(self.Y)

        self.front = pf[pf[:, 0].argsort(), :]

        return not np.array_equal(current, self.front)

    def update(self, Y=None, generic_strategy=False):
        """
        Update with new output data.
        Computes the Pareto set and if it has changed recalculates the cell bounds covering the non-dominated region.
        For the latter, a direct algorithm is used for two objectives, otherwise a
        generic divide and conquer strategy is employed.
        :param Y: output data points
        :param generic_strategy: Force the generic divide and conquer strategy regardless of the number of objectives
            (default False)
        """
        self.Y = Y if Y is not None else self.Y

        # Find (new) set of non-dominated points
        changed = self._update_front()

        # Recompute cell bounds if required
        # Note: if the Pareto set is based on model predictions it will almost always change in between optimizations
        if changed:
            # Clear data container
            self.bounds.clear()
            if generic_strategy:
                self.divide_conquer_nd()
            else:
                self.bounds_2d() if self.Y.shape[1] == 2 else self.divide_conquer_nd()

    def divide_conquer_nd(self):
        """
        Divide and conquer strategy to compute the cells covering the non-dominated region.
        Generic version: works for an arbitrary number of objectives.
        """
        outdim = self.Y.shape[1]

        # The divide and conquer algorithm operates on a pseudo Pareto set
        # that is a mapping of the real Pareto set to discrete values
        pseudo_pf = np.argsort(self.front, axis=0) + 1  # +1 as index zero is reserved for the ideal point

        # Extend front with the ideal and anti-ideal point
        min_pf = np.min(self.front, axis=0) - 1
        max_pf = np.max(self.front, axis=0) + 1

        pf_ext = np.vstack((min_pf, self.front, max_pf))  # Needed for early stopping check (threshold)
        pf_ext_idx = np.vstack((np.zeros(outdim, dtype=np.int32), pseudo_pf, np.ones(outdim, dtype=np.int32) * self.front.shape[0] + 1))

        # Start with one cell covering the whole front
        dc = [(np.zeros(outdim, dtype=np.int32),
               (int(pf_ext_idx.shape[0]) - 1) * np.ones(outdim, dtype=np.int32))]
        total_size = np.prod(max_pf - min_pf)

        # Start divide and conquer until we processed all cells
        while dc:
            # Process test cell
            cell = dc.pop()

            # Acceptance test:
            if self._is_test_required((cell[1] - 0.5) < pseudo_pf):
                # Cell is a valid integral bound: store
                self.bounds.append(pf_ext_idx[cell[0], np.arange(outdim)],
                                   pf_ext_idx[cell[1], np.arange(outdim)])
            # Reject test:
            elif self._is_test_required((cell[0] + 0.5) < pseudo_pf):
                # Cell can not be discarded: calculate the size of the cell
                dc_dist = cell[1] - cell[0]
                hc = BoundedVolumes(pf_ext[pf_ext_idx[cell[0], np.arange(outdim)], np.arange(outdim)],
                                    pf_ext[pf_ext_idx[cell[1], np.arange(outdim)], np.arange(outdim)])

                # Only divide when it is not an unit cell and the volume is above the approx. threshold
                if np.any(dc_dist > 1) and np.all((hc.size()[0] / total_size) > self.threshold):
                    # Divide the test cell over its largest dimension
                    edge_size, idx = np.max(dc_dist), np.argmax(dc_dist)
                    edge_size1 = int(np.round(edge_size / 2.0))
                    edge_size2 = edge_size - edge_size1

                    # Store divided cells
                    ub = np.copy(cell[1])
                    ub[idx] -= edge_size1
                    dc.append((np.copy(cell[0]), ub))

                    lb = np.copy(cell[0])
                    lb[idx] += edge_size2
                    dc.append((lb, np.copy(cell[1])))
            # else: cell can be discarded

    def bounds_2d(self):
        """
        Computes the cells covering the non-dominated region for the specific case of only two objectives.
        Assumes the Pareto set has been sorted in ascending order on the first objective.
        This implies the second objective is sorted in descending order.
        """
        outdim = self.Y.shape[1]
        assert outdim == 2

        pf_idx = np.argsort(self.front, axis=0)
        pf_ext_idx = np.vstack((np.zeros(outdim, dtype=np.int32), pf_idx + 1,
                                np.ones(outdim, dtype=np.int32) * self.front.shape[0] + 1))

        for i in range(pf_ext_idx[-1, 0]):
            self.bounds.append((i, 0), (i+1, pf_ext_idx[-i-1, 1]))