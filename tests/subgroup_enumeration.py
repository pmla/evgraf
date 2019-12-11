import itertools
import numpy as np
from evgraf import subgroup_enumeration


# Test subgroup elements
for dim in [1, 2, 3]:
    for dims in itertools.product(range(8), repeat=dim):
        dims = np.array(dims)
        bases = list(subgroup_enumeration.enumerate_subgroup_bases(dims))
        assert len(bases) == subgroup_enumeration.count_subgroups(dims)

        for H in bases:
            elements = subgroup_enumeration.get_subgroup_elements(dims, H)
            assert len(set([tuple(e) for e in elements])) == len(elements)
            # when numpy dependency is at least 1.13 we can write:
            # assert elements.shape == np.unique(elements, axis=0).shape
            group_index = np.prod(dims // np.diag(H))
            assert group_index == len(elements)
