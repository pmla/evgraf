import numpy as np
import scipy
import scipy.optimize
from distutils.version import LooseVersion

# Scipy has a fast linear_sum_assignment since version 1.4.0.
# If an older version of scipy is loaded, look for a faster external module
# first.  As a last resort, load the slow, older scipy version


def hungarian_assignment(cost):

    n = len(cost)
    # hungarian.lap overwrites the cost matrix so we have to pass a copy
    answers = hungarian.lap(np.copy(cost))
    return (np.arange(n), answers[0])


if LooseVersion(scipy.__version__) >= '1.4.0':
    linear_sum_assignment = scipy.optimize.linear_sum_assignment
else:
    try:
        import hungarian
        linear_sum_assignment = hungarian_assignment
    except ImportError:
        linear_sum_assignment = scipy.optimize.linear_sum_assignment
