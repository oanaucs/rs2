import math
import numpy as np


def f(p, q, s):
    """Rotiert einen Punkt s um den Punkt p, sodass p und q durch die Rotation dieselbe x-Koordinate erhalten."""

    alpha = math.atan2(q[1] - p[1], q[0] - p[0])
    R = np.matrix([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    f_s = np.matmul(R, (s - p) + p)
    return f_s
