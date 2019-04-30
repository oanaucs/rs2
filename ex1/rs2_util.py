import numpy as np


def vec2d(x, y=None):
    """Liefert einen zweidimensionalen Vektor (x,y) als numpy-Array zurück."""
    if y is not None:
        return np.array([x, y], dtype=np.float32)
    else:
        return np.array(x, dtype=np.float32)


def vec3d(x, y=None, z=None):
    """Liefert einen dreidimensionalen Vektor (x,y,z) als numpy-Array zurück."""
    if y is None and z is None:
        return np.array(x, dtype=np.float32)
    elif y is not None and z is not None:
        return np.array([x, y, z], dtype=np.float32)
    else:
        raise ValueError('Unsupported configuration')


def to_homogenous(x):
    """Überführt einen Vektor in ein homogenes Koordinatensystem."""
    x_ = np.concatenate((x, np.array([1], dtype=np.float32)), axis=0)
    return x_


def from_homogenous(x):
    """Rücktransformation aus homogenen Koordinaten."""
    x_ = x[:-1] / x[-1]
    return x_