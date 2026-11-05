import numpy as np
import typing as T


def sigmoid(x:T.Union[float, int], L0=0.0, c=0.1) -> float:
    z = c * (L0 - x)
    # Clip to prevent overflow
    z = np.clip(z, -709, 709)  # e^500 is near max float, e^-500 is near 0
    return 1 / (1 + np.exp(z))

def exponential_ramp(x, x_zero, x_one, k=4.0):
    """Exponential transition from y=0 at x_zero to y=1 at x_one.

    Accepts a scalar or NumPy array.
    """
    if x_zero == x_one:
        raise ValueError("x_zero and x_one must be different.")

    x = np.asarray(x, dtype=float)
    t = np.clip((x - x_zero) / (x_one - x_zero), 0.0, 1.0)

    if k == 0:
        return t

    # Equivalent exponential formulas, arranged to avoid overflow.
    if k > 0:
        return (
            np.exp(k * (t - 1))
            * np.expm1(-k * t)
            / np.expm1(-k)
        )

    return np.expm1(k * t) / np.expm1(k)

def flatbottom(x, a, b, c, d, k_left=6.0, k_right=6.0):
    """Two exponential ramps with a zero interval [b, c]."""
    if not a < b <= c < d:
        raise ValueError("Breakpoints must satisfy a < b <= c < d.")

    left = exponential_ramp(
        x, x_zero=b, x_one=a, k=k_left
    )
    right = exponential_ramp(
        x, x_zero=c, x_one=d, k=k_right
    )

    return left + right
