def newton_method(f, x0, tol=1e-6, max_iter=100, h=1e-5):
    """
    Find a stationary point of a function using Newton’s method with 
    finite-difference derivatives.
    Approximates f'(x) and f''(x) numerically and iteratively updates 
    x until convergence or the maximum iteration count is reached.

    Parameters
    ----------
    f : callable
        Function to optimize, taking a float and returning a float.
    x0 : float
        Initial guess.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    h : float, optional
        Step size for finite differences (default 1e-5).

    Returns
    -------
    x : float
        Estimated stationary point.
    log : list of (float, float)
        Iteration history as (x, f(x)).
    """
    x = x0
    log = [(x, f(x))]
    for i in range(max_iter):
        f_deriv = (f(x + h) - f(x)) / (h)
        f_second_deriv = (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

        if abs(f_second_deriv) < 1e-10:
            break

        x_new = x - f_deriv / f_second_deriv

        if abs(x_new - x) < tol:
            break

        x = x_new
        log.append((x, f(x)))

    return x, log