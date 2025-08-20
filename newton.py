import pandas as pd

def newton_method(f, x0, tol=1e-6, max_iter=100, h=1e-5):
    x = x0
    log = [(x, f(x))]
    for i in range(max_iter):
        f_deriv = (f(x + h) - f(x)) / (h)
        f_second_deriv = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
        
        if abs(f_second_deriv) < 1e-10:
            break
            
        x_new = x - f_deriv / f_second_deriv
        
        if abs(x_new - x) < tol:
            break
            
        x = x_new
        log.append((x, f(x)))
    
    return x, log