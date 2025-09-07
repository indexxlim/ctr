# Data: logs = [{x, a, p_b, r}, ...]

# IPS estimator for policy pi:
def IPS_estimate(logs, pi):
    # pi(a|x) returns probability under eval policy
    total = 0.0
    N = len(logs)
    for entry in logs:
        p_eval = pi.prob(entry.x, entry.a)
        w = p_eval / entry.p_b
        total += w * entry.r
    return total / N

# Self-normalized IPS
def SNIPS(logs, pi):
    num = 0.0; denom = 0.0; N=len(logs)
    for e in logs:
        w = pi.prob(e.x, e.a) / e.p_b
        num += w * e.r
        denom += w
    return num / denom if denom>0 else 0

# Doubly Robust
# Requires a value model q_hat(x,a) approximating E[r | x,a]
def DR_estimate(logs, pi, q_hat):
    total = 0.0; N=len(logs)
    for e in logs:
        p_eval = pi.prob(e.x, e.a)
        w = p_eval / e.p_b
        total += q_hat.predict(e.x, pi) + w * (e.r - q_hat.predict(e.x, e.a))
    return total / N

# q_hat.predict(x, a) can be:
#   - a regression model trained on logs: minimize (r - q(x,a))^2 with importance weighting
#   - or a direct model for expected reward per (x,a)

# Practical pipeline:
# 1) Fit q_hat on logs (regularized)
# 2) Compute IPS, SNIPS, DR for candidate policies
# 3) Provide confidence intervals (bootstrap)
# 4) If discrepancy between IPS and DR large -> flag insufficient support / distribution shift

