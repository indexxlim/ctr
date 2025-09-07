# Models
# - For each arm a: posterior over theta_a ~ N(mu_a, Sigma_a)
# - Win-rate model w(b|x): prob of winning given bid b and context x (learned offline)
# - Cost model c(b|x): expected cost if win (could be second-price approx)

# initialize
for arm in arms:
    mu[arm] = zeros(d); Sigma[arm] = I * prior_var

lambda = lambda0   # Lagrangian multiplier for budget constraint
B = total_budget; budget_spent = 0
T = total_impressions_target

# per-impression loop
for t, impression in enumerate(stream):
    x = impression.context
    candidates = impression.candidates  # list of arms

    # Thompson Sampling: sample theta for each arm
    sampled_theta = {a: mvn_sample(mu[a], Sigma[a]) for a in candidates}

    # estimate per-arm expected reward and optimal bid
    arm_scores = {}
    for a in candidates:
        phi = phi_func(x, a)                           # feature vector
        p_conv_sample = sigmoid( phi.dot(sampled_theta[a]) )  # approx p(conv)
        value = campaign_value_for_conversion(impression.campaign)
        # expected_profit before cost: p_conv * value
        # incorporate Lagrangian penalty for cost: obj = p_conv * value - lambda * expected_cost
        # but expected_cost depends on bid; we search a small grid of bid levels
        best_bid = None; best_obj = -inf
        for bid in bid_grid:
            win_prob = win_rate_model.prob_win(bid, x, a)
            exp_cost = win_prob * cost_estimate(bid, x)  # expected payment
            exp_conv = win_prob * p_conv_sample
            obj = exp_conv * value - lambda * exp_cost
            if obj > best_obj:
                best_obj = obj; best_bid = bid
        arm_scores[a] = (best_obj, best_bid, p_conv_sample)

    # pick arm with max obj
    chosen_arm = argmax_a arm_scores[a].best_obj
    bid = arm_scores[chosen_arm].best_bid

    # send bid to auction; observe win, cost, reward (click/conv)
    result = auction_simulator(bid, chosen_arm, impression)  # or real auction
    # log for training/OPE
    log_event(impression.id, chosen_arm, bid, prob_under_policy=..., result)

    # update posterior for chosen arm with observed reward (Bernoulli conv)
    # Use Laplace / Bayesian linear regression update (approx)
    phi_chosen = phi_func(x, chosen_arm)
    # Online Bayesian update (ridge-like)
    Sigma_inv = inv(Sigma[chosen_arm])
    Sigma_inv_new = Sigma_inv + phi_chosen[:,None] @ phi_chosen[None,:] * result.reward
    Sigma[chosen_arm] = inv(Sigma_inv_new)
    mu[chosen_arm] = Sigma[chosen_arm] @ (Sigma_inv @ mu[chosen_arm] + phi_chosen * result.reward)

    # budget accounting
    budget_spent += result.cost
    # update lambda (dual ascent) to respect budget over horizon
    # simple rule: lambda <- lambda + eta * (budget_spent - (B * (t/T)))
    lambda = max(0, lambda + eta * (budget_spent - B * (t / T)))

    # periodic re-train win_rate_model / cost_model using logged auctions

# Notes:
# - using per-arm posteriors can be expensive if many arms; use factorized or shared-parameter model
# - alternatively use neural-linear TS (shared NN -> last-layer Bayesian linear)
