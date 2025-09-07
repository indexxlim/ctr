# COMPONENTS (summary)
# - FeatureStore (online + batch)
# - ModelServing (lightweight inference service)
# - DecisionEngine (bidding/policy)
# - OfflineTrain (training jobs, experiments)
# - OPEService (policy evaluation)
# - ExperimentMgr (A/B, rollout)
# - Monitoring (drift, calibration, KPI)

# API: /score endpoint (for real-time auction)
POST /score
Request JSON:
{
  "impression_id": str,
  "user_id": str,
  "placement_id": str,
  "creative_candidates": [creative_id...],
  "context_features": { ... },   # time, device, location etc.
  "budget_token": {campaign_id, budget_left}
}

Response JSON:
{
  "decisions": [
    { "creative_id": id,
      "bid": float,
      "pred_ctr": float,
      "pred_cvr": float,
      "pred_ctcvr": float,
      "win_prob": float,
      "expected_value": float
    }, ...
  ],
  "chosen": { "creative_id": id, "bid": float }
}

# Real-time flow (pseudo)
def handle_request(req):
    x_context = req.context_features
    user_online_feats = FeatureStore.get_online_features(req.user_id)
    candidates = req.creative_candidates
    # assemble batched inputs
    batch_inputs = assemble_inputs(user_online_feats, x_context, candidates)
    preds = ModelServing.batch_infer(batch_inputs)  # returns ctr,cvr,ctcvr,uncertainty
    bids = DecisionEngine.compute_bids(preds, req.budget_token, auction_meta)
    chosen = DecisionEngine.select_bid(bids, auction_meta)
    log_request(req, batch_inputs, preds, bids, chosen)  # for OPE & training
    return make_response(bids, chosen)
<Paste>
