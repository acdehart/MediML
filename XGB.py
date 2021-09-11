from xgboost import XGBClassifier


def get_classifier():
    return XGBClassifier(
        n_estimators=120,  # 200
        # min_sample_split = 30,
        max_depth=4,  # 4, higher values might learn too-specific examples
        # max_leaf_nodes=4,  # replaces max depth
        min_child_weight=3,  # 3, usually 1 but higher values eliminate model-specific trees
        early_stopping_rounds=10,  # Stop if loss doesn't improve in N rounds
        # eval_metric='aucpr',
        eval_metric='aucpr',
        gamma=0.1,  # 0.1 determines minimum gain for split
        reg_lambda=1,  # L2 regularization
        reg_alpha=0,
        max_delta_step=0,  # helps with unbalanced classes
        silent=1,
        booster='gbtree',
        # eta=0.3,  # Old parameter for learning rate
        scale_pos_weight=1,  # 1
        subsample=0.85,  # 0.85 Reduces the number of rows to learn from
        colsample_bytree=0.85,  # 0.85 Reduces features the tree is allowed to train on
        learning_rate=0.05,  # 0.1
        # warm_start=True,
        verbose=1,
        max_features='sqrt'
    )
