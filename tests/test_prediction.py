import unittest

import numpy as np
import pandas as pd

from latentra.prediction import budget_recall, metrics


class PredictionTests(unittest.TestCase):
    def test_budget_and_auc_use_correct_score_direction_and_ids(self):
        risk = pd.Series(np.arange(20) / 100, index=range(100, 120))
        y = (risk >= 0.16).iloc[::-1]
        out = metrics(risk, y)
        self.assertEqual((out["tn"], out["fp"], out["fn"], out["tp"]), (16, 0, 2, 2))
        self.assertEqual(out["roc_auc"], 1)
        self.assertEqual(out["recall"], .5)

        class FixedModel:
            def predict_proba(self, x):
                return np.column_stack([1 - x["risk"], x["risk"]])

        self.assertEqual(budget_recall(FixedModel(), risk.to_frame("risk"), y), out["recall"])
