import unittest

import pandas as pd

from latentra import config as cfg
from latentra.baseline import compare_rules, review_scores, select_for_review
from latentra.data import prediction_inputs


class BaselineTests(unittest.TestCase):
    def test_inputs_exclude_answer_and_id(self):
        rows = pd.DataFrame(0, index=pd.Index([34, 72], name="id"), columns=cfg.MEASUREMENTS)
        rows[cfg.TARGET] = [16, 17]
        rows["copy_of_target"] = rows[cfg.TARGET]
        x, y = prediction_inputs(rows)
        self.assertEqual(set(x), set(cfg.MEASUREMENTS) - {cfg.TARGET})
        self.assertTrue(x.index.equals(rows.index))
        self.assertEqual(y.tolist(), [False, True])

    def test_boundary_ties_preserve_budget_and_selection(self):
        scores = pd.Series([0, *[1] * 18, 2], index=range(30, 50))
        ids = select_for_review(scores)
        self.assertEqual(len(ids), 2)
        self.assertIn(30, ids)
        self.assertNotIn(49, ids)
        self.assertTrue(ids.equals(select_for_review(scores.iloc[::-1])))

    def test_percentiles_use_training_reference_with_ties(self):
        train = pd.DataFrame({"f_08": [1, 2, 2, 4], "f_12": [2, 2, 3, 3]})
        val = pd.DataFrame({"f_08": [2, 0], "f_12": [2, 4]}, index=[10, 11])
        self.assertEqual(review_scores(train, val)["mean_percentile"].tolist(), [.625, .5])

    def test_metrics_use_label_ids_not_positions(self):
        scores = pd.DataFrame({"rule": range(20)})
        high = pd.Series([i in (0, 4, 5, 6) for i in range(20)]).iloc[::-1]
        result = compare_rules(scores, high).iloc[0]
        self.assertEqual(result["caught"], 1)
        self.assertEqual(result["recall"], .25)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["lift"], 2.5)
