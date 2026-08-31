"""Tests for the local blind-listening helper."""

import csv
from pathlib import Path
import tempfile
import unittest

from review_blind import load_rows, next_unanswered, save_answer
from report_blind_review import summarize


class BlindReviewTests(unittest.TestCase):
    def make_sheet(self, path):
        fields = ("query_id", "label", "track_id", "artist", "title", "audio_file",
                  "auditor_hears_label", "auditor_reason")
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
            for index in range(40):
                writer.writerow({"query_id": index + 1, "label": "rock",
                                 "track_id": f"track_{index}", "artist": "", "title": "",
                                 "audio_file": f"audio/{index}.wav", "auditor_hears_label": "",
                                 "auditor_reason": ""})

    def test_answer_is_saved_and_review_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"; self.make_sheet(path)
            save_answer("1", "yes", path=path)
            rows = load_rows(path)
            self.assertEqual(rows[0]["auditor_hears_label"], "yes")
            self.assertTrue(rows[0]["auditor_reason"])
            self.assertEqual(next_unanswered(rows)["query_id"], "2")

    def test_invalid_decision_does_not_change_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"; self.make_sheet(path)
            before = path.read_bytes()
            with self.assertRaises(ValueError): save_answer("1", "maybe", path=path)
            self.assertEqual(path.read_bytes(), before)

    def test_audit_summary_excludes_uncertain_from_human_agreement(self):
        rows = [
            {"human_decision": "yes", "human_target": 1, "source_target": 1,
             "model_prediction": 1},
            {"human_decision": "no", "human_target": 0, "source_target": 1,
             "model_prediction": 0},
            {"human_decision": "uncertain", "human_target": None, "source_target": 0,
             "model_prediction": 1},
        ]
        result = summarize(rows)
        self.assertEqual(result["decisive_queries"], 2)
        self.assertEqual(result["source_human_agreement_count"], 1)
        self.assertEqual(result["model_human_agreement_count"], 2)
        self.assertEqual(result["model_human_precision_decisive"], 1.0)


if __name__ == "__main__":
    unittest.main()
