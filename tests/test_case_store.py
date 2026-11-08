import tempfile
import unittest
from pathlib import Path

from retrieval_lab.case_store import ResearchCaseStore
from retrieval_lab.serving import build_app
from fastapi.testclient import TestClient


class ResearchCaseStoreTests(unittest.TestCase):
    def test_frozen_verifier_settings_return_validation_error_without_saving_case(self):
        class FrozenRag:
            def verify_claim(self, claim, *, top_k, max_retries):
                raise ValueError('Retrieval settings must match the frozen calibration')
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(build_app(FrozenRag(), case_db_path=Path(directory)/'cases.sqlite3')) as client:
                for endpoint in ('/v1/cases', '/v1/agent/verify'):
                    response = client.post(endpoint, json={'claim': 'Scientific claim', 'max_retries': 1})
                    self.assertEqual(response.status_code, 422)
                    self.assertIn('frozen calibration', response.json()['detail'])
                self.assertEqual(client.get('/v1/cases').json(), [])

    def test_case_api_runs_verification_then_persists_human_verdict(self):
        class FakeRag:
            def verify_claim(self, claim, *, top_k, max_retries):
                return {"action": "abstain", "label": "NEI", "confidence": 0.2, "evidence": [], "trace": []}

        with tempfile.TemporaryDirectory() as directory:
            app = build_app(FakeRag(), case_db_path=Path(directory) / "cases.sqlite3")
            with TestClient(app) as client:
                created = client.post("/v1/cases", json={"claim": "Claim text", "top_k": 5, "max_retries": 1})
                self.assertEqual(created.status_code, 200)
                case_id = created.json()["case_id"]
                reviewed = client.post(
                    f"/v1/cases/{case_id}/reviews",
                    json={"reviewer": "reviewer-1", "verdict": "NEI", "note": "No evidence"},
                )
                self.assertEqual(reviewed.json()["status"], "reviewed")
                self.assertEqual(len(client.get("/v1/cases").json()), 1)
                report = client.get(f"/v1/cases/{case_id}/report").raise_for_status()
                self.assertFalse(report.json()["training_eligible"])
                self.assertEqual(report.json()["reviews"][0]["reviewer"], "reviewer-1")
                self.assertEqual(client.get("/v1/cases/missing/report").status_code, 404)
                self.assertEqual(client.post("/v1/cases", json={"claim":"   "}).status_code, 422)

    def test_case_and_human_review_survive_store_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.sqlite3"
            first = ResearchCaseStore(path)
            case = first.create("Claim text", {"label": "NEI", "action": "abstain", "trace": []})
            self.assertEqual(case["status"], "needs_review")

            reopened = ResearchCaseStore(path)
            reviewed = reopened.review(case["case_id"], "reviewer-1", "NEI", "Evidence is insufficient.")
            self.assertEqual(reviewed["status"], "reviewed")
            self.assertEqual(reviewed["reviews"][0]["verdict"], "NEI")
            self.assertEqual(reopened.list(10)[0]["case_id"], case["case_id"])

    def test_unknown_case_is_not_silently_created(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchCaseStore(Path(directory) / "cases.sqlite3")
            with self.assertRaises(KeyError):
                store.get("missing")


if __name__ == "__main__":
    unittest.main()
