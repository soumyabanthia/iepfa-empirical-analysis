import unittest
from fastapi.testclient import TestClient
from backend.app import app
from backend.matching_engine import levenshtein_ratio, soundex_key, evaluate_identity_match


class TestUnifiedPortal(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "Operational")
        self.assertEqual(len(data["regulators_connected"]), 4)

    def test_matching_engine_soundex_and_levenshtein(self):
        ratio = levenshtein_ratio("SOUMYA BANTHIA", "SOUMYA BANTHIA")
        self.assertEqual(ratio, 1.0)

        ratio_partial = levenshtein_ratio("SOUMYA BANTHIA", "SOUMYA K BANTHIA")
        self.assertGreater(ratio_partial, 0.8)

        soundex_code = soundex_key("BANTHIA")
        self.assertEqual(len(soundex_code), 4)

        is_match, conf, match_type = evaluate_identity_match(
            claimant_pan="ABCDE1234F",
            claimant_name="SOUMYA BANTHIA",
            record_pan="ABCDE1234F",
            record_name="SOUMYA BANTHIA"
        )
        self.assertTrue(is_match)
        self.assertEqual(conf, 100.0)
        self.assertEqual(match_type, "EXACT_PAN_MATCH")

    def test_unified_search_endpoint(self):
        payload = {
            "pan": "ABCDE1234F",
            "full_name": "SOUMYA BANTHIA"
        }
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["pan_searched"], "ABCDE1234F")
        self.assertGreaterEqual(data["total_assets_found"], 4)
        self.assertGreater(data["total_estimated_value_inr"], 500000)

        regulators = [a["regulator"] for a in data["assets"]]
        self.assertTrue(any("IEPFA" in r for r in regulators))
        self.assertTrue(any("RBI" in r for r in regulators))
        self.assertTrue(any("SEBI" in r for r in regulators))
        self.assertTrue(any("IRDAI" in r for r in regulators))

    def test_claim_submission_and_tracking_workflow(self):
        search_res = self.client.post("/api/v1/search", json={"pan": "ABCDE1234F", "full_name": "SOUMYA BANTHIA"})
        self.assertEqual(search_res.status_code, 200)
        assets = search_res.json()["assets"]
        selected_ids = [assets[0]["asset_id"], assets[1]["asset_id"]]

        submit_payload = {
            "claimant": {
                "pan": "ABCDE1234F",
                "full_name": "SOUMYA BANTHIA",
                "date_of_birth": "2000-01-01",
                "aadhaar_hash": "a8f5c...99b2",
                "mobile_number": "+91-9876543210",
                "email": "soumya@example.com",
                "digilocker_kyc_verified": True,
                "bank_account_number": "998877665544",
                "bank_ifsc": "SBIN0001234",
                "demat_account_id": "1208160012345678"
            },
            "selected_asset_ids": selected_ids,
            "is_heir_succession": False,
            "digilocker_document_ids": ["DOC-PAN-VERIFIED", "DOC-AADHAAR-EKYC"],
            "indemnity_bond_signed": True
        }

        claim_res = self.client.post("/api/v1/claims/submit", json=submit_payload)
        self.assertEqual(claim_res.status_code, 200)
        claim_data = claim_res.json()
        master_id = claim_data["master_tracking_id"]
        self.assertTrue(master_id.startswith("NAT-ASSET-REC-"))
        self.assertEqual(len(claim_data["sub_claims"]), 2)

        track_res = self.client.get(f"/api/v1/claims/{master_id}/track")
        self.assertEqual(track_res.status_code, 200)
        track_data = track_res.json()
        self.assertEqual(track_data["master_tracking_id"], master_id)
        self.assertEqual(track_data["overall_status"], "IN_PROGRESS_WITH_REGULATORS")

    def test_analytics_summary_endpoint(self):
        res = self.client.get("/api/v1/analytics/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("national_unclaimed_corpus_inr_cr", data)
        self.assertGreater(data["national_unclaimed_corpus_inr_cr"], 100000)


if __name__ == "__main__":
    unittest.main()

