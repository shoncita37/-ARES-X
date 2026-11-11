import unittest
from unittest.mock import patch

from ares_x_server import app


class ApiOriginTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_mutation_rejects_missing_and_untrusted_origins(self):
        for headers in ({}, {"Origin": "https://attacker.example"}):
            with self.subTest(headers=headers):
                response = self.client.post("/api/reset", json={}, headers=headers)
                self.assertEqual(response.status_code, 403)

    def test_mutation_accepts_the_deployed_frontend_origin(self):
        with patch("ares_x_server._get_engine") as get_engine:
            get_engine.return_value.step.return_value = {"steps": 1}
            response = self.client.post(
                "/api/step",
                headers={"Origin": "https://ares-x-ac.netlify.app"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"steps": 1})
        self.assertEqual(response.headers["Cache-Control"], "no-store")


if __name__ == "__main__":
    unittest.main()
