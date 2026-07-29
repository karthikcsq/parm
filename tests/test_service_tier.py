import os
import unittest
from unittest import mock

from parm_bench.service_tier import service_tier_kwargs


class ServiceTierTests(unittest.TestCase):
    def test_unset_returns_no_kwargs(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_SERVICE_TIER", None)
            self.assertEqual(service_tier_kwargs(), {})

    def test_empty_returns_no_kwargs(self):
        with mock.patch.dict(os.environ, {"OPENAI_SERVICE_TIER": "  "}):
            self.assertEqual(service_tier_kwargs(), {})

    def test_flex_is_passed_through(self):
        with mock.patch.dict(os.environ, {"OPENAI_SERVICE_TIER": "flex"}):
            self.assertEqual(
                service_tier_kwargs(), {"service_tier": "flex"}
            )

    def test_tier_is_case_insensitive(self):
        with mock.patch.dict(os.environ, {"OPENAI_SERVICE_TIER": "Flex"}):
            self.assertEqual(
                service_tier_kwargs(), {"service_tier": "flex"}
            )

    def test_unknown_tier_raises(self):
        with mock.patch.dict(os.environ, {"OPENAI_SERVICE_TIER": "turbo"}):
            with self.assertRaises(ValueError):
                service_tier_kwargs()


if __name__ == "__main__":
    unittest.main()
