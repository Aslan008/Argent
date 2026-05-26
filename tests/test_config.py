import unittest
import os
import json
import config

class TestConfigCache(unittest.TestCase):
    def setUp(self):
        # Force cache reload by resetting it
        config._CONFIG_CACHE = None

    def test_cache_population(self):
        # Verify cache is populated on load
        self.assertIsNone(config._CONFIG_CACHE)
        cfg = config.load_config()
        self.assertIsNotNone(config._CONFIG_CACHE)
        self.assertEqual(cfg, config._CONFIG_CACHE)

    def test_save_updates_cache(self):
        cfg = config.load_config()
        original_model = cfg.get("model")
        
        # Modify model
        config.set_current_model("test-model-abc")
        self.assertEqual(config._CONFIG_CACHE.get("model"), "test-model-abc")
        
        # Restore model
        if original_model:
            config.set_current_model(original_model)

if __name__ == "__main__":
    unittest.main()
