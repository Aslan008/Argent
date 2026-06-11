from src.agent.strategy import TinyLocalStrategy, StandardLocalStrategy, CloudStrategy


class TestTierAdaptationFlags:
    """The whole point of the tier system: behaviour adapts automatically and
    cloud models are never burdened by small-model accommodations."""

    def test_tiny_gets_constrained_decoding_and_anchor(self):
        s = TinyLocalStrategy()
        assert s.wants_constrained_decoding() is True
        assert s.wants_objective_anchor() is True
        assert s.supports_native_tools() is False

    def test_standard_local_gets_anchor_but_native_tools(self):
        s = StandardLocalStrategy()
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is True
        assert s.supports_native_tools() is True

    def test_cloud_is_completely_unburdened(self):
        s = CloudStrategy()
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is False
        assert s.supports_native_tools() is True
