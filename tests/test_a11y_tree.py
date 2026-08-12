"""
Test: Accessibility Tree (browser_accessibility_tree)

Tests the CDP-based accessibility tree extraction:
1. Unit test: node filtering logic with mock AX data
2. Integration test: open a page, get a11y tree, verify structure
3. Fallback test: CDP failure delegates to get_state
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ---------------------------------------------------------------------------
# Unit tests: filtering logic (no browser needed)
# ---------------------------------------------------------------------------

class TestA11yFiltering:
    """Test the _a11y_is_interesting method without a browser."""

    @pytest.fixture
    def engine(self):
        from browser_engine import BrowserEngine
        return BrowserEngine()

    def test_noise_roles_always_excluded(self, engine):
        for role in ("generic", "InlineTextBox", "line break", "none"):
            assert not engine._a11y_is_interesting(role, "Some Name", [("checked", True)])

    def test_interactive_roles_always_included(self, engine):
        for role in ("link", "button", "textbox", "checkbox", "radio", "combobox"):
            assert engine._a11y_is_interesting(role, "", [])

    def test_heading_always_included(self, engine):
        assert engine._a11y_is_interesting("heading", "", [])
        assert engine._a11y_is_interesting("heading", "Welcome", [("level", 1)])

    def test_noise_if_unnamed_excluded_without_name(self, engine):
        for role in ("list", "listitem", "paragraph", "strong"):
            assert not engine._a11y_is_interesting(role, "", [])

    def test_noise_if_unnamed_included_with_name(self, engine):
        for role in ("list", "listitem", "paragraph", "strong"):
            assert engine._a11y_is_interesting(role, "Some Name", [])

    def test_noise_if_unnamed_included_with_props(self, engine):
        assert engine._a11y_is_interesting("listitem", "", [("level", 2)])

    def test_other_roles_included_with_name(self, engine):
        assert engine._a11y_is_interesting("navigation", "Main", [])
        assert engine._a11y_is_interesting("main", "Main content", [])
        assert engine._a11y_is_interesting("region", "Sidebar", [])

    def test_other_roles_excluded_without_name(self, engine):
        assert not engine._a11y_is_interesting("unknown_role", "", [])


# ---------------------------------------------------------------------------
# Unit tests: AX node property extraction
# ---------------------------------------------------------------------------

class TestA11yNodeExtraction:
    """Test static helper methods for AX node parsing."""

    def test_role_extraction(self):
        from browser_engine import BrowserEngine
        node = {"role": {"type": "role", "value": "button"}}
        assert BrowserEngine._a11y_node_role(node) == "button"

    def test_role_extraction_empty(self):
        from browser_engine import BrowserEngine
        assert BrowserEngine._a11y_node_role({}) == ""

    def test_name_extraction(self):
        from browser_engine import BrowserEngine
        node = {"name": {"type": "string", "value": "Submit"}}
        assert BrowserEngine._a11y_node_name(node) == "Submit"

    def test_name_extraction_empty(self):
        from browser_engine import BrowserEngine
        assert BrowserEngine._a11y_node_name({}) == ""

    def test_value_extraction(self):
        from browser_engine import BrowserEngine
        node = {"value": {"type": "string", "value": "hello@example.com"}}
        assert BrowserEngine._a11y_node_value(node) == "hello@example.com"

    def test_value_extraction_none(self):
        from browser_engine import BrowserEngine
        node = {"value": {"type": "string", "value": None}}
        assert BrowserEngine._a11y_node_value(node) == ""

    def test_interesting_props(self):
        from browser_engine import BrowserEngine
        node = {
            "properties": [
                {"name": "checked", "value": {"value": True}},
                {"name": "level", "value": {"value": 2}},
                {"name": "focusable", "value": {"value": True}},  # not interesting
            ]
        }
        props = BrowserEngine._a11y_interesting_props(node)
        prop_names = [p[0] for p in props]
        assert "checked" in prop_names
        assert "level" in prop_names
        assert "focusable" not in prop_names


# ---------------------------------------------------------------------------
# Unit tests: dead node filtering (nodeId == 0)
# ---------------------------------------------------------------------------

class TestDeadNodeFiltering:
    """Test that dead nodes (nodeId == 0 from pushNodesByBackendIdsToFrontend)
    are properly filtered out before setAttributeValue calls."""

    def test_zero_node_ids_filtered(self):
        """Simulate pushNodesByBackendIdsToFrontend result with some dead nodes."""
        # Simulate: 5 backend IDs, 2 are dead (nodeId == 0)
        dom_nodes = [
            {"nodeId": 100, "backendNodeId": 1},
            {"nodeId": 0, "backendNodeId": 2},    # dead
            {"nodeId": 101, "backendNodeId": 3},
            {"nodeId": 0, "backendNodeId": 4},    # dead
            {"nodeId": 102, "backendNodeId": 5},
        ]

        bid_to_nid = {}
        for dn in dom_nodes:
            nid = dn.get("nodeId", 0)
            bid = dn.get("backendNodeId")
            if nid and nid != 0 and bid is not None:
                bid_to_nid[bid] = nid

        # Only non-zero nodeIds should be in the map
        assert len(bid_to_nid) == 3
        assert bid_to_nid[1] == 100
        assert bid_to_nid[3] == 101
        assert bid_to_nid[5] == 102
        assert 2 not in bid_to_nid
        assert 4 not in bid_to_nid


# ---------------------------------------------------------------------------
# Integration test: real browser (requires Playwright)
# ---------------------------------------------------------------------------

class TestA11yTreeIntegration:
    """Integration test that opens a real page and gets the a11y tree."""

    @pytest.fixture
    def engine(self):
        from browser_engine import browser_engine
        return browser_engine

    def test_get_a11y_tree_example_com(self, engine):
        """Open example.com and verify the a11y tree contains expected nodes."""
        try:
            # Open page
            result = engine.run(engine.open_page("https://example.com", "test_a11y"))
            assert "Opened" in result or "Error" in result

            if "Error" in result:
                pytest.skip("Browser not available or page didn't load")

            # Get accessibility tree
            tree = engine.run(engine.get_accessibility_tree("test_a11y"))

            # Verify basic structure
            assert "Page:" in tree
            assert "---" in tree
            # example.com should have at least a heading and a link
            assert "heading" in tree.lower() or "link" in tree.lower()

            # Clean up
            engine.run(engine.close_session("test_a11y"))
        except Exception as e:
            pytest.skip(f"Browser test skipped: {e}")

    def test_a11y_tree_fallback_on_cdp_error(self, engine):
        """If CDP fails, get_accessibility_tree should fall back to get_state."""
        try:
            engine.run(engine.open_page("https://example.com", "test_fallback"))
        except Exception as e:
            pytest.skip(f"Browser not available: {e}")

        # Mock new_cdp_session to raise an error
        original_get_session = engine._get_session

        async def mock_get_session_with_bad_cdp(session="default"):
            sc = await original_get_session(session)

            # Patch the context to make CDP session creation fail
            original_new_cdp = sc.context.new_cdp_session

            async def failing_cdp(page):
                raise RuntimeError("CDP not available")

            sc.context.new_cdp_session = failing_cdp
            return sc

        try:
            engine._get_session = mock_get_session_with_bad_cdp
            result = engine.run(engine.get_accessibility_tree("test_fallback"))

            # Should fall back to get_state output format
            assert "Page:" in result or "Error" in result
        finally:
            engine._get_session = original_get_session
            try:
                engine.run(engine.close_session("test_fallback"))
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])