import unittest
import os
import asyncio
from browser_engine import browser_engine, _SessionContext
import config

class TestBrowserEnhancements(unittest.TestCase):
    def setUp(self):
        # Save original settings
        self.original_mode = config.get_browser_mode()
        # Reset browser engine to clean state
        browser_engine.run(browser_engine.shutdown())

    def tearDown(self):
        # Restore settings and shutdown browser
        config.set_browser_mode(self.original_mode)
        browser_engine.run(browser_engine.shutdown())

    def test_selector_and_text_click(self):
        # 1. Open a page with simple buttons
        html = "data:text/html,<html><body><button id='btn1'>Button One</button><button id='btn2'>Button Two</button></body></html>"
        res = browser_engine.run(browser_engine.open_page(html, session="test_click"))
        self.assertNotIn("Error", res)

        # 2. Click by selector
        click_res = browser_engine.run(browser_engine.click(selector="#btn1", session="test_click"))
        self.assertIn("Clicked element matching selector", click_res)

        # 3. Click by text
        click_res = browser_engine.run(browser_engine.click(text="Button Two", session="test_click"))
        self.assertIn("Clicked element matching text", click_res)

    def test_selector_input(self):
        # 1. Open a page with an input field
        html = "data:text/html,<html><body><input type='text' id='inp' value='original'/></body></html>"
        browser_engine.run(browser_engine.open_page(html, session="test_input"))

        # 2. Fill input by selector
        fill_res = browser_engine.run(browser_engine.fill_input(text="new value", selector="#inp", session="test_input"))
        self.assertIn("Filled element matching selector", fill_res)

        # 3. Verify value
        val = browser_engine.run(browser_engine.eval_js("document.getElementById('inp').value", session="test_input"))
        self.assertIn("new value", val)

    def test_iframe_element_extraction(self):
        # 1. Open a page containing a nested iframe with an interactive button
        iframe_src = "data:text/html,<html><body><button id='inner-btn'>Inside Frame</button></body></html>"
        html = f'data:text/html,<html><body><h1>Outer</h1><iframe src="{iframe_src}"></iframe></body></html>'
        
        browser_engine.run(browser_engine.open_page(html, session="test_iframe"))
        
        # Wait a bit for iframe to load
        sc = browser_engine.run(browser_engine._get_session("test_iframe"))
        browser_engine.run(sc.page.wait_for_timeout(2000))

        # 2. Get state and verify iframe button is indexed
        state = browser_engine.run(browser_engine.get_state(session="test_iframe"))
        self.assertIn("button \"Inside Frame\"", state)

        # 3. Verify we mapped the element to the iframe
        self.assertNotEqual(len(browser_engine._element_frames), 0)

        # Find the index of the inside frame button
        button_idx = None
        for idx, frame in browser_engine._element_frames.items():
            if frame != sc.page.main_frame:
                button_idx = idx
                break

        self.assertIsNotNone(button_idx, "Could not find element in iframe")

        # 4. Click the element by index (which resides in the iframe)
        click_res = browser_engine.run(browser_engine.click(index=button_idx, session="test_iframe"))
        self.assertIn(f"Clicked element [{button_idx}]", click_res)

    def test_scroll_to_element(self):
        # 1. Create a tall page with a button at the bottom
        html = "data:text/html,<html><body><div style='height: 2000px;'>Spacer</div><button id='bot-btn'>Bottom Button</button></body></html>"
        browser_engine.run(browser_engine.open_page(html, session="test_scroll"))

        # Wait for rendering
        sc = browser_engine.run(browser_engine._get_session("test_scroll"))
        browser_engine.run(sc.page.wait_for_timeout(1000))

        # 2. Extract state to map the element
        state = browser_engine.run(browser_engine.get_state(session="test_scroll"))
        self.assertIn("bot-btn", state)

        # Get its index
        button_idx = list(browser_engine._element_frames.keys())[0]

        # 3. Scroll to it
        scroll_res = browser_engine.run(browser_engine.scroll(index=button_idx, session="test_scroll"))
        self.assertIn("Scrolled element", scroll_res)

        # Check that scrollY is greater than 0
        scroll_y = browser_engine.run(browser_engine.eval_js("window.scrollY", session="test_scroll"))
        self.assertGreater(int(scroll_y.split(":")[-1].strip()), 0)

    def test_cdp_mode_auto_attach_mock(self):
        # We simulate CDP mode by mocking config and browser engine context
        config.set_browser_mode("user")
        browser_engine._cdp_mode = True
        
        # Mock class to stub Playwright Browser and Context
        class MockPage:
            def __init__(self):
                self.url = "https://mock.url"
            def is_closed(self):
                return False
            async def title(self):
                return "Mock Title"

        class MockContext:
            def __init__(self):
                self.pages = [MockPage()]

        class MockBrowser:
            def __init__(self):
                self.contexts = [MockContext()]

        browser_engine._browser = MockBrowser()
        
        # Verify that calling _get_session without browser_open auto-attaches
        sc = browser_engine.run(browser_engine._get_session("mock_session"))
        self.assertIsNotNone(sc)
        self.assertEqual(sc.page.url, "https://mock.url")
        self.assertIn("mock_session", browser_engine._sessions)

    def test_keyword_filtering(self):
        # 1. Open a page with multiple different elements
        html = "data:text/html,<html><body><button id='btn-submit'>Submit Form</button><a href='https://link.com'>Cancel Link</a><input type='text' placeholder='Email Address'/></body></html>"
        browser_engine.run(browser_engine.open_page(html, session="test_filter"))

        # 2. Get state without filter (verify all elements are shown)
        state_all = browser_engine.run(browser_engine.get_state(session="test_filter"))
        elements_all = state_all.split("---")[1]
        self.assertIn("button \"Submit Form\"", elements_all)
        self.assertIn("a \"Cancel Link\"", elements_all)
        self.assertIn('input[type=text] "Email Address"', elements_all)

        # 3. Get state with filter 'Submit'
        state_submit = browser_engine.run(browser_engine.get_state(session="test_filter", query="Submit"))
        elements_submit = state_submit.split("---")[1]
        self.assertIn("button \"Submit Form\"", elements_submit)
        self.assertNotIn("Cancel Link", elements_submit)
        self.assertNotIn("Email Address", elements_submit)

        # 4. Get state with filter 'Email'
        state_email = browser_engine.run(browser_engine.get_state(session="test_filter", query="Email"))
        elements_email = state_email.split("---")[1]
        self.assertNotIn("Submit Form", elements_email)
        self.assertNotIn("Cancel Link", elements_email)
        self.assertIn('input[type=text] "Email Address"', elements_email)

        # 5. Get state with multi-word filter 'Submit, Email' (should match both)
        state_multi = browser_engine.run(browser_engine.get_state(session="test_filter", query="Submit, Email"))
        elements_multi = state_multi.split("---")[1]
        self.assertIn("button \"Submit Form\"", elements_multi)
        self.assertNotIn("Cancel Link", elements_multi)
        self.assertIn('input[type=text] "Email Address"', elements_multi)

if __name__ == "__main__":
    unittest.main()
