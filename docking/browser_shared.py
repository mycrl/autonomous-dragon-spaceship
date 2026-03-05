"""Shared managed-browser lifecycle helpers."""

from __future__ import annotations

import logging

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


class SharedLaunchCoordinator:
    """Helpers for one-browser multi-tab managed launch mode."""

    @staticmethod
    def connect_shared_launch(browser) -> None:
        """Connect using a single shared browser and one tab per env."""
        cls = type(browser)
        if cls._shared_browser is None:
            shared_playwright = sync_playwright().start()
            shared_browser = shared_playwright.chromium.launch(
                headless=browser._headless,
                args=browser.BROWSER_LAUNCH_ARGS,
            )
            shared_context = shared_browser.new_context(no_viewport=True)
            cls._shared_playwright = shared_playwright
            cls._shared_browser = shared_browser
            cls._shared_context = shared_context
            cls._shared_ref_count = 0
            cls._shared_instances = []
            cls._shared_tabs_prepared = False
            cls._shared_expected_tabs = browser._expected_shared_tabs
        else:
            cls._shared_expected_tabs = max(
                cls._shared_expected_tabs,
                browser._expected_shared_tabs,
            )

        assert cls._shared_context is not None
        browser._playwright = cls._shared_playwright
        browser._browser = cls._shared_browser
        browser._context = cls._shared_context
        browser._page = browser._context.new_page()

        browser._page.goto(
            browser.SIMULATOR_URL,
            wait_until="domcontentloaded",
            timeout=int(browser._page_load_timeout * 1_000),
        )
        browser._skip_next_reset_reload = True
        cls._shared_instances.append(browser)

        cls._shared_ref_count += 1
        logger.info(
            "Attached shared browser tab (%d/%d).",
            cls._shared_ref_count,
            cls._shared_expected_tabs,
        )

        if (
            not cls._shared_tabs_prepared
            and len(cls._shared_instances) >= cls._shared_expected_tabs
        ):
            browser._prepare_all_shared_tabs_before_training()

    @staticmethod
    def disconnect_shared_launch(browser) -> None:
        """Close only this tab; close shared browser when last tab exits."""
        cls = type(browser)

        if browser in cls._shared_instances:
            cls._shared_instances.remove(browser)

        if browser._page is not None:
            try:
                browser._page.close()
            except Exception:
                pass

        browser._page = None
        browser._context = None
        browser._browser = None
        browser._playwright = None

        cls._shared_ref_count = max(0, cls._shared_ref_count - 1)
        if cls._shared_ref_count > 0:
            return

        if cls._shared_browser is not None:
            try:
                cls._shared_browser.close()
            except Exception:
                pass
        if cls._shared_playwright is not None:
            try:
                cls._shared_playwright.stop()
            except Exception:
                pass

        cls._shared_context = None
        cls._shared_browser = None
        cls._shared_playwright = None
        cls._shared_instances = []
        cls._shared_tabs_prepared = False
        cls._shared_expected_tabs = 0
