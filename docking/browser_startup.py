"""Startup workflow helpers for simulator browser automation."""

from __future__ import annotations

import logging
import re
import time

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


class BrowserStartupCoordinator:
    """Stateless startup helpers that operate on a SimulatorBrowser instance."""

    @staticmethod
    def read_preloader_percent(
        browser,
        page: Page,
        timeout_ms: int = 1000,
    ) -> tuple[int | None, str]:
        """Return parsed percent and raw text from #preloader-percent."""
        try:
            raw = page.locator(browser.PRELOADER_PERCENT_SELECTOR).first.text_content(
                timeout=timeout_ms
            )
        except PlaywrightTimeoutError:
            return None, ""
        except Exception:
            return None, ""

        text = (raw or "").strip()
        match = re.search(r"\d{1,3}(?:\.\d+)?", text)
        if not match:
            return None, text

        try:
            return int(float(match.group(0))), text
        except ValueError:
            return None, text

    @staticmethod
    def wait_for_begin_button_ready(browser, page: Page, timeout_seconds: float) -> None:
        """Wait until begin button is visible/clickable in the DOM."""
        deadline = time.time() + max(1.0, timeout_seconds)
        last_error: str = ""

        while time.time() < deadline:
            try:
                locator = page.locator(browser.BEGIN_BUTTON_SELECTOR).first
                if locator.is_visible(timeout=200):
                    return
            except Exception as exc:
                last_error = str(exc)

            time.sleep(browser.BEGIN_CLICK_RETRY_INTERVAL_SECONDS)

        raise RuntimeError(
            "Timed out waiting for begin button readiness: "
            f"selector={browser.BEGIN_BUTTON_SELECTOR}, last_error='{last_error}'"
        )

    @staticmethod
    def click_begin_button_with_retries(browser, page: Page, timeout_seconds: float) -> None:
        """Click begin with retries and JS fallback for transient click failures."""
        deadline = time.time() + max(1.0, timeout_seconds)
        last_error: str = ""

        while time.time() < deadline:
            try:
                page.click(
                    browser.BEGIN_BUTTON_SELECTOR,
                    timeout=1000,
                    force=True,
                )
                logger.info("Clicked begin button: %s", browser.BEGIN_BUTTON_SELECTOR)
                return
            except Exception as exc:
                last_error = str(exc)

            # Fallback: try clicking via JS in case of transient Playwright click issues
            try:
                clicked = bool(
                    page.evaluate(
                        """
                        (selector) => {
                            const el = document.querySelector(selector);
                            if (!el) {
                                return false;
                            }
                            el.click();
                            return true;
                        }
                        """,
                        browser.BEGIN_BUTTON_SELECTOR,
                    )
                )
                if clicked:
                    logger.info("Clicked begin button via JS: %s", browser.BEGIN_BUTTON_SELECTOR)
                    return
            except Exception as exc:
                last_error = str(exc)

            time.sleep(browser.BEGIN_CLICK_RETRY_INTERVAL_SECONDS)

        raise RuntimeError(
            "Timed out clicking begin button: "
            f"selector={browser.BEGIN_BUTTON_SELECTOR}, last_error='{last_error}'"
        )

    @staticmethod
    def prepare_all_shared_tabs_before_training(browser) -> None:
        """Run fixed startup flow for all shared tabs and block until all are ready."""
        cls = type(browser)
        if cls._shared_tabs_prepared:
            return

        instances = [inst for inst in cls._shared_instances if inst._page is not None]
        if not instances:
            raise RuntimeError("No shared tab instances available for startup.")

        timeout_seconds = max(browser._page_load_timeout * 4, 600.0)
        deadline = time.time() + timeout_seconds

        state: dict[int, dict[str, float | str]] = {}
        for inst in instances:
            state[id(inst)] = {
                "phase": "wait_preloader",
                "preloader_ready_at": 0.0,
                "begin_clicked_at": 0.0,
            }

        logger.info(
            "Preparing %d shared tabs in parallel startup workflow ...",
            len(instances),
        )

        while time.time() < deadline:
            now = time.time()
            all_done = True

            for inst in instances:
                if inst._page is None:
                    continue

                tab_state = state[id(inst)]
                phase = str(tab_state["phase"])

                if phase == "done":
                    continue

                all_done = False
                page = inst._page

                if phase == "wait_preloader":
                    try:
                        percent, _ = BrowserStartupCoordinator.read_preloader_percent(
                            inst,
                            page,
                            timeout_ms=300,
                        )
                        if percent is not None and percent >= 100:
                            tab_state["preloader_ready_at"] = now
                            tab_state["phase"] = "wait_after_load"
                    except PlaywrightTimeoutError:
                        pass
                    except Exception:
                        pass
                    continue

                if phase == "wait_after_load":
                    preloader_ready_at = float(tab_state["preloader_ready_at"])
                    if (now - preloader_ready_at) >= browser.AFTER_LOAD_WAIT_SECONDS:
                        try:
                            BrowserStartupCoordinator.wait_for_begin_button_ready(
                                inst,
                                page,
                                timeout_seconds=8.0,
                            )
                            BrowserStartupCoordinator.click_begin_button_with_retries(
                                inst,
                                page,
                                timeout_seconds=8.0,
                            )
                            tab_state["begin_clicked_at"] = now
                            tab_state["phase"] = "wait_after_begin"
                        except Exception:
                            pass
                    continue

                if phase == "wait_after_begin":
                    begin_clicked_at = float(tab_state["begin_clicked_at"])
                    if (now - begin_clicked_at) >= browser.AFTER_BEGIN_WAIT_SECONDS:
                        tab_state["phase"] = "done"
                        inst._startup_completed = True
                        inst._skip_next_reset_reload = True
                    continue

            if all_done:
                cls._shared_tabs_prepared = True
                logger.info("All shared tabs are ready; starting parallel training.")
                return

            time.sleep(browser.PRELOADER_POLL_INTERVAL_SECONDS)

        raise RuntimeError(
            "Timed out while preparing shared tabs for startup. "
            f"expected={cls._shared_expected_tabs}, connected={len(instances)}"
        )

    @staticmethod
    def wait_for_preloader_complete(browser, timeout_seconds: float) -> None:
        """Wait until #preloader-percent reaches 100."""
        browser._require_page()
        page = browser._page
        assert page is not None

        deadline = time.time() + timeout_seconds
        last_seen: str = ""

        while time.time() < deadline:
            try:
                percent, text = BrowserStartupCoordinator.read_preloader_percent(
                    browser,
                    page,
                    timeout_ms=1000,
                )
                if text:
                    last_seen = text
                if percent is not None and percent >= 100:
                    logger.info("Preloader reached 100%% (%s).", last_seen or text)
                    return
            except PlaywrightTimeoutError:
                pass
            except Exception:
                pass

            time.sleep(browser.PRELOADER_POLL_INTERVAL_SECONDS)

        raise RuntimeError(
            "Timed out waiting for preloader completion: "
            f"selector={browser.PRELOADER_PERCENT_SELECTOR}, last_seen='{last_seen}'"
        )

    @staticmethod
    def auto_start_simulator_if_needed(browser) -> None:
        """Run deterministic managed-mode startup sequence."""
        browser._require_page()
        page = browser._page
        assert page is not None

        BrowserStartupCoordinator.wait_for_preloader_complete(
            browser,
            timeout_seconds=max(browser._page_load_timeout * 4, 600.0),
        )

        time.sleep(browser.AFTER_LOAD_WAIT_SECONDS)
        BrowserStartupCoordinator.wait_for_begin_button_ready(
            browser,
            page,
            timeout_seconds=max(browser._page_load_timeout, 10.0),
        )
        BrowserStartupCoordinator.click_begin_button_with_retries(
            browser,
            page,
            timeout_seconds=max(browser._page_load_timeout, 10.0),
        )

        time.sleep(browser.AFTER_BEGIN_WAIT_SECONDS)
