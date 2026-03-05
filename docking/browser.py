"""
Browser automation layer for the SpaceX ISS Docking Simulator.

Supports two operating modes:

**Managed mode** (``launch=True``)
    Playwright launches its own Chromium browser, navigates to the simulator
    URL, and manages the full browser lifecycle.  No manual Chrome startup is
    required::

        browser = SimulatorBrowser(launch=True)
        browser.connect()

    For multi-env training, managed mode can also reuse one shared browser
    instance with multiple tabs (one tab per environment)::

        browser = SimulatorBrowser(launch=True, shared_launch=True)
        browser.connect()

**CDP mode** (default, ``launch=False``)
    Playwright connects to an already-opened Chrome instance via the Chrome
    DevTools Protocol.  Start Chrome manually first::

        google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/

    Then::

        browser = SimulatorBrowser()   # or SimulatorBrowser(cdp_url="http://localhost:9222")
        browser.connect()

.. note::
    In managed mode, the first :meth:`SimulatorBrowser.reset` runs a fixed
    startup sequence: wait until ``#preloader-percent`` reaches ``100``, wait
    10 seconds, click ``#begin-button``, then wait another 10 seconds.
    Later resets are fully automatic and click the in-page restart button
    (``#option-restart``) instead of reloading the page.
"""

import logging
import re
import time
from typing import Optional

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

from .browser_shared import SharedLaunchCoordinator
from .browser_startup import BrowserStartupCoordinator

logger = logging.getLogger(__name__)


class SimulatorBrowser:
    """
    Controls the SpaceX ISS Docking Simulator via Playwright.

    Two modes are supported — select via the *launch* constructor parameter:

    * **Managed mode** (``launch=True``): Playwright starts its own Chromium
      browser, navigates to :attr:`SIMULATOR_URL`, and owns the lifecycle.
      The browser is closed when :meth:`disconnect` is called.
    * **CDP mode** (``launch=False``, default): Playwright connects to an
      already-running Chrome instance via the Chrome DevTools Protocol endpoint
      at *cdp_url*.

    Parameters
    ----------
    launch:
        When ``True``, launch a new browser automatically.
        When ``False`` (default), connect to an existing browser via CDP.
    headless:
        Only used when ``launch=True``.  If ``True``, the browser runs without
        a visible window (headless).  Defaults to ``False`` so the simulator
        UI is visible during training.
    cdp_url:
        URL of the Chrome remote-debugging endpoint used in CDP mode
        (default: ``http://localhost:9222``).  Ignored when ``launch=True``.
    page_load_timeout:
        Seconds to wait for the simulator page to load before raising a
        timeout error.
    shared_launch:
        Only used when ``launch=True``. If ``True``, all instances in the
        same process share one Chromium instance and open one tab per
        environment.
    """

    # Shared managed browser state for single-process multi-tab training.
    _shared_playwright: Optional[Playwright] = None
    _shared_browser: Optional[Browser] = None
    _shared_context: Optional[BrowserContext] = None
    _shared_ref_count: int = 0
    _shared_instances: list["SimulatorBrowser"] = []
    _shared_tabs_prepared: bool = False
    _shared_expected_tabs: int = 0

    # Keep simulator active in background tabs during multi-tab training.
    BROWSER_LAUNCH_ARGS: list[str] = [
        "--start-maximized",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
    ]

    CDP_URL: str = "http://localhost:9222"
    SIMULATOR_URL: str = "https://iss-sim.spacex.com/"

    # ------------------------------------------------------------------
    # CSS selectors for control buttons.
    # ------------------------------------------------------------------
    BUTTON_SELECTORS: dict[str, str] = {
        "translate_forward":  "#translate-forward-button",
        "translate_backward": "#translate-backward-button",
        "translate_up":       "#translate-up-button",
        "translate_down":     "#translate-down-button",
        "translate_left":     "#translate-left-button",
        "translate_right":    "#translate-right-button",
        "roll_left":          "#roll-left-button",
        "roll_right":         "#roll-right-button",
        "pitch_up":           "#pitch-up-button",
        "pitch_down":         "#pitch-down-button",
        "yaw_left":           "#yaw-left-button",
        "yaw_right":          "#yaw-right-button",
        "toggle_translation": "#toggle-translation",
        "toggle_rotation":    "#toggle-rotation",
    }

    # ------------------------------------------------------------------
    # CSS selectors for state readout elements.
    # The simulator DOM uses a different child-element structure per field:
    #   x, y, z       — single div inside their container (#x-range, etc.)
    #   roll/yaw/pitch — first child div = angle (°); second = angular rate (°/s)
    #   range          — second child div = distance to port (m)
    #   rate           — second child div = approach rate (m/s)
    # ------------------------------------------------------------------
    STATE_SELECTORS: dict[str, str] = {
        "x":          "#x-range div",
        "y":          "#y-range div",
        "z":          "#z-range div",
        "roll":       "#roll div:nth-child(1)",
        "roll_rate":  "#roll div:nth-child(2)",
        "range":      "#range div:nth-child(2)",
        "yaw":        "#yaw div:nth-child(1)",
        "yaw_rate":   "#yaw div:nth-child(2)",
        "rate":       "#rate div:nth-child(2)",
        "pitch":      "#pitch div:nth-child(1)",
        "pitch_rate": "#pitch div:nth-child(2)",
    }

    def __init__(
        self,
        launch: bool = False,
        headless: bool = False,
        cdp_url: str = CDP_URL,
        page_load_timeout: float = 30.0,
        shared_launch: bool = False,
        expected_shared_tabs: int | None = None,
    ) -> None:
        self._launch = launch
        self._headless = headless
        self._cdp_url = cdp_url
        self._page_load_timeout = page_load_timeout
        self._shared_launch = shared_launch
        self._expected_shared_tabs = max(1, int(expected_shared_tabs or 1))
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._skip_next_reset_reload: bool = False
        self._startup_completed: bool = not launch

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the simulator.

        In **managed mode** (``launch=True``), Playwright launches a new
        Chromium browser and navigates directly to :attr:`SIMULATOR_URL`.

        In **CDP mode** (``launch=False``, default), Playwright connects to an
        already-running Chrome instance via the DevTools Protocol endpoint
        specified by ``cdp_url``.
        """
        if self._launch and self._shared_launch:
            self._connect_shared_launch()
            return

        self._playwright = sync_playwright().start()

        if self._launch:
            self._browser = self._playwright.chromium.launch(
                headless=self._headless,
                args=self.BROWSER_LAUNCH_ARGS,
            )
            self._context = self._browser.new_context(no_viewport=True)
            self._page = self._context.new_page()
            self._page.goto(
                self.SIMULATOR_URL,
                wait_until="domcontentloaded",
                timeout=int(self._page_load_timeout * 1_000),
            )
            self._skip_next_reset_reload = True
            logger.info("Launched browser and navigated to %s", self.SIMULATOR_URL)
        else:
            self._browser = self._playwright.chromium.connect_over_cdp(self._cdp_url)

            contexts = self._browser.contexts
            if not contexts:
                raise RuntimeError(
                    "No browser contexts found. "
                    f"Open the simulator at {self.SIMULATOR_URL} first."
                )

            pages = contexts[0].pages
            if not pages:
                raise RuntimeError("No open pages found in the browser context.")

            # Prefer the simulator page; fall back to the first available page.
            self._page = next(
                (p for p in pages if self.SIMULATOR_URL in p.url),
                pages[0],
            )

            if self.SIMULATOR_URL not in self._page.url:
                logger.warning(
                    "Simulator page not found; using page: %s", self._page.url
                )
            else:
                logger.info("Connected to simulator page: %s", self._page.url)

    def disconnect(self) -> None:
        """Close the CDP connection and release Playwright resources."""
        if self._launch and self._shared_launch:
            self._disconnect_shared_launch()
            return

        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._context = None
        self._browser = None
        self._page = None
        self._playwright = None

    def _connect_shared_launch(self) -> None:
        """Connect using a single shared browser and one tab per env."""
        SharedLaunchCoordinator.connect_shared_launch(self)

    def _disconnect_shared_launch(self) -> None:
        """Close only this tab; close shared browser when last tab exits."""
        SharedLaunchCoordinator.disconnect_shared_launch(self)

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    # CSS selector for the in-simulator restart button.
    RESTART_BUTTON_SELECTOR: str = "#option-restart"

    def reset(self, wait: float = 5.0) -> None:
        """Reset the simulator to start a new episode.

        Clicks the simulator's own restart button (``#option-restart``) to
        return to the initial state without reloading the page.  On the very
        first call after :meth:`connect` the click is skipped because the
        simulator has already been freshly started.

        Parameters
        ----------
        wait:
            Extra seconds to sleep after clicking restart so that the
            simulator's initialisation animation can complete.
        """
        self._require_page()
        page = self._page
        assert page is not None

        if not self._startup_completed:
            if self._launch and self._shared_launch:
                if not SimulatorBrowser._shared_tabs_prepared:
                    self._prepare_all_shared_tabs_before_training()
                if not SimulatorBrowser._shared_tabs_prepared:
                    raise RuntimeError("Shared tab startup was not completed.")
            else:
                self._auto_start_simulator_if_needed()
            self._startup_completed = True
            # Startup was already done on the fresh page; next reset should
            # use the simulator restart button.
            self._skip_next_reset_reload = False
            if wait > 0 and not (self._launch and self._shared_launch):
                time.sleep(wait)
            return

        if self._skip_next_reset_reload:
            self._skip_next_reset_reload = False
            if self._launch and self._shared_launch:
                return
        else:
            page.click(
                self.RESTART_BUTTON_SELECTOR,
                timeout=int(self._page_load_timeout * 1_000),
            )
            logger.info("Clicked restart button (%s).", self.RESTART_BUTTON_SELECTOR)
        if wait > 0:
            time.sleep(wait)

    # ------------------------------------------------------------------
    # Actions & observations
    # ------------------------------------------------------------------

    def click_action(self, action_name: str) -> None:
        """Click the control button that corresponds to *action_name*.

        Parameters
        ----------
        action_name:
            One of the keys in :attr:`BUTTON_SELECTORS`.

        Raises
        ------
        ValueError
            If the selector for *action_name* has not been configured.
        """
        self._require_page()
        page = self._page
        assert page is not None
        selector = self.BUTTON_SELECTORS.get(action_name)
        if not selector:
            raise ValueError(
                f"Selector for action '{action_name}' is not configured. "
                "Fill in BUTTON_SELECTORS with the correct CSS selectors."
            )
        page.click(selector)

    def read_state(self) -> dict[str, float]:
        """Read the current simulator state from the page DOM.

        Returns
        -------
        dict[str, float]
            Keys: ``x``, ``y``, ``z``,
            ``roll``, ``roll_rate``,
            ``range``,
            ``yaw``, ``yaw_rate``,
            ``rate``,
            ``pitch``, ``pitch_rate``.

        Raises
        ------
        ValueError
            If any selector in :attr:`STATE_SELECTORS` has not been configured.
        """
        self._require_page()
        page = self._page
        assert page is not None
        state: dict[str, float] = {}
        for key, selector in self.STATE_SELECTORS.items():
            if not selector:
                raise ValueError(
                    f"Selector for state '{key}' is not configured. "
                    "Fill in STATE_SELECTORS with the correct CSS selectors."
                )
            text = page.inner_text(selector).strip()
            # The DOM includes units (e.g. "200.0 m", "15.0°", "0.039 m/s").
            # Extract just the leading numeric token (with optional sign).
            match = re.search(r"-?\d+\.?\d*", text)
            try:
                state[key] = float(match.group()) if match else 0.0
            except (ValueError, AttributeError):
                logger.warning(
                    "Could not parse '%s' for state key '%s'; defaulting to 0.0",
                    text,
                    key,
                )
                state[key] = 0.0
        return state

    # ------------------------------------------------------------------
    # Terminal-state helpers
    # ------------------------------------------------------------------

    def is_docked(self) -> bool:
        """Return ``True`` if the simulator shows a successful docking state.

        .. todo::
            Replace the body with the correct DOM check, e.g. detecting a
            success overlay element becoming visible.
        """
        return False  # TODO: implement DOM check for docking success

    def is_crashed(self) -> bool:
        """Return ``True`` if the simulator shows a collision/failure state.

        .. todo::
            Replace the body with the correct DOM check, e.g. detecting a
            failure banner or a red alert element becoming visible.
        """
        return False  # TODO: implement DOM check for crash/failure

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    PRELOADER_PERCENT_SELECTOR: str = "#preloader-percent"
    BEGIN_BUTTON_SELECTOR: str = "#begin-button"
    PRELOADER_POLL_INTERVAL_SECONDS: float = 1
    AFTER_LOAD_WAIT_SECONDS: float = 10.0
    AFTER_BEGIN_WAIT_SECONDS: float = 10.0
    BEGIN_CLICK_RETRY_INTERVAL_SECONDS: float = 0.5

    def _read_preloader_percent(self, page: Page, timeout_ms: int = 1000) -> tuple[int | None, str]:
        """Return parsed percent and raw text from #preloader-percent."""
        return BrowserStartupCoordinator.read_preloader_percent(
            self,
            page=page,
            timeout_ms=timeout_ms,
        )

    def _wait_for_begin_button_ready(self, page: Page, timeout_seconds: float) -> None:
        """Wait until begin button is visible/clickable in the DOM."""
        BrowserStartupCoordinator.wait_for_begin_button_ready(
            self,
            page=page,
            timeout_seconds=timeout_seconds,
        )

    def _click_begin_button_with_retries(self, page: Page, timeout_seconds: float) -> None:
        """Click begin with retries and JS fallback for transient click failures."""
        BrowserStartupCoordinator.click_begin_button_with_retries(
            self,
            page=page,
            timeout_seconds=timeout_seconds,
        )

    def _prepare_all_shared_tabs_before_training(self) -> None:
        """Run fixed startup flow for all shared tabs and block until all are ready."""
        BrowserStartupCoordinator.prepare_all_shared_tabs_before_training(self)

    def _wait_for_preloader_complete(self, timeout_seconds: float) -> None:
        """Wait until #preloader-percent reaches 100."""
        BrowserStartupCoordinator.wait_for_preloader_complete(
            self,
            timeout_seconds=timeout_seconds,
        )

    def _auto_start_simulator_if_needed(self) -> None:
        """Run deterministic managed-mode startup sequence."""
        BrowserStartupCoordinator.auto_start_simulator_if_needed(self)

    def _require_page(self) -> None:
        """Raise :class:`RuntimeError` if not yet connected."""
        if self._page is None:
            raise RuntimeError("Not connected. Call connect() first.")
