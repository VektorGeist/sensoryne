"""tests/test_viewer_page.py - the served viewer page is self-contained.

The page must fetch nothing at runtime and must not reference
the removed 3D library. The served bytes are the file the
viewer server serves at ``/`` and ``/index.html``.

The page is now one uniform software shell: a header with the
Sensoryne identity and a hamburger navigation drawer holding
three views (Dashboard, Sessions, Settings), exactly one view
visible at a time, and ONE scroll for the whole page. The old
three separately scrolling columns are gone.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import viewer as viewer_module
from stage2.viewer import _INDEX_PATH


def _read_page():
    with open(_INDEX_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


class ViewerPageSelfContainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = _read_page()

    def test_served_page_has_no_external_or_library_reference(self):
        page = self.page
        # No external script, stylesheet/link, or font source.
        self.assertIsNone(
            re.search(r"<script[^>]+src\s*=\s*[\"']https?://", page, re.I),
            "page loads an external script",
        )
        self.assertIsNone(
            re.search(r"<link[^>]+href\s*=\s*[\"']https?://", page, re.I),
            "page loads an external stylesheet/link",
        )
        self.assertIsNone(
            re.search(r"url\s*\(\s*[\"']?https?://", page, re.I),
            "page references an external font/image/resource",
        )
        self.assertIsNone(
            re.search(r"@import[^;]*https?://", page, re.I),
            "page imports an external resource",
        )
        self.assertIsNone(
            re.search(r"(src|href)\s*=\s*[\"']https?://", page, re.I),
            "page has an external resource reference",
        )
        # No three.js reference and no THREE usage anywhere.
        self.assertNotIn("three", page.lower())
        self.assertNotIn("THREE", page)


class ViewerPanelTests(unittest.TestCase):
    """The served page draws a panel for the seven applications that have one.

    Panels are keyed by application name: each panel function
    calls ``panelWrap("<application>", ...)`` (panelWrap emits
    ``data-panel="<application>"`` with
    ``id="panel-<application>"``), and renderVisualPanel
    dispatches on that same name.
    """

    APPLICATIONS = (
        "music",
        "typography",
        "drawing",
        "animation",
        "code",
        "poetry",
        "story",
        "argument",
    )

    @classmethod
    def setUpClass(cls):
        cls.page = _read_page()

    def test_served_page_draws_a_panel_for_each_application(self):
        # The panel wrapper is keyed by application name.
        self.assertIn('data-panel="', self.page)
        self.assertIn('id="panel-', self.page)
        for application in self.APPLICATIONS:
            self.assertIn(
                'panelWrap("%s",' % (application,),
                self.page,
                "no drawn panel keyed by application %r" % (application,),
            )
            self.assertIn(
                'appType==="%s"' % (application,),
                self.page,
                "no panel dispatch for application %r" % (application,),
            )


class ViewerShellTests(unittest.TestCase):
    """The page is one uniform shell: hamburger drawer + three views."""

    @classmethod
    def setUpClass(cls):
        cls.page = _read_page()

    def test_header_keeps_icon_name_and_tagline(self):
        page = self.page
        self.assertIn('src="sensoryne-icon.webp"', page)
        self.assertIn('href="sensoryne-icon.webp"', page)
        self.assertIn("Sensoryne", page)
        self.assertIn(
            "Independent Perceptual Evaluation and Improvement Platform", page
        )
        self.assertIn('id="current-view-name"', page)
        self.assertIn('id="selected-session"', page)

    def test_icon_appears_once_header_only_no_dash_icon(self):
        page = self.page
        self.assertNotIn("dash-icon", page)
        icon_imgs = re.findall(r'<img[^>]*src="sensoryne-icon\.webp"', page)
        self.assertEqual(len(icon_imgs), 1)
        self.assertIn(
            '<img class="brand-icon" src="sensoryne-icon.webp" '
            'alt="Sensoryne icon" width="44" height="44">',
            page,
        )
        self.assertIn('href="sensoryne-icon.webp"', page)

    def test_hamburger_drawer_holds_all_navigation(self):
        page = self.page
        self.assertIn('id="nav-toggle"', page)
        self.assertIn('aria-controls="nav-drawer"', page)
        self.assertIn('aria-expanded="false"', page)
        self.assertIn('id="nav-drawer"', page)
        self.assertIn('id="nav-backdrop"', page)
        self.assertIn('data-view="dashboard"', page)
        self.assertIn('data-view="sessions"', page)
        self.assertIn('data-view="settings"', page)
        # Drawer closes on Escape, on backdrop click, on selection,
        # and returns focus to the toggle button (the expanded-viewer
        # modal pattern).
        self.assertIn('e.key==="Escape"', page)
        self.assertIn('el("nav-backdrop").addEventListener("click"', page)
        self.assertIn("closeNav(true)", page)
        self.assertIn("t.focus()", page)

    def test_exactly_one_view_visible_and_dashboard_on_load(self):
        page = self.page
        self.assertIn('<section id="view-dashboard" class="view"', page)
        self.assertNotIn('<section id="view-dashboard" class="view" hidden', page)
        self.assertIn('<section id="view-sessions" class="view" hidden', page)
        self.assertIn('<section id="view-settings" class="view" hidden', page)
        self.assertIn('showView("dashboard")', page)
        # The page remembers nothing between loads.
        self.assertNotIn("localStorage", page)
        self.assertNotIn("sessionStorage", page)

    def test_dashboard_identity_sentence_cards_and_recent(self):
        page = self.page
        self.assertIn(
            "Sensoryne perceives a model's finished work, judges it against "
            "the work as written, and names what is wrong.",
            page,
        )
        lines = [
            ("Music &amp; Sound", "Judges a rendered piece against the written score, and sounds against their written events."),
            ("Poetry", "Judges a poem against its declared form."),
            ("Story", "Judges a story against its outline."),
            ("Drawing", "Judges a drawing against the shapes it declares."),
            ("Code", "Judges code against the checks it states."),
            ("Typography", "Judges a layout against its declared blocks."),
            (
                "Animation",
                "Judges an animation against its keyframes and timing.",
            ),
            ("Argument", "Judges an argument against its claims and sources."),
        ]
        for name, line in lines:
            self.assertIn("<h3>%s</h3>" % name, page)
            self.assertIn(line, page)
        self.assertEqual(page.count('<article class="app-card">'), 8)
        self.assertNotIn("<h3>Sound design</h3>", page)
        self.assertNotIn("sound_design", page)
        self.assertIn("Recent sessions", page)
        self.assertIn('id="recent-sessions"', page)
        self.assertIn("No sessions yet.", page)
        self.assertIn(
            "Open Sessions from the menu to look through recorded work, and "
            "Settings to connect the models.",
            page,
        )
        # A recent row opens the Sessions view with that session selected.
        self.assertIn('showView("sessions")', page)

    def test_settings_view_fields_status_and_note(self):
        page = self.page
        self.assertIn('<label for="worker-key">Working model API key</label>', page)
        self.assertIn(
            "The model that does the work Sensoryne judges.", page
        )
        self.assertIn(
            '<label for="judge-key">Judge model API key</label>', page
        )
        self.assertIn(
            "A separate model that reads a finished session's record and "
            "judges the work itself.",
            page,
        )
        self.assertEqual(page.count('type="password"'), 2)
        self.assertIn('id="settings-save"', page)
        self.assertIn('id="worker-key-remove"', page)
        self.assertIn('id="judge-key-remove"', page)
        self.assertIn("Key saved", page)
        self.assertIn("No key saved", page)
        self.assertIn(
            "Keys are stored on this machine, in the settings file beside "
            "the sessions folder. They are sent only to the model service "
            "when a session runs.",
            page,
        )


class ViewerOneScrollTests(unittest.TestCase):
    """One scroll for the whole page; only drawer and modal may scroll."""

    @classmethod
    def setUpClass(cls):
        cls.page = _read_page()
        match = re.search(r"<style>(.*?)</style>", cls.page, re.S)
        cls.css = match.group(1) if match else ""

    def test_old_three_column_grid_is_gone(self):
        self.assertNotIn("grid-template-areas", self.page)
        self.assertNotIn("grid-template-columns:300px", self.page)
        self.assertIn("#app{display:block", self.page)

    def test_layout_containers_do_not_scroll(self):
        for selector in ("#left", "#center", "#right"):
            for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", self.css):
                rule_selector, rule_body = match.groups()
                if selector in rule_selector:
                    self.assertNotIn(
                        "overflow-y",
                        rule_body,
                        "%s still scrolls vertically" % selector,
                    )
                    self.assertNotIn(
                        "overflow:auto",
                        rule_body,
                        "%s still scrolls" % selector,
                    )

    def test_only_drawer_and_expanded_modal_scroll_vertically(self):
        allowed = (".nav-drawer", ".expanded-modal")
        offenders = []
        for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", self.css):
            rule_selector, rule_body = match.groups()
            if re.search(r"overflow-y\s*:\s*(auto|scroll)", rule_body) or re.search(
                r"overflow\s*:\s*auto\b", rule_body
            ):
                if not any(name in rule_selector for name in allowed):
                    offenders.append(rule_selector.strip())
        self.assertEqual(offenders, [])


class ViewerScriptParseTests(unittest.TestCase):
    def test_inline_script_parses_with_node_vm(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not available")
        page = _read_page()
        scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
        self.assertEqual(len(scripts), 1, "expected exactly one inline script")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".js", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(scripts[0])
            script_path = handle.name
        try:
            runner = (
                "const vm=require('vm');const fs=require('fs');"
                "new vm.Script(fs.readFileSync(process.argv[1],'utf8'));"
                "console.log('PARSE OK');"
            )
            result = subprocess.run(
                [node, "-e", runner, script_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            os.unlink(script_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PARSE OK", result.stdout)


class ViewerSettingsEndpointTests(unittest.TestCase):
    """GET/POST /api/settings store keys beside the sessions folder."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sensoryne-viewer-test-")
        self.sessions_dir = os.path.join(self.tmp, "sessions")
        os.makedirs(self.sessions_dir)
        self.previous = viewer_module._SESSIONS_ROOT_OVERRIDE
        viewer_module._SESSIONS_ROOT_OVERRIDE = self.sessions_dir
        self.server = viewer_module.ThreadingHTTPServer(
            ("127.0.0.1", 0), viewer_module._Handler
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
        viewer_module._SESSIONS_ROOT_OVERRIDE = self.previous
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _request(self, method, path, payload=None):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8")

    def test_settings_round_trip_returns_booleans_only(self):
        settings_file = os.path.join(self.tmp, "sensoryne-settings.json")
        # Absence of the file is tolerated.
        self.assertFalse(os.path.exists(settings_file))
        status, body = self._request("GET", "/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(body),
            {"worker_key_set": False, "judge_key_set": False},
        )
        # Saving stores both keys; the file is created beside the
        # sessions folder (in its parent directory).
        status, body = self._request(
            "POST",
            "/api/settings",
            {"worker_key": "wk-secret-1", "judge_key": "jk-secret-2"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(body),
            {"worker_key_set": True, "judge_key_set": True},
        )
        self.assertTrue(os.path.isfile(settings_file))
        with open(settings_file, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored["worker_key"], "wk-secret-1")
        self.assertEqual(stored["judge_key"], "jk-secret-2")
        # Values are never sent back to the page.
        self.assertNotIn("wk-secret-1", body)
        self.assertNotIn("jk-secret-2", body)
        # An empty string leaves that key unchanged.
        status, body = self._request(
            "POST", "/api/settings", {"worker_key": ""}
        )
        self.assertEqual(
            json.loads(body),
            {"worker_key_set": True, "judge_key_set": True},
        )
        # Remove deletes one key only.
        status, body = self._request(
            "POST", "/api/settings", {"remove": "worker"}
        )
        self.assertEqual(
            json.loads(body),
            {"worker_key_set": False, "judge_key_set": True},
        )
        status, body = self._request("GET", "/api/settings")
        self.assertEqual(
            json.loads(body),
            {"worker_key_set": False, "judge_key_set": True},
        )
        # Existing routes still work, and unknown POST routes 404.
        status, _body = self._request("GET", "/api/sessions")
        self.assertEqual(status, 200)
        status, _body = self._request("GET", "/")
        self.assertEqual(status, 200)
        status, _body = self._request("POST", "/nope", {})
        self.assertEqual(status, 404)


class ViewerSettingsFileLocationTests(unittest.TestCase):
    def test_settings_file_is_beside_the_sessions_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = os.path.join(tmp, "nested", "sessions")
            os.makedirs(sessions_dir)
            previous = viewer_module._SESSIONS_ROOT_OVERRIDE
            viewer_module._SESSIONS_ROOT_OVERRIDE = sessions_dir
            try:
                path = viewer_module._settings_path()
            finally:
                viewer_module._SESSIONS_ROOT_OVERRIDE = previous
            self.assertEqual(
                path,
                os.path.join(tmp, "nested", "sensoryne-settings.json"),
            )


if __name__ == "__main__":
    unittest.main()
