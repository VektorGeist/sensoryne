"""tests/test_viewer_page.py - the served viewer page is self-contained.

The page must fetch nothing at runtime and must not reference
the removed 3D library. The served bytes are the file the
viewer server serves at ``/`` and ``/index.html``.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.viewer import _INDEX_PATH


class ViewerPageSelfContainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(_INDEX_PATH, "r", encoding="utf-8") as handle:
            cls.page = handle.read()

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
    """The served page draws a panel for each application.

    Panels are keyed by application name: each panel function
    calls ``panelWrap("<application>", ...)`` (panelWrap emits
    ``data-panel="<application>"`` with
    ``id="panel-<application>"``), and renderVisualPanel
    dispatches on that same name.
    """

    APPLICATIONS = (
        "sound_design",
        "typography",
        "drawing",
        "animation",
        "code",
        "poetry",
        "story",
    )

    @classmethod
    def setUpClass(cls):
        with open(_INDEX_PATH, "r", encoding="utf-8") as handle:
            cls.page = handle.read()

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


if __name__ == "__main__":
    unittest.main()
