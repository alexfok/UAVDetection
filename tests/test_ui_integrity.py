from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "web" / "annotator" / "index.html"
APP_PATH = PROJECT_ROOT / "web" / "annotator" / "app.js"
CSS_PATH = PROJECT_ROOT / "web" / "annotator" / "styles.css"


class IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.attrs_by_id: dict[str, dict[str, str]] = {}
        self.classes: list[str] = []
        self.buttons: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        if "id" in attr:
            self.ids.append(attr["id"])
            self.attrs_by_id[attr["id"]] = attr
            if tag == "button":
                self.buttons[attr["id"]] = attr.get("type", "")
        if "class" in attr:
            self.classes.extend(attr["class"].split())


def parse_index() -> IdCollector:
    parser = IdCollector()
    parser.feed(INDEX_PATH.read_text(encoding="utf-8"))
    return parser


class UiIntegrityTests(unittest.TestCase):
    def test_no_duplicate_dom_ids(self) -> None:
        parser = parse_index()
        duplicates = sorted({dom_id for dom_id in parser.ids if parser.ids.count(dom_id) > 1})
        self.assertEqual(duplicates, [])

    def test_redesigned_shell_contract_is_present(self) -> None:
        ids = set(parse_index().ids)
        required = {
            "workspace",
            "appBrand",
            "panelMediaToggle",
            "panelControlsToggle",
            "panelStatusToggle",
            "panelEventsToggle",
            "liveModeLiveButton",
            "liveModeAdvancedButton",
            "liveMinimalControls",
            "liveAdvancedControls",
            "liveCameraChipList",
            "liveConfChipValue",
            "liveSnapshotButton",
            "liveFullscreenButton",
            "liveStatusBar",
            "liveStatusState",
            "liveStatusCameras",
            "liveStatusFps",
            "liveStatusDetectFps",
            "liveStatusLatency",
            "liveStatusTracks",
            "liveStatusSource",
        }
        self.assertEqual(sorted(required - ids), [])

    def test_existing_behavioral_controls_remain_addressable(self) -> None:
        ids = set(parse_index().ids)
        required = {
            "folderInput",
            "scanButton",
            "mediaList",
            "annotationTab",
            "liveTab",
            "trainingTab",
            "annotationView",
            "liveView",
            "trainingView",
            "eventsPanel",
            "liveCameraSelect",
            "liveCameraProfileSelect",
            "liveSourceInput",
            "liveConfInput",
            "livePreviewFpsInput",
            "liveDetectionFpsInput",
            "liveFrameSkipInput",
            "liveImageSizeInput",
            "liveRecordInput",
            "liveRecordLabelsInput",
            "liveStartButton",
            "liveStopButton",
            "liveStreamGrid",
            "liveEventList",
            "liveVoiceInput",
            "liveVoiceTestButton",
            "diagnosticsRunButton",
            "diagnosticsPrivacySelect",
            "diagnosticsReportButton",
            "diagnosticsSysdumpButton",
            "diagnosticsCheckList",
            "debugSessionStateValue",
            "debugSessionIdValue",
        }
        self.assertEqual(sorted(required - ids), [])

    def test_diagnostics_privacy_uses_existing_live_advanced_controls(self) -> None:
        ids = set(parse_index().ids)
        self.assertIn("diagnosticsPrivacySelect", ids)
        self.assertNotIn("diagnosticsAdvancedToolbar", ids)
        self.assertNotIn("diagnosticsAdvancedToggle", ids)
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertLess(html.index('class="liveRunControls"'), html.index("diagnosticsPrivacySelect"))
        self.assertLess(html.index("diagnosticsPrivacySelect"), html.index("liveRecordInput"))

    def test_diagnostics_camera_field_is_read_only(self) -> None:
        parser = parse_index()
        attrs = parser.attrs_by_id["diagnosticsCameraInput"]
        self.assertIn("readonly", attrs)
        self.assertEqual(attrs.get("aria-readonly"), "true")
        self.assertEqual(attrs.get("tabindex"), "-1")
        self.assertIn("readonlyInput", attrs.get("class", "").split())

    def test_layout_persistence_and_live_facade_hooks_exist(self) -> None:
        app = APP_PATH.read_text(encoding="utf-8")
        for token in [
            'UI_LAYOUT_STORAGE_KEY = "uavUiLayout"',
            "function loadUiLayout",
            "function saveUiLayout",
            "function applyUiLayout",
            "function togglePanel",
            "function setLiveMode",
            "function renderLiveCameraChips",
            "function snapshotLiveFrame",
            "function requestLiveFullscreen",
            "function prepareLiveVoiceAudio",
            "function testLiveVoiceWarning",
            "function pollLiveVoiceEvents",
            "function handleLiveVoiceEvent",
            "function maybePlayLiveVoiceWarning",
            '"/api/live/audio/warning"',
            '"/api/live/audio/all-clear"',
            "localStorage.getItem(UI_LAYOUT_STORAGE_KEY)",
            "localStorage.setItem(UI_LAYOUT_STORAGE_KEY",
        ]:
            self.assertIn(token, app)

    def test_responsive_layout_rules_are_guarded(self) -> None:
        css = CSS_PATH.read_text(encoding="utf-8")
        for token in [
            ".layoutControls",
            ".panelToggle",
            ".liveModeSwitch",
            ".liveMinimalControls",
            ".liveStatusBar",
            ".liveStreamGrid[data-count=\"1\"]",
            "@media (max-width: 759px)",
            "@media (min-width: 760px) and (max-width: 1179px)",
        ]:
            self.assertIn(token, css)

    def test_buffered_preview_uses_server_negotiated_fps(self) -> None:
        app = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('response.headers.get("X-Stream-FPS")', app)
        self.assertIn("startPlaybackTimer", app)
        self.assertIn("previewFps * 1.5", app)
        self.assertIn("overdueFrames", app)
        self.assertIn("frames.splice(0, dropCount)", app)
        self.assertIn("nextFrameAt - performance.now()", app)
        self.assertNotIn("window.setInterval(async () =>", app)

    def test_clickable_buttons_declare_type_button(self) -> None:
        parser = parse_index()
        offenders = sorted(button_id for button_id, button_type in parser.buttons.items() if button_type != "button")
        self.assertEqual(offenders, [])

    def test_annotation_save_buttons_expose_shortcut_tooltips(self) -> None:
        parser = parse_index()
        save_attrs = parser.attrs_by_id["saveButton"]
        negative_attrs = parser.attrs_by_id["negativeButton"]
        self.assertEqual(save_attrs.get("aria-keyshortcuts"), "S")
        self.assertEqual(negative_attrs.get("aria-keyshortcuts"), "A 0")
        self.assertIn("(S)", save_attrs.get("title", ""))
        self.assertIn("A; 0 also works", negative_attrs.get("title", ""))

    def test_annotation_save_shortcuts_match_tooltips(self) -> None:
        app = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('return key === "s" || code === "KeyS";', app)
        self.assertIn('return key === "a" || code === "KeyA" || key === "0"', app)

    def test_app_get_element_ids_match_static_dom_except_known_optional(self) -> None:
        html_ids = set(parse_index().ids)
        app = APP_PATH.read_text(encoding="utf-8")
        app_ids = set(re.findall(r'document\.getElementById\("([^"]+)"\)', app))
        optional = {"diagnosticsTabButton"}
        self.assertEqual(sorted(app_ids - html_ids - optional), [])


if __name__ == "__main__":
    unittest.main()
