import tempfile
import unittest
import warnings
from unittest.mock import patch
import types
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


class AnalysisVisualizationTests(unittest.TestCase):
    def setUp(self):
        self._original_show = go.Figure.show
        go.Figure.show = lambda *args, **kwargs: None

    def tearDown(self):
        go.Figure.show = self._original_show

    def test_scatter_accepts_marker_and_mode_options(self):
        from MAna.analysis import scatter

        frame = pd.DataFrame(
            {
                "age": [22, 34, 45, 52],
                "rating": [5, 4, 3, 5],
                "department": ["Tops", "Dresses", "Tops", "Bottoms"],
                "feedback": [1, 4, 2, 8],
            }
        )

        fig = scatter(
            frame,
            "age",
            "rating",
            color_col="department",
            size_col="feedback",
            mode="markers",
            symbol="circle",
            marker={"line": {"width": 1, "color": "black"}},
            auto_open=False,
        )

        self.assertGreaterEqual(len(fig.data), 1)

    def test_scatter_can_save_html(self):
        from MAna.analysis import scatter

        frame = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "scatter_example"
            scatter(frame, "x", "y", filename=str(output), auto_open=False)
            self.assertTrue(output.with_suffix(".html").exists())

    def test_returned_figure_helpers_can_skip_show(self):
        from MAna.analysis import funnel_chart

        calls = []
        go.Figure.show = lambda *args, **kwargs: calls.append("show")
        frame = pd.DataFrame({"stage": ["Visit", "Signup"], "count": [100, 25]})

        fig = funnel_chart(frame, "stage", "count", show=False)

        self.assertEqual(calls, [])
        self.assertGreaterEqual(len(fig.data), 1)

    def test_sunburst_supports_multi_level_hierarchy(self):
        from MAna.analysis import sunburst

        frame = pd.DataFrame(
            {
                "genre": ["Drama", "Drama", "Comedy"],
                "decade": ["1990s", "2000s", "1990s"],
                "language": ["en", "fr", "en"],
                "movies": [10, 6, 8],
                "rating": [7.1, 6.8, 6.5],
            }
        )

        fig = sunburst(
            frame,
            hierarchy_cols=["genre", "decade", "language"],
            size_col="movies",
            color_col="rating",
            show=False,
        )

        self.assertGreaterEqual(len(fig.data), 1)

    def test_dist_uses_non_deprecated_line_traces(self):
        from MAna.analysis import dist

        frame = pd.DataFrame({"value": [1, 2, 2, 3, 4]})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist(frame, "value")

        messages = [str(item.message) for item in caught]
        self.assertFalse(any("graph_objs.Line is deprecated" in message for message in messages))

    def test_pairplot_accepts_fixed_marker_size(self):
        from MAna.analysis import pairplot

        frame = pd.DataFrame(
            {
                "Age": [22, 34, 45],
                "Rating": [5, 4, 3],
                "Feedback": [1, 3, 8],
                "Group": ["A", "B", "A"],
            }
        )

        fig = pairplot(frame, color="Group", size=20, show=False)

        self.assertGreaterEqual(len(fig.data), 1)

    def test_dashboard_template_can_start_server_when_requested(self):
        from MAna.analysis import create_dashboard_template

        fake_dash = self._fake_dash_module()

        with patch.dict("sys.modules", {"dash": fake_dash}):
            app = create_dashboard_template(
                title="Live Dashboard",
                port=8060,
                debug=False,
                run=True,
            )

        self.assertEqual(app.run_calls, [{"host": "127.0.0.1", "port": 8060, "debug": False}])

    def test_quick_dashboard_can_start_server_when_requested(self):
        from MAna.analysis import quick_dashboard

        fake_dash = self._fake_dash_module()
        frame = pd.DataFrame(
            {
                "age": [22, 34, 45],
                "rating": [5, 4, 3],
                "department": ["Tops", "Dresses", "Tops"],
            }
        )

        with patch.dict("sys.modules", {"dash": fake_dash}):
            app = quick_dashboard(
                frame,
                numerical_cols=["age", "rating"],
                categorical_cols=["department"],
                port=8061,
                debug=False,
                run=True,
            )

        self.assertEqual(app.run_calls, [{"host": "127.0.0.1", "port": 8061, "debug": False}])

    def _fake_dash_module(self):
        class ComponentFactory:
            def __getattr__(self, name):
                def component(*children, **props):
                    return {"component": name, "children": children, "props": props}

                return component

        class FakeDash:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.layout = None
                self.run_calls = []

            def callback(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def run(self, **kwargs):
                self.run_calls.append(kwargs)

        fake_dash = types.SimpleNamespace(
            Dash=FakeDash,
            html=ComponentFactory(),
            dcc=ComponentFactory(),
            Input=lambda *args, **kwargs: ("Input", args, kwargs),
            Output=lambda *args, **kwargs: ("Output", args, kwargs),
        )
        return fake_dash


if __name__ == "__main__":
    unittest.main()
