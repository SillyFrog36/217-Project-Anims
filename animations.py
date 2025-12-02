# pyright: reportMissingImports=false
from manim import Scene, Axes, MathTex, UP, DOWN, LEFT, RIGHT, ORIGIN, Write, Create, FadeIn, FadeOut, ReplacementTransform, BLUE, YELLOW
from manim import Circle, RegularPolygon, Transform, WHITE, GREEN
from manim import VGroup, config
from manim import Polygon
from manim import VMobject, Dot, MoveAlongPath
import numpy as np
import math
from decimal import Decimal, getcontext


class TaylorApproximation(Scene):
    def taylor_sin(self, x, degree):
        total = 0.0
        for k in range((degree + 1) // 2):
            n = 2 * k + 1
            total += ((-1) ** k) * (x ** n) / math.factorial(n)
        return total

    def construct(self):
        # Axes and title
        axes = Axes(x_range=(-4, 4, 1), y_range=(-2, 2, 1))
        title = MathTex(r"f(x)=\sin(x)\text{ and Taylor polynomials at }0").scale(0.7).to_edge(UP)
        self.play(Write(title))
        self.play(Create(axes))

        # True function
        f_graph = axes.plot(lambda x: np.sin(x), x_range=(-4, 4), color=YELLOW)
        f_label = MathTex(r"\sin(x)").set_color(YELLOW).next_to(title, DOWN, aligned_edge=LEFT)
        self.play(Create(f_graph), FadeIn(f_label))

        # Sequence of Taylor polynomial approximations around 0
        degrees = [1, 3, 5, 7, 9, 11]
        approx_graph = None
        approx_label = None
        for d in degrees:
            poly_graph = axes.plot(lambda x, dd=d: self.taylor_sin(x, dd), x_range=(-4, 4), color=BLUE)
            label = MathTex(r"T_{%d}(x)" % d).set_color(BLUE).next_to(f_label, DOWN, aligned_edge=LEFT)

            if approx_graph is None:
                self.play(Create(poly_graph), FadeIn(label))
            else:
                self.play(ReplacementTransform(approx_graph, poly_graph), ReplacementTransform(approx_label, label))

            approx_graph = poly_graph
            approx_label = label
            self.wait(0.5)

        self.wait(1)

class NewtonPiApproximation(Scene):
    def pi_partial(self, n: int) -> Decimal:
        # Compute S_n = 6 * sum_{k=0}^n [(2k)! / (4^k (k!)^2 (2k+1))] * (1/2)^{2k+1}
        # Use sufficient precision for the requested n.
        getcontext().prec = max(40, 2 * n + 30)
        s = Decimal(0)
        two = Decimal(2)
        four = Decimal(4)
        for k in range(n + 1):
            num = Decimal(math.factorial(2 * k))
            den = (four ** k) * (Decimal(math.factorial(k)) ** 2) * Decimal(2 * k + 1)
            coeff = num / den
            s += coeff / (two ** (2 * k + 1))
        return Decimal(6) * s

    def construct(self):
        getcontext().prec = 60

        title = MathTex(r"\text{Newton's } \arcsin \text{ series for } \pi").scale(0.7).to_edge(UP)
        self.play(Write(title))

        series = MathTex(
            r"\arcsin x"
            r" = \sum_{k=0}^\infty \frac{(2k)!}{4^k (k!)^2 (2k+1)} x^{2k+1}"
        ).scale(0.8).next_to(title, DOWN)
        self.play(Write(series))

        sub = MathTex(
            r"\arcsin\!\left(\tfrac{1}{2}\right) = \tfrac{\pi}{6}"
        ).scale(0.9).next_to(series, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(sub))

        pi_eq = MathTex(
            r"\pi"
            r" = 6 \sum_{k=0}^\infty \frac{(2k)!}{4^k (k!)^2 (2k+1)}"
            r" \left(\tfrac{1}{2}\right)^{2k+1}"
        ).scale(0.8).next_to(sub, DOWN, aligned_edge=LEFT)
        self.play(Write(pi_eq))
        self.wait(0.5)

        # Reference value for comparison (shown once)
        pi_ref = MathTex(r"\pi \approx 3.141592653589793").scale(0.8).next_to(pi_eq, DOWN)
        self.play(FadeIn(pi_ref))

        # Animate partial sums approaching pi
        n_values = [0, 1, 2, 3, 5, 8, 12]
        approx_mobj = None
        for n in n_values:
            val = self.pi_partial(n)
            # Limit displayed digits for readability
            val_str = str(val)[:18]
            approx = MathTex(rf"S_{{{n}}} \approx {val_str}").scale(0.9).next_to(pi_ref, DOWN)
            if approx_mobj is None:
                self.play(FadeIn(approx))
            else:
                self.play(ReplacementTransform(approx_mobj, approx))
            approx_mobj = approx
            self.wait(0.4)

        self.wait(1)

class InscribedPolygonPiApproximation(Scene):
    def polygon_perimeter_unit(self, n: int) -> float:
        # Perimeter of a regular n-gon inscribed in a unit circle.
        return 2 * n * math.sin(math.pi / n)

    def construct(self):
        # Title
        title = MathTex(r"\text{Approximating } \pi \text{ with inscribed regular polygons}").scale(0.7).to_edge(UP)
        self.play(Write(title))

        # Draw circle (visual radius only; math uses unit circle)
        R_vis = 2.5
        circle = Circle(radius=R_vis, color=YELLOW).set_stroke(width=3)
        self.play(Create(circle))

        # First polygon
        n_values = [6, 8, 12, 24, 48, 96]
        n0 = n_values[0]
        poly = RegularPolygon(n=n0, radius=R_vis, color=BLUE).set_stroke(width=2)
        self.play(Create(poly))

        # Formula box
        formula = MathTex(r"P_n = 2n\sin\!\left(\tfrac{\pi}{n}\right),\quad \pi \approx \tfrac{P_n}{2}").scale(0.8)
        formula.next_to(circle, DOWN)
        self.play(FadeIn(formula))

        # Numeric approximation text (updates per n)
        def approx_tex(n: int):
            Pn = self.polygon_perimeter_unit(n)
            pi_est = Pn / 2.0
            return MathTex(rf"n={n}\ :\ P_n \approx {Pn:.6f},\ \ \pi \approx {pi_est:.6f}").scale(0.8).next_to(formula, DOWN)

        approx_mobj = approx_tex(n0)
        self.play(FadeIn(approx_mobj))

        # Iterate polygons with increasing n
        for n in n_values[1:]:
            new_poly = RegularPolygon(n=n, radius=R_vis, color=BLUE).set_stroke(width=2)
            new_approx = approx_tex(n)
            # Morph polygon and replace text
            self.play(Transform(poly, new_poly), ReplacementTransform(approx_mobj, new_approx))
            approx_mobj = new_approx
            self.wait(0.4)

        # Final comparison to pi
        pi_ref = MathTex(r"\pi \approx 3.141592653589793").scale(0.8).next_to(approx_mobj, DOWN)
        self.play(FadeIn(pi_ref))
        self.wait(1)

class MadhavaPiSeries(Scene):
    def leibniz_partial(self, n: int) -> Decimal:
        # π/4 = Σ (-1)^k/(2k+1)
        getcontext().prec = max(60, 2 * n + 40)
        s = Decimal(0)
        one = Decimal(1)
        for k in range(n + 1):
            term = one / Decimal(2 * k + 1)
            if k % 2 == 1:
                term = -term
            s += term
        return Decimal(4) * s

    def madhava_partial(self, n: int) -> Decimal:
        # π = √12 * Σ (-1)^k / ((2k+1) * 3^k)
        getcontext().prec = max(60, 2 * n + 40)
        s = Decimal(0)
        three = Decimal(3)
        for k in range(n + 1):
            term = Decimal(1) / (Decimal(2 * k + 1) * (three ** k))
            if k % 2 == 1:
                term = -term
            s += term
        return Decimal(12).sqrt() * s

    def construct(self):
        getcontext().prec = 80

        # Title and historical context
        title = MathTex(r"\text{Madhava of Sangamagrama (c.~1350)}").scale(0.8).to_edge(UP)
        subtitle = MathTex(r"\text{First infinite series for } \pi").scale(0.7).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))

        # arctan series
        atan_series = MathTex(
            r"\arctan x = \sum_{k=0}^\infty \frac{(-1)^k}{2k+1}\,x^{2k+1}"
        ).scale(0.9).next_to(subtitle, DOWN)
        self.play(Write(atan_series))

        # Two-column layout: left = Leibniz, right = Madhava
        leibniz_tex = MathTex(
            r"x=1:\quad \frac{\pi}{4} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots"
        ).scale(0.9)
        madhava_tex = MathTex(
            r"x=\tfrac{1}{\sqrt{3}}:\quad \pi = \sqrt{12}\Big(1 - \tfrac{1}{3\cdot 3} + \tfrac{1}{5\cdot 3^2} - \cdots\Big)"
        ).scale(0.9)

        left_label = MathTex(r"\text{Leibniz (}x=1\text{)}").set_color(BLUE)
        right_label = MathTex(r"\text{Madhava (}x=1/\sqrt{3}\text{)}").set_color(GREEN)

        # Move Leibniz label down by exactly one line
        left_col = VGroup(leibniz_tex, left_label).arrange(DOWN, aligned_edge=LEFT)
        right_col = VGroup(madhava_tex, right_label).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        # Force equal column widths to split the frame 50/50
        available = config.frame_width - 1.0  # side margins
        gap = 0.6
        col_width = (available - gap) / 2
        left_col.set_width(col_width)
        right_col.set_width(col_width)

        columns = VGroup(left_col, right_col).arrange(RIGHT, buff=gap, aligned_edge=UP)
        columns.next_to(atan_series, DOWN, buff=0.5)
        columns.to_edge(LEFT, buff=0.5)

        self.play(Write(leibniz_tex), Write(madhava_tex))
        self.play(FadeIn(left_label), FadeIn(right_label))
        self.wait(0.3)

        # Animate partial sums under each column
        n_values = [0, 1, 2, 3, 5, 10, 20, 50, 100]
        approxL_mobj = None
        approxM_mobj = None

        for n in n_values:
            valL = self.leibniz_partial(n)
            valM = self.madhava_partial(n)

            approxL = MathTex(rf"n={n}:\ \pi \approx {valL:.12f}").scale(0.75).next_to(left_col, DOWN, aligned_edge=LEFT)
            approxM = MathTex(rf"n={n}:\ \pi \approx {valM:.12f}").scale(0.75).next_to(right_col, DOWN, aligned_edge=LEFT)

            if approxL_mobj is None:
                self.play(FadeIn(approxL), FadeIn(approxM))
            else:
                self.play(ReplacementTransform(approxL_mobj, approxL), ReplacementTransform(approxM_mobj, approxM))

            approxL_mobj, approxM_mobj = approxL, approxM
            self.wait(0.3)

        # Reference value placed below both approximation texts and centered relative to the columns
        approx_group = VGroup(approxL_mobj, approxM_mobj)
        pi_ref = MathTex(r"\pi \approx 3.141592653589793").scale(0.8)
        pi_ref.next_to(approx_group, DOWN, buff=0.5)
        pi_ref.set_x(columns.get_center()[0])
        self.play(FadeIn(pi_ref))
        self.wait(1)

class TriangleParallelogramPiApproximation(Scene):
    def chord_length(self, n: int, R: float) -> float:
        return 2 * R * math.sin(math.pi / n)

    def make_circle_triangles(self, n: int, R: float, center: np.ndarray) -> VGroup:
        tris = VGroup()
        for k in range(n):
            th0 = 2 * math.pi * k / n
            th1 = 2 * math.pi * (k + 1) / n
            p0 = center + R * np.array([math.cos(th0), math.sin(th0), 0.0])
            p1 = center + R * np.array([math.cos(th1), math.sin(th1), 0.0])
            tri = Polygon(center, p0, p1, stroke_width=1, color=BLUE if k % 2 == 0 else GREEN, fill_opacity=0.6)
            tris.add(tri)
        return tris

    def make_row_triangles(self, n: int, R: float, center_x: float, base_y: float) -> VGroup:
        # Arrange alternating triangles into two interleaved rows forming a near-parallelogram:
        # - Use base length = arc length per slice (2πR/n) and height = R to avoid overlaps.
        base_len = 2 * math.pi * R / n
        cols = n // 2
        # Include half-base offset in total width so the group stays centered
        total_w = cols * base_len + base_len / 2.0
        start_x = center_x - total_w / 2.0

        tris = VGroup()
        # Upward-pointing triangles: base on base_y, apex at base_y + R
        for i in range(cols):
            x0 = start_x + i * base_len
            pA = np.array([x0, base_y, 0.0])
            pB = np.array([x0 + base_len, base_y, 0.0])
            pC = np.array([x0 + base_len / 2.0, base_y + R, 0.0])
            tri = Polygon(pA, pB, pC, stroke_width=1, color=BLUE, fill_opacity=0.6)
            tris.add(tri)
        # Downward-pointing triangles: offset by half base length; base on base_y + R, apex at base_y
        for i in range(cols):
            x0 = start_x + i * base_len + base_len / 2.0  # half-base horizontal offset
            pA = np.array([x0, base_y + R, 0.0])
            pB = np.array([x0 + base_len, base_y + R, 0.0])
            pC = np.array([x0 + base_len / 2.0, base_y, 0.0])
            tri = Polygon(pA, pB, pC, stroke_width=1, color=GREEN, fill_opacity=0.6)
            tris.add(tri)
        return tris

    def pi_estimate(self, n: int) -> float:
        # For r=1, arc-based triangle width tends to πR; with R=1, π ≈ n*sin(π/n)
        return n * math.sin(math.pi / n)

    def construct(self):
        # Two-column layout: left = circle with wedges, right = rearranged triangles (parallelogram-like)
        R_vis = 2.3
        left_x = -config.frame_width * 0.3
        right_x = +config.frame_width * 0.2
        row_base_y = -2.0

        title = MathTex(r"\text{Approximating } \pi \text{ by rearranging circle wedges}").scale(0.7).to_edge(UP)
        self.play(Write(title))

        # Place circle just below the title (no overlap)
        center_y = title.get_bottom()[1] - 0.4 - R_vis  # 0.4 = visual gap
        circle_center = np.array([left_x, center_y, 0.0])

        # Left column: circle + wedges
        circle = Circle(radius=R_vis, color=YELLOW).set_stroke(width=3).move_to(circle_center)
        self.play(Create(circle))

        n_values = [8, 12, 24, 48, 96, 192]
        n0 = n_values[0]

        wedges_left = self.make_circle_triangles(n0, R_vis, circle_center)
        self.play(FadeIn(wedges_left))

        # Right column: start by transforming a copy of wedges into the row (no overlaps, arc-length bases)
        wedges_to_move = wedges_left.copy()
        self.add(wedges_to_move)

        row_group = self.make_row_triangles(n0, R_vis, center_x=right_x, base_y=row_base_y)
        self.play(Transform(wedges_to_move, row_group))

        # Formula and first numeric approximation (assuming r=1)
        formula = MathTex(r"\pi \approx \frac{P_n}{2r} = n\,\sin\!\left(\tfrac{\pi}{n}\right)\ \ (r=1)").scale(0.8)
        formula.next_to(row_group, UP)
        self.play(Write(formula))

        approx = MathTex(rf"n={n0}:\ \pi \approx {self.pi_estimate(n0):.10f}").scale(0.8).next_to(row_group, DOWN)
        self.play(FadeIn(approx))

        # Increase n to improve the approximation on both columns
        current = wedges_to_move
        for n in n_values[1:]:
            new_row = self.make_row_triangles(n, R_vis, center_x=right_x, base_y=row_base_y)
            new_approx = MathTex(rf"n={n}:\ \pi \approx {self.pi_estimate(n):.10f}").scale(0.8).next_to(new_row, DOWN)
            new_wedges_left = self.make_circle_triangles(n, R_vis, circle_center)

            self.play(
                ReplacementTransform(current, new_row),
                ReplacementTransform(approx, new_approx),
                ReplacementTransform(wedges_left, new_wedges_left),
            )
            current = new_row
            approx = new_approx
            wedges_left = new_wedges_left
            self.wait(0.4)

        self.wait(1)

class RopePiEstimation(Scene):
    def build_rope_on_circle(self, center: np.ndarray, R: float, nseg: int, color=WHITE, width=6) -> VMobject:
        # Polyline approximating the circle with nseg short chords (closed loop).
        pts = []
        for i in range(nseg + 1):
            theta = 2 * math.pi * i / nseg
            pts.append(center + R * np.array([math.cos(theta), math.sin(theta), 0.0]))
        rope = VMobject().set_stroke(color=color, width=width)
        rope.set_points_as_corners(pts)
        return rope

    def build_straight_rope(self, x_center: float, y: float, R: float, nseg: int, color=WHITE, width=6) -> tuple[VMobject, float]:
        # Layout same number of segments end-to-end; each segment = chord length for circle partition.
        chord = 2 * R * math.sin(math.pi / nseg)
        total_len = nseg * chord
        x0 = x_center - total_len / 2.0
        pts = [np.array([x0 + i * chord, y, 0.0]) for i in range(nseg + 1)]
        straight = VMobject().set_stroke(color=color, width=width)
        straight.set_points_as_corners(pts)
        return straight, total_len

    def construct(self):
        title = MathTex(r"\text{Estimating } \pi \text{ with a rope around a circle}").scale(0.7).to_edge(UP)
        self.play(Write(title))

        # Circle placed under the title, no overlap
        R_vis = 2.2
        center_y = title.get_bottom()[1] - 0.8 - R_vis
        circle_center = np.array([0.0, center_y + 0.6, 0.0])  # lift a bit for space below
        circle = Circle(radius=R_vis, color=YELLOW).set_stroke(width=3).move_to(circle_center)
        self.play(Create(circle))

        # Rope as many tiny straight segments around the circle
        n0 = 192
        rope_circle = self.build_rope_on_circle(circle_center, R_vis, n0)
        tip = Dot(rope_circle.get_start()).set_color(WHITE)
        self.play(Create(rope_circle, run_time=2.0), MoveAlongPath(tip, rope_circle, run_time=2.0))

        # Straighten the rope into a line of equal length
        line_y = -config.frame_height * 0.35
        straight_rope, total_len = self.build_straight_rope(0.0, line_y, R_vis, n0)
        self.play(ReplacementTransform(rope_circle, straight_rope), FadeOut(tip))

        # Show numeric estimation: pi ≈ L / (2r)
        pi_est = total_len / (2.0 * R_vis)
        formula = MathTex(r"L \approx \text{circumference},\quad \pi \approx \frac{L}{2r}").scale(0.8)
        formula.next_to(straight_rope, UP)
        value = MathTex(rf"L \approx {total_len:.3f},\ \ r={R_vis:.2f},\ \ \pi \approx {pi_est:.6f}").scale(0.8)
        value.next_to(straight_rope, DOWN)
        self.play(FadeIn(formula), FadeIn(value))

        # Improve approximation by increasing segment count n
        for n in [192]:
            new_straight, new_len = self.build_straight_rope(0.0, line_y, R_vis, n)
            new_pi = new_len / (2.0 * R_vis)
            new_value = MathTex(rf"L \approx {new_len:.3f},\ \ r={R_vis:.2f},\ \ \pi \approx {new_pi:.6f}").scale(0.8)
            new_value.next_to(new_straight, DOWN)
            self.play(ReplacementTransform(straight_rope, new_straight), ReplacementTransform(value, new_value))
            straight_rope, value = new_straight, new_value
            self.wait(0.3)

        self.wait(1)

# --------------------
# Scene discovery + CLI
# --------------------
if __name__ == "__main__":
    import argparse
    import importlib.util
    import inspect
    import sys
    from pathlib import Path
    from typing import Dict, List, Type
    from manim import tempconfig

    BASE_DIR = Path(__file__).parent
    THIS_STEM = Path(__file__).stem

    def load_module_from_path(path: Path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        return module

    def iter_scene_classes_from_module(module) -> List[Type[Scene]]:
        # Use RENDER_SCENES if present; otherwise collect all Scene subclasses declared in the module.
        if hasattr(module, "RENDER_SCENES"):
            return [cls for cls in getattr(module, "RENDER_SCENES") if inspect.isclass(cls) and issubclass(cls, Scene)]
        classes = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Scene) and obj is not Scene and obj.__module__ == module.__name__:
                classes.append(obj)
        return classes

    def discover_scenes() -> Dict[str, Type[Scene]]:
        scenes: Dict[str, Type[Scene]] = {}
        # Include scenes from this file
        this_module = sys.modules[__name__]
        for cls in iter_scene_classes_from_module(this_module):
            key = f"{THIS_STEM}:{cls.__name__}"
            scenes[key] = cls
        # Include scenes from sibling .py files
        for py in BASE_DIR.glob("*.py"):
            if py.stem in {THIS_STEM, "__init__"}:
                continue
            module = load_module_from_path(py)
            if not module:
                continue
            for cls in iter_scene_classes_from_module(module):
                key = f"{py.stem}:{cls.__name__}"
                scenes[key] = cls
        return scenes

    def resolve_requested(scenes: Dict[str, Type[Scene]], requested: List[str]) -> List[Type[Scene]]:
        # Allow either "module:ClassName" or just "ClassName" (if unique).
        result: List[Type[Scene]] = []
        for name in requested:
            if name in scenes:
                result.append(scenes[name])
                continue
            # Try by class name
            matches = [cls for key, cls in scenes.items() if key.endswith(f":{name}")]
            if len(matches) == 1:
                result.append(matches[0])
            elif len(matches) == 0:
                print(f"[WARN] No scene matched '{name}'. Use --list to see available scenes.")
            else:
                print(f"[WARN] Ambiguous '{name}'. Use full name module:ClassName.")
        return result

    parser = argparse.ArgumentParser(description="Render Manim scenes discovered in this directory.")
    parser.add_argument("--list", action="store_true", help="List discovered scenes and exit.")
    parser.add_argument("--all", action="store_true", help="Render all discovered scenes.")
    parser.add_argument("--render", nargs="*", help="Specific scenes to render. Use module:ClassName or ClassName if unique.")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="low", help="Render quality.")
    parser.add_argument("--preview", action="store_true", default=True, help="Open preview window.")
    parser.add_argument("--no-preview", dest="preview", action="store_false", help="Disable preview window.")
    args = parser.parse_args()

    all_scouted = discover_scenes()

    if args.list:
        print("Discovered scenes:")
        for key in sorted(all_scouted.keys()):
            print(f"  {key}")
        sys.exit(0)

    targets: List[Type[Scene]] = []
    if args.all:
        targets = list(all_scouted.values())
    elif args.render:
        targets = resolve_requested(all_scouted, args.render)
    else:
        print("Nothing to render. Use --list, --all, or --render.")
        sys.exit(0)

    quality_map = {"low": "low_quality", "medium": "medium_quality", "high": "high_quality"}
    with tempconfig({"quality": quality_map[args.quality], "preview": args.preview}):
        for cls in targets:
            print(f"[INFO] Rendering {cls.__module__}.{cls.__name__} ...")
            scene = cls()
            scene.render()





