# Pi Approximation Animations

A collection of educational animations demonstrating various historical and mathematical methods for approximating π (pi) using the [Manim](https://www.manim.community/) animation library.

## Overview

This project contains 7 animated scenes that visualize different approaches to approximating π throughout history, from ancient geometric methods to infinite series discovered by mathematicians like Newton and Madhava.

## Scenes

### 1. **TaylorApproximation**
Visualizes how Taylor polynomial approximations of sin(x) centered at 0 converge to the true sine function. Shows polynomials of degrees 1, 3, 5, 7, 9, and 11.

### 2. **NewtonPiApproximation**
Demonstrates Newton's arcsin series method for computing π:
- Uses the series expansion of arcsin(x)
- Substitutes x = 1/2 to get π/6
- Shows partial sums converging to π with high precision

### 3. **InscribedPolygonPiApproximation**
Classic geometric method using inscribed regular polygons in a circle:
- Starts with a hexagon (6 sides)
- Progressively increases to 96 sides
- Shows how perimeter approaches circumference as n increases

### 4. **MadhavaPiSeries**
Compares two infinite series for π discovered by Madhava of Sangamagrama (c. 1350):
- **Leibniz series** (x=1): Slow convergence
- **Madhava series** (x=1/√3): Much faster convergence
- Side-by-side comparison showing convergence rates

### 5. **TriangleParallelogramPiApproximation**
Visual "proof without words" showing how circle wedges can be rearranged:
- Divides circle into triangular wedges
- Rearranges wedges into a near-parallelogram shape
- As the number of wedges increases, the shape approaches a rectangle with dimensions πr × r

### 6. **RopePiEstimation**
Intuitive physical method for estimating π:
- Wraps a "rope" around a circle (shown as many small segments)
- Straightens the rope into a line
- Measures length L and calculates π ≈ L/(2r)

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
manim
numpy
```

### Installation

1. **Install Manim Community Edition:**
   ```powershell
   pip install manim
   ```

2. **Install NumPy (usually comes with Manim):**
   ```powershell
   pip install numpy
   ```

3. **Additional Requirements:**
   - Manim requires LaTeX for rendering mathematical formulas
   - Windows users: Install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)
   - For detailed installation instructions, see the [Manim Installation Guide](https://docs.manim.community/en/stable/installation.html)

## Usage

### List Available Scenes
```powershell
python animations.py --list
```

### Render a Single Scene
```powershell
python animations.py --render TaylorApproximation
```

### Render Multiple Scenes
```powershell
python animations.py --render TaylorApproximation NewtonPiApproximation
```

### Render All Scenes
```powershell
python animations.py --all
```

### Quality Options
```powershell
# Low quality (fast, default)
python animations.py --render TaylorApproximation --quality low

# Medium quality
python animations.py --render TaylorApproximation --quality medium

# High quality (slow but best output)
python animations.py --render TaylorApproximation --quality high
```

### Preview Control
```powershell
# With preview (default)
python animations.py --render TaylorApproximation --preview

# Without preview
python animations.py --render TaylorApproximation --no-preview
```

### Combined Examples
```powershell
# Render all scenes in high quality without preview
python animations.py --all --quality high --no-preview

# Render specific scenes in medium quality
python animations.py --render MadhavaPiSeries InscribedPolygonPiApproximation --quality medium
```

## Output

Rendered videos are saved in the `media/videos/` directory by default, organized by scene name and quality level.

Example output path:
```
media/videos/animations/720p30/TaylorApproximation.mp4
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--list` | List all available scenes and exit |
| `--all` | Render all discovered scenes |
| `--render [SCENES...]` | Render specific scenes by name |
| `--quality {low,medium,high}` | Set render quality (default: low) |
| `--preview` | Open preview window after rendering (default: true) |
| `--no-preview` | Disable preview window |

## Educational Context

These animations are designed for educational purposes to:
- Illustrate the historical development of π approximation methods
- Show the difference between geometric and analytical approaches
- Demonstrate convergence rates of different series
- Provide visual intuition for abstract mathematical concepts

## Technical Notes

- **Precision**: The `NewtonPiApproximation` and `MadhavaPiSeries` scenes use Python's `Decimal` module for high-precision arithmetic
- **Performance**: Higher quality renders take significantly longer; use low quality for testing
- **Scene Discovery**: The script automatically discovers all Scene classes in the file and any sibling `.py` files

## Credits

- Built with [Manim Community Edition](https://www.manim.community/)
- Historical mathematical methods from various sources spanning ancient Greece to medieval India

## License

This project is provided for educational purposes.

## Troubleshooting

### Common Issues

**LaTeX not found:**
- Ensure LaTeX is installed and in your system PATH
- Try running: `manim --version` to check if Manim can find LaTeX

**Import errors:**
- Verify all dependencies are installed: `pip install manim numpy`
- Check Python version: `python --version` (should be 3.7+)

**Slow rendering:**
- Use `--quality low` for faster previews
- High-quality rendering can take several minutes per scene

**Scene not found:**
- Use `--list` to see exact scene names
- Scene names are case-sensitive

## Contributing

Feel free to add new π approximation methods or improve existing animations!
