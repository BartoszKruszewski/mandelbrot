# Mandelbrot Explorer

An interactive Mandelbrot set viewer built with vanilla HTML, CSS, and JavaScript. It renders the fractal on a canvas and lets you smoothly pan and zoom to explore intricate details.

Avalible at https://bartoszkruszewski.github.io/mandelbrot/

## Features

- Smooth zooming and panning
- Responsive full‑screen canvas
- Adjustable render quality (iteration depth)
- Clean, minimal UI

## Algorithm

This viewer uses **GPU perturbation** for deep zooms. A high‑precision reference orbit for the current center is computed on the CPU using `Decimal`, then packed into a 1‑D texture. The fragment shader iterates the **perturbation** delta around that orbit per pixel, which is much faster than recomputing the full orbit in high precision. All GPU math uses **double‑single** (hi/lo split) floats to extend precision. Rendering is two‑stage: a quick low‑res pass while interacting, then a tiled high‑res pass when idle.
