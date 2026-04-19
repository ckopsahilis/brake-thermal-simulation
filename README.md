# F1 Brake Disc Heat Diffusion Simulator

A compact, fully local Python simulation of transient heat diffusion in an F1 brake disc. The model uses a vectorized Finite Difference Method (FDM) with an explicit FTCS scheme in polar coordinates to evolve the temperature field and render a 10-second MP4 animation.

Preview video:

<div align="center">
  <video src="https://github.com/user-attachments/assets/881b16d2-c350-45fb-bdae-188a210bc5c4" controls width="900"></video>
</div>

If the embedded player does not load, open the video directly:
[Watch preview video](https://github.com/user-attachments/assets/881b16d2-c350-45fb-bdae-188a210bc5c4)

## Physics & Math

The temperature field $T(r, \theta, t)$ is governed by the 2D heat equation in polar coordinates:

$$
\frac{\partial T}{\partial t} = a\left(\frac{\partial^2 T}{\partial r^2} + \frac{1}{r}\frac{\partial T}{\partial r} + \frac{1}{r^2}\frac{\partial^2 T}{\partial \theta^2}\right)
$$

The solver uses an explicit Forward-Time, Centered-Space (FTCS) update on a $(N_r, N_\theta)$ grid. The angular direction is periodic, so the implementation uses wrap-around indexing. The inner radial boundary is held at a fixed temperature, and the outer radial boundary applies a Neumann heat-flux condition using a ghost-node formulation.

This method is straightforward and fast for moderate grid sizes, but it is conditionally stable. The script includes a CFL-style stability warning if the timestep is too large for the chosen spatial resolution.

## Model Parameters

The simulation uses the following values:

- Radial domain: $r \in [0.05, 0.2]$ m
- Angular domain: $\theta \in [0, 2\pi]$ rad
- Thermal diffusivity: $a = 1.5 \times 10^{-4}$ m$^2$/s
- Initial temperature: $600$ K everywhere
- Applied heat flux: $q_{\mathrm{flux}} = 5.0 \times 10^5$ W/m$^2$
- Grid: $N_r = 30$, $N_\theta = 60$
- Time steps: $N_t = 400$
- Final simulation time: $t_{\max} = 10.0$ s

## Run the Simulation

Run the main script directly:

```bash
python main.py
```

This will:

1. Run the vectorized FTCS heat diffusion simulation in memory.
2. Render the final result as an MP4 animation named `brake_disc_10s.mp4`.
3. Use a local FFmpeg binary provided through `imageio-ffmpeg`, so no system-wide FFmpeg installation is required.

If you want to use the bundled Windows launcher, run:

```bash
run_sim.bat
```

## Output

The generated video shows the temperature evolution across the brake disc over the full 10-second simulation. The repository is configured to ignore MP4 files so large rendered artifacts do not get committed by accident.

## Repository Layout

- `main.py` - vectorized FTCS solver and animation export
- `requirements.txt` - runtime dependencies
- `.gitignore` - Python and output-file ignores
- `run_sim.bat` - Windows launcher for local execution

## Notes

- The solver is implemented with NumPy vectorized slicing and `np.roll` for the periodic angular direction.
- The animation is encoded locally through `imageio-ffmpeg` and saved as MP4.
- If the CFL warning appears, reduce the timestep or adjust the grid resolution before trusting the results.
