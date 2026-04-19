import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio_ffmpeg


def run_simulation():
    # Parameters
    r_min = 0.05
    r_max = 0.2
    N_r = 30
    N_theta = 60
    N_t = 400
    t_max = 10.0

    a = 1.5e-4
    k = 50.0
    q_flux = 5.0e5
    T_init = 600.0

    # Discretization
    dr = (r_max - r_min) / (N_r - 1)
    dtheta = (2.0 * np.pi) / N_theta
    dt = t_max / (N_t - 1)

    inv_dr = 1.0 / dr
    inv_dr2 = inv_dr * inv_dr
    inv_dtheta2 = 1.0 / (dtheta * dtheta)

    # Conservative explicit FTCS stability check in polar coordinates.
    cfl_value = a * dt * (inv_dr2 + 1.0 / ((r_min * r_min) * (dtheta * dtheta)))
    if cfl_value > 0.5:
        print(
            f"WARNING: CFL stability condition violated (value={cfl_value:.6f}, required<=0.5). "
            "Results may be unstable."
        )

    # Grid and state arrays
    r = r_min + np.arange(N_r, dtype=float) * dr
    T = np.full((N_r, N_theta), T_init, dtype=float)
    T_next = T.copy()

    # Precompute radial factors for internal nodes (broadcast over theta)
    r_inner = r[1:-1][:, None]
    inv_r_inner = 1.0 / r_inner
    inv_r2_inner = inv_r_inner * inv_r_inner

    dTdr_outer = q_flux / k
    r_outer = r[-1]
    inv_r_outer = 1.0 / r_outer
    inv_r2_outer = inv_r_outer * inv_r_outer

    frames = [T.copy()]

    # Time integration (no spatial for-loops)
    for _ in range(N_t - 1):
        # Internal radial nodes: i = 1 .. N_r-2 (vectorized over i and j)
        T_i = T[1:-1, :]
        T_ip1 = T[2:, :]
        T_im1 = T[:-2, :]

        T_jp1 = np.roll(T_i, shift=-1, axis=1)
        T_jm1 = np.roll(T_i, shift=1, axis=1)

        dT_dr = (T_ip1 - T_im1) * (0.5 * inv_dr)
        d2T_dr2 = (T_ip1 - 2.0 * T_i + T_im1) * inv_dr2
        d2T_dtheta2 = (T_jp1 - 2.0 * T_i + T_jm1) * inv_dtheta2

        T_next[1:-1, :] = T_i + dt * a * (
            d2T_dr2 + inv_r_inner * dT_dr + inv_r2_inner * d2T_dtheta2
        )

        # Inner boundary (Dirichlet)
        T_next[0, :] = 600.0

        # Outer boundary (Neumann with ghost node, vectorized in theta)
        T_ghost = T[-2, :] + 2.0 * dr * dTdr_outer
        T_outer = T[-1, :]

        T_outer_jp1 = np.roll(T_outer, shift=-1)
        T_outer_jm1 = np.roll(T_outer, shift=1)

        dT_dr_outer = (T_ghost - T[-2, :]) * (0.5 * inv_dr)
        d2T_dr2_outer = (T_ghost - 2.0 * T_outer + T[-2, :]) * inv_dr2
        d2T_dtheta2_outer = (T_outer_jp1 - 2.0 * T_outer + T_outer_jm1) * inv_dtheta2

        T_next[-1, :] = T_outer + dt * a * (
            d2T_dr2_outer + inv_r_outer * dT_dr_outer + inv_r2_outer * d2T_dtheta2_outer
        )

        T, T_next = T_next, T

        frames.append(T.copy())

    return np.asarray(frames), r, dtheta, t_max


def save_temperature_video(frames, r, dtheta, t_max, output_path="brake_disc_10s.mp4"):
    # Force a project-local ffmpeg binary from the Python environment.
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe

    # Build a fixed output timeline so file duration is exactly t_max seconds.
    src_frames = frames.shape[0]
    fps = 30
    out_frames = max(2, int(round(t_max * fps)))
    sample_idx = np.linspace(0, src_frames - 1, out_frames).astype(int)
    frames_out = frames[sample_idx]
    n_frames = frames_out.shape[0]

    theta = np.arange(frames.shape[2], dtype=float) * dtheta
    Theta, R = np.meshgrid(theta, r)

    vmin = float(np.min(frames))
    vmax = float(np.max(frames))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_rorigin(0)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = None

    def draw_frame(frame_idx):
        nonlocal cbar
        ax.clear()
        ax.set_rorigin(0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"F1 Brake Disc Temperature Distribution (t = {frame_idx * (t_max / (n_frames - 1)):.2f} s)",
            pad=20,
        )
        contour = ax.contourf(
            Theta,
            R,
            frames_out[frame_idx],
            levels=100,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        if cbar is None:
            cbar = fig.colorbar(contour, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label("Temperature (K)")
        return []

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 / fps, blit=False)
    anim.save(output_path, writer="ffmpeg", fps=fps)
    print(f"Saved MP4: {output_path} ({n_frames / fps:.2f}s at {fps} fps)")
    plt.close(fig)


def main():
    frames, r, dtheta, t_max = run_simulation()
    save_temperature_video(frames, r, dtheta, t_max)


if __name__ == "__main__":
    main()
