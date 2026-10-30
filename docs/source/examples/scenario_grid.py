"""Run a matrix of pyDeltaRCM scenarios with the Preprocessor.

This script expands the matrix defined in ``scenario_grid.yaml`` into eight
jobs (subsidence rate x sea-level rise x bedload fraction), runs them
through the :obj:`~pyDeltaRCM.preprocessor.Preprocessor`, and prints a
table of metrics computed from the final model state of each job. It also
saves a figure comparing the final bed elevation (``eta``) of all eight
runs to ``scenario_grid_eta_maps.png``, next to this script.

Run it from the repository root, or from ``docs/source`` as::

    python examples/scenario_grid.py

"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figure export without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pyDeltaRCM  # noqa: E402


#: matrix axes, in the order they appear in the printed table
AXES = ("subsidence_rate", "SLR", "f_bedload")

#: short labels for the figure panel titles
AXIS_LABELS = {
    "subsidence_rate": "subs",
    "SLR": "SLR",
    "f_bedload": "fbl",
}

#: yaml configuration for the grid, located next to this script
GRID_YAML = Path(__file__).parent / "scenario_grid.yaml"

#: where the comparison figure is written (next to this script)
FIGURE_PATH = Path(__file__).parent / "scenario_grid_eta_maps.png"

#: number of updates per grid job, matching ``timesteps`` in the yaml
TIMESTEPS = 10


def run_grid():
    """Expand and run the scenario grid.

    The yaml is passed to the :obj:`~pyDeltaRCM.Preprocessor` unchanged,
    except that ``out_dir`` is overridden with a keyword argument to point
    the job outputs at a temporary directory (keyword arguments passed to
    the Preprocessor take precedence over values in the yaml file, and
    multi-job expansion requires an ``out_dir`` that does not yet exist).

    Jobs are run in serial (the default), so the model objects stay in
    this process and their final state can be read directly from the job
    list. Larger grids can be run with ``parallel: True`` (or an integer
    number of workers) in the yaml; note that each parallel worker then
    compiles the numba kernels on its own, which dominates the runtime of
    a grid this small.

    Returns
    -------
    pp : Preprocessor
        The preprocessor, with the grid completed. Each job's model is
        available at ``pp.job_list[i].deltamodel``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = pyDeltaRCM.Preprocessor(
            input_file=str(GRID_YAML),
            out_dir=str(Path(tmpdir) / "output"),
        )
        pp.run_jobs()

    return pp


def collect_metrics(pp):
    """Assemble one metrics record per grid job.

    Metrics are computed from the final state of each job's model: the
    mean and standard deviation of the final bed elevation ``eta``, the
    deposited sediment volume (the integral of the bed change since
    initialization), and the timestep ``dt`` derived from the inlet
    conditions.
    """
    rows = []
    for i, job in enumerate(pp.job_list):
        model = job.deltamodel
        config = pp.config_list[i]

        eta_change = model.eta - model.eta_init
        deposit_volume = np.sum(eta_change * model.dx**2)

        assert np.all(np.isfinite(model.eta)), (
            f"job {config['out_dir']} produced non-finite bed elevations"
        )

        row = {axis: config[axis] for axis in AXES}
        row["mean_eta"] = float(np.mean(model.eta))
        row["std_eta"] = float(np.std(model.eta))
        row["deposit_volume"] = float(deposit_volume)
        row["dt"] = float(model.dt)
        row["model"] = model
        rows.append(row)

    return rows


def format_table(rows):
    """Format the metrics as an aligned table string."""
    headers = (*AXES, "mean eta", "std eta", "deposit vol", "dt")
    formats = (".0e", ".0e", ".2f", ".2f", ".2f", ".2f", ".1f")
    widths = [
        max(len(header), 8) for header in headers
    ]  # field plus one space on each side

    def fmt(value, style):
        return f"{value:{style}}" if isinstance(value, float) else str(value)

    def make_row(values):
        cells = [
            f"{fmt(value, style):>{width}}"
            for value, style, width in zip(values, formats, widths)
        ]
        return "| " + " | ".join(cells) + " |"

    lines = ["|" + (sum(widths) + 3 * (len(widths) - 1) + 2) * "-" + "|"]
    lines.append(make_row(headers))
    lines.append(lines[0])
    for row in rows:
        values = (
            tuple(row[axis] for axis in AXES)
            + tuple(
                row[k] for k in ("mean_eta", "std_eta", "deposit_volume", "dt")
            )
        )
        lines.append(make_row(values))
    lines.append(lines[0])

    return "\n".join(lines)


def plot_eta_maps(rows):
    """Plot the final ``eta`` of each job, in one shared color scale."""
    fig, axs = plt.subplots(
        2, len(rows) // 2, figsize=(16, 7.0), constrained_layout=True
    )
    vmin = min(row["model"].eta.min() for row in rows)
    vmax = max(row["model"].eta.max() for row in rows)

    for ax, row in zip(axs.ravel(), rows):
        im = ax.imshow(
            row["model"].eta,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            cmap="cividis",
        )
        label = ", ".join(
            f"{AXIS_LABELS[axis]} {row[axis]:.0e}" for axis in AXES
        )
        ax.set_title(label, fontsize=9)

    fig.supxlabel("along-channel width (cells)", fontsize=9)
    fig.supylabel("downstream length (cells)", fontsize=9)
    cbar = fig.colorbar(im, ax=axs, shrink=0.8)
    cbar.set_label("final bed elevation eta (m)")

    fig.suptitle(
        "pyDeltaRCM scenario grid: "
        f"final bed elevation after {TIMESTEPS} updates"
    )
    fig.savefig(FIGURE_PATH, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    grid = run_grid()
    metrics = collect_metrics(grid)
    print(format_table(metrics))
    print("")
    plot_eta_maps(metrics)
    print(f"Final-eta maps written to {FIGURE_PATH.name}")
