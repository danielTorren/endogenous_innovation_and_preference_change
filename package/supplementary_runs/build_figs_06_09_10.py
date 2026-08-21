"""
One-shot driver for the three supplementary figures that were still missing
after the 21/08/2026 cluster batch: S6 (10-panel local sensitivity), S9 (BAU
time series) and S10 (BAU elasticity).

Runs the two generation jobs they need, then builds all three figures straight
from the folders those jobs just produced -- so nothing has to be pasted into
build_figures.RUNS by hand between the simulation finishing and the PNGs
landing in docs/paper/supplementary_figs/.

  S6      <- fig06_panels.run_all(): the 9 missing vary_single sweeps (panel a
             is already on disk and is skipped), then the combine step.
             2,304 runs at 456 steps.
  S9, S10 <- fig09_10_bau_gen.run_fig9_10(): one decarb x elec-price grid,
             768 runs at 768 steps, plotted two different ways.

The folder names are taken from the Python return values, not scraped from
stdout, and handed to build_figures via its --set flag.

Deliberately sequential: the two generation steps each already saturate all 64
workers, so running them concurrently would only make them contend for the
same cores and the same memory ceiling.

Usage (normally via submit_figs_06_09_10.slurm):
    python -m package.supplementary_runs.build_figs_06_09_10
    python -m package.supplementary_runs.build_figs_06_09_10 --skip-fig06
    python -m package.supplementary_runs.build_figs_06_09_10 --build-only
"""
import argparse
import os
import subprocess
import sys

# Where build_figures puts the finished supplementary PNGs, and which figure
# each of our three keys lands as.
SUPP_DIR = os.path.join("docs", "paper", "supplementary_figs")
EXPECTED = {"S6": "Supp_Figure_6.png", "S9": "Supp_Figure_9.png",
            "S10": "Supp_Figure_10.png"}


def _banner(text):
    print("\n" + "=" * 72, flush=True)
    print(text, flush=True)
    print("=" * 72, flush=True)


def run_fig06():
    """The 9 missing local-sensitivity panels + combine. Returns 10 folders."""
    from package.supplementary_runs.fig06_panels import resolve_panels, run_all

    _banner("S6: local sensitivity -- running missing panels")
    _, failures = run_all()

    resolved, _stale = resolve_panels(quiet=True)
    folders = [folder for _letter, folder in resolved]
    missing = [letter for letter, folder in resolved if not folder]

    if failures:
        print(f"WARNING: panels that raised: {', '.join(failures)}", flush=True)
    if missing:
        # build_figures refuses a partial S6 rather than emitting a figure with
        # blank panels, so say so here instead of letting it fail opaquely.
        print(
            f"WARNING: {len(missing)}/10 panels still missing "
            f"({', '.join(missing)}) -- S6 cannot be built.",
            flush=True,
        )
        return None

    print("\nAll 10 panels present, in panel order:", flush=True)
    for letter, folder in resolved:
        print(f"  {letter}) {folder}", flush=True)
    return folders


def run_fig09_10():
    """The shared decarb x elec-price grid behind S9 and S10. Returns 1 folder."""
    from package.supplementary_runs.fig09_10_bau_gen import run_fig9_10

    _banner("S9 + S10: BAU decarb x elec-price grid")
    folder = run_fig9_10()
    print(f"\nBAU grid folder: {folder}", flush=True)
    return folder


def build(local_sensitivity_folders, bau_grid_folder, extra_args=()):
    """
    Invoke build_figures for whichever of S6/S9/S10 we have folders for.

    A subprocess rather than an in-process call: it starts from a clean
    interpreter, so the plotting step is not sharing an address space with
    whatever the 3,000-run sweeps above left behind.
    """
    only, overrides = [], []
    if local_sensitivity_folders:
        only.append("S6")
        overrides += ["--set", "local_sensitivity=" + ",".join(local_sensitivity_folders)]
    if bau_grid_folder:
        only += ["S9", "S10"]
        overrides += ["--set", f"bau_grid={bau_grid_folder}"]

    if not only:
        print("\nNothing to build: no generation step produced a usable folder.",
              flush=True)
        return 1

    cmd = [sys.executable, "-m", "package.paper_figures.build_figures",
           "--only", ",".join(only), *overrides, *extra_args]

    # Stamp the destination PNGs before building. Checking mere existence
    # afterwards would not do: Supp_Figure_6.png is already on disk from an
    # earlier run, so a skipped S6 would still look like a success. Requiring a
    # NEWER file is what distinguishes "built" from "left alone".
    before = {}
    for key in only:
        path = os.path.join(SUPP_DIR, EXPECTED[key])
        before[key] = os.path.getmtime(path) if os.path.exists(path) else None

    _banner("Building " + ", ".join(only))
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd)

    # build_figures exits 0 even when it skips a figure for a missing folder,
    # which on the cluster would leave the job reporting success having
    # produced nothing. Check each PNG actually arrived, and is new.
    missing = []
    for key in only:
        path = os.path.join(SUPP_DIR, EXPECTED[key])
        if not os.path.exists(path):
            missing.append(f"{key} (no file)")
        elif before[key] is not None and os.path.getmtime(path) <= before[key]:
            missing.append(f"{key} (not rewritten -- build skipped it)")
    if missing:
        print(f"\nFAILED in {SUPP_DIR}/: {'; '.join(missing)}", flush=True)
        return rc or 1

    print("\nIn place:", flush=True)
    for key in only:
        path = os.path.join(SUPP_DIR, EXPECTED[key])
        print(f"  {key}: {path} ({os.path.getsize(path):,} bytes)", flush=True)
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--skip-fig06", action="store_true",
                        help="do not run the local-sensitivity panels (S9/S10 only)")
    parser.add_argument("--skip-fig09-10", action="store_true",
                        help="do not run the BAU grid (S6 only)")
    parser.add_argument("--build-only", action="store_true",
                        help="run no simulation; build from folders already on disk")
    parser.add_argument("--skip-plot", action="store_true",
                        help="passed through to build_figures: reuse existing PNGs")
    args = parser.parse_args(argv)

    panels = bau = None

    if args.build_only:
        from package.supplementary_runs.fig06_panels import resolve_panels

        resolved, _ = resolve_panels(quiet=True)
        if all(folder for _l, folder in resolved):
            panels = [folder for _l, folder in resolved]
        else:
            print("--build-only: not all 10 S6 panels are on disk, skipping S6.",
                  flush=True)
        print("--build-only: pass the BAU grid folder via build_figures --set "
              "bau_grid=... ; this driver only auto-detects S6 panels.", flush=True)
    else:
        if not args.skip_fig06:
            panels = run_fig06()
        if not args.skip_fig09_10:
            bau = run_fig09_10()

    extra = ["--skip-plot"] if args.skip_plot else []
    return build(panels, bau, extra_args=extra)


if __name__ == "__main__":
    sys.exit(main())
