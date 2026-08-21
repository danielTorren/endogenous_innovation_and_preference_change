"""
Runs the ten local-sensitivity sweeps behind supplementary Figure 6, one
parameter at a time, and combines whatever is on disk into the 10-panel figure.

fig06_local_sensitivity_gen.run_fig6() already does all ten in one go, but it
runs them as a single unbroken sequence: if one sweep dies (or the job hits its
--time limit) the panels that did finish are left in results/ with no record of
which parameter each one holds, and the whole thing has to be run again. That is
what happened to the run behind results/single_param_vary_14_11_40__19_08_2026,
which is panel a (alpha) and the only panel that survived.

So this module is the resumable version of the same work:

    python -m package.supplementary_runs.fig06_panels list      # what is on disk
    python -m package.supplementary_runs.fig06_panels run-all   # run what is missing, then combine
    python -m package.supplementary_runs.fig06_panels run e     # one panel
    python -m package.supplementary_runs.fig06_panels combine   # re-combine only

A panel folder is matched to its parameter by reading Data/vary_single.pkl and
comparing (subdict, property_varied, property_list) against the JSON config in
fig06_local_sensitivity_gen.PANELS -- the folder names are all
results/single_param_vary_<timestamp> and say nothing about the parameter. A run
whose value grid no longer matches its config is reported and not reused, so
editing a vary_single_*.json forces that panel to be re-run rather than quietly
plotting the old grid. The newest matching folder wins.

Panels are run against base_params_calibration.json, the same base parameters
run_fig6 uses (duration_future=0, the paper's Time Step axis stopping at ~456).
"""
import argparse
import glob
import json
import os
import sys

from package.generating_data.vary_single_param_gen import main as generate_vary_single
from package.resources.utility import load_object
from package.supplementary_runs.fig06_local_sensitivity_gen import BASE_PARAMS_LOAD, PANELS
from package.supplementary_runs.fig06_local_sensitivity_plot import (
    _display_name,
    plot_fig6_combined,
)

RESULTS_GLOB = "results/single_param_vary_*"


def _config(vary_load):
    with open(vary_load) as handle:
        return json.load(handle)


def _identity(vary_single):
    """What makes two runs the same sweep: the parameter and its value grid."""
    return (vary_single.get("subdict"), vary_single["property_varied"],
            tuple(vary_single["property_list"]))


def _existing_runs():
    """Every results/single_param_vary_* folder that can be identified, newest first."""
    runs = []
    for folder in glob.glob(RESULTS_GLOB):
        if not os.path.isdir(folder + "/Data"):
            continue
        try:
            vary_single = load_object(folder + "/Data", "vary_single")
            load_object(folder + "/Data", "data_array_ev_prop")
        except Exception:
            # A folder from a run that died before saving: not a usable panel.
            continue
        runs.append((os.path.getmtime(folder), folder.replace("\\", "/"), vary_single))
    return sorted(runs, reverse=True)


def resolve_panels(panels=PANELS, quiet=False):
    """
    [(letter, folder-or-None)] in panel order, plus the runs that look like a
    stale version of a panel (right parameter, different value grid).
    """
    runs = _existing_runs()
    resolved, stale = [], []

    for letter, vary_load in panels:
        config = _config(vary_load)
        wanted = _identity(config)
        match = next((folder for _, folder, vary in runs if _identity(vary) == wanted), None)
        if match is None:
            near = [folder for _, folder, vary in runs
                    if _identity(vary)[:2] == wanted[:2]]
            stale.extend((letter, folder) for folder in near)
        resolved.append((letter, match))
        if not quiet:
            name = _display_name(config)
            print("  %s) %-6s %s" % (letter, name, match if match else "MISSING -> " + vary_load))

    if stale and not quiet:
        print("\n  Ignored (same parameter, different value grid than the JSON config):")
        for letter, folder in stale:
            print("    %s) %s" % (letter, folder))

    return resolved, stale


def _combined_folder(first_folder):
    """Same derivation as run_fig6 and build_figures._local_sensitivity_output_folder."""
    return "results/" + os.path.basename(first_folder.rstrip("/")) + "_fig6_combined"


def combine(panels=PANELS):
    """Plot the 10-panel figure. Returns the combined folder, or None if incomplete."""
    print("Panels on disk:")
    resolved, _ = resolve_panels(panels)

    missing = [letter for letter, folder in resolved if folder is None]
    if missing:
        print("\nNot combining: %d panel(s) missing (%s). Run them first:"
              % (len(missing), ", ".join(missing)))
        print("  python -m package.supplementary_runs.fig06_panels run-all")
        return None

    panel_folders = [(letter, folder) for letter, folder in resolved]
    output_folder = _combined_folder(panel_folders[0][1])
    plot_fig6_combined(panel_folders, output_folder=output_folder)

    print("\nPaste into RUNS in package/paper_figures/build_figures.py:")
    print('    "local_sensitivity": [')
    for letter, folder in panel_folders:
        print('        "%s",  # %s' % (folder, letter))
    print("    ],")
    return output_folder


def run_panel(letter, base_params_load=BASE_PARAMS_LOAD, force=False, panels=PANELS):
    """Run one panel, reusing an existing folder for it unless force=True."""
    letters = dict(panels)
    if letter not in letters:
        raise KeyError("no panel %r; panels are %s" % (letter, ", ".join(l for l, _ in panels)))
    vary_load = letters[letter]
    config = _config(vary_load)
    name = _display_name(config)

    if not force:
        resolved, _ = resolve_panels([(letter, vary_load)], quiet=True)
        existing = resolved[0][1]
        if existing:
            print("=== Panel %s (%s): reusing %s" % (letter, name, existing))
            return existing

    runs = len(config["property_list"])
    print("=== Panel %s (%s): %d values x seeds, from %s"
          % (letter, name, runs, vary_load), flush=True)
    return generate_vary_single(BASE_PARAMS_LOAD=base_params_load, VARY_LOAD=vary_load)


def run_all(base_params_load=BASE_PARAMS_LOAD, panels=PANELS, force=False, do_combine=True):
    """
    Run every missing panel, then combine. A panel that raises is reported and
    the rest still run, so one bad sweep does not cost the other nine.
    """
    failures = []
    for letter, _ in panels:
        try:
            run_panel(letter, base_params_load=base_params_load, force=force, panels=panels)
        except Exception as error:
            print("=== Panel %s FAILED: %s: %s" % (letter, type(error).__name__, error),
                  flush=True)
            failures.append(letter)

    print()
    output_folder = combine(panels) if do_combine else None

    if failures:
        print("\nFailed panels: %s" % ", ".join(failures))
    return output_folder, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("command", choices=["list", "run-all", "run", "combine"])
    parser.add_argument("panel", nargs="?", help="panel letter a-j, for 'run'")
    parser.add_argument("--force", action="store_true",
                        help="re-run panels that already have a results folder")
    parser.add_argument("--no-combine", action="store_true",
                        help="run-all: skip the combining step")
    parser.add_argument("--base-params", default=BASE_PARAMS_LOAD)
    args = parser.parse_args(argv)

    if args.command == "list":
        print("Panels on disk:")
        resolved, _ = resolve_panels()
        missing = [letter for letter, folder in resolved if folder is None]
        print("\n%d/%d panels ready%s"
              % (len(resolved) - len(missing), len(resolved),
                 "" if not missing else "; missing " + ", ".join(missing)))
        return 0

    if args.command == "combine":
        return 0 if combine() else 1

    if args.command == "run":
        if not args.panel:
            parser.error("'run' needs a panel letter, e.g. 'run e'")
        run_panel(args.panel, base_params_load=args.base_params, force=args.force)
        return 0

    _, failures = run_all(base_params_load=args.base_params, force=args.force,
                          do_combine=not args.no_combine)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
