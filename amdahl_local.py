#!/usr/bin/env python3
"""
amdahl_local.py - measure Amdahl's Law speedup on ONE machine (no dependencies).

A fixed amount of CPU-bound work is split into a serial part (always runs on
one process) and a parallel part (divided among N processes). We time it for
N = 1, 2, 4, ... and compare the measured speedup with Amdahl's prediction:

    S(N) = 1 / ((1 - p) + p / N)

Usage:
    python3 amdahl_local.py                    # p = 0.90, N up to your core count
    python3 amdahl_local.py -p 0.5 0.9 0.95    # compare several parallel fractions
    python3 amdahl_local.py --max-procs 16 --work 30000000 --repeats 3
    python3 amdahl_local.py --plot             # save amdahl_local.png (needs matplotlib)
"""
import argparse
import multiprocessing as mp
import os
import time


def burn(iterations: int) -> int:
    """CPU-bound busy work (pure Python so it scales with cores, not libraries)."""
    x = 0
    for i in range(iterations):
        x += (i * i) % 7
    return x


def run_once(total_work: int, p: float, n_procs: int, pool) -> float:
    serial_work = int(total_work * (1 - p))
    parallel_work = total_work - serial_work
    chunk = parallel_work // n_procs

    start = time.perf_counter()
    burn(serial_work)                                   # serial part: one process
    if n_procs == 1:
        burn(parallel_work)
    else:
        pool.map(burn, [chunk] * n_procs)               # parallel part: N processes
    return time.perf_counter() - start


def measure(total_work, p, n_procs, repeats):
    # Pool start-up is excluded from timing so we measure the algorithm,
    # not process creation. Best-of-k reduces noise from other programs.
    if n_procs == 1:
        return min(run_once(total_work, p, 1, None) for _ in range(repeats))
    with mp.Pool(n_procs) as pool:
        pool.map(burn, [1000] * n_procs)                # warm up workers
        return min(run_once(total_work, p, n_procs, pool) for _ in range(repeats))


def amdahl(p, n):
    return 1.0 / ((1.0 - p) + p / n)


def proc_counts(max_procs):
    counts, n = [], 1
    while n < max_procs:
        counts.append(n)
        n *= 2
    counts.append(max_procs)
    return counts


def estimate_p(speedup, n):
    """Invert Amdahl's Law: the parallel fraction implied by one measurement."""
    if n == 1 or speedup <= 0:
        return float("nan")
    return (1 - 1 / speedup) / (1 - 1 / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--parallel-fraction", type=float, nargs="+", default=[0.90],
                    help="parallel fraction(s) between 0 and 1 (default: 0.90)")
    ap.add_argument("--max-procs", type=int, default=os.cpu_count() or 4,
                    help="largest process count to test (default: CPU count)")
    ap.add_argument("--work", type=int, default=20_000_000,
                    help="total loop iterations, i.e. problem size (default: 20,000,000)")
    ap.add_argument("--repeats", type=int, default=3, help="runs per point; best is kept (default: 3)")
    ap.add_argument("--plot", action="store_true", help="save amdahl_local.png (needs matplotlib)")
    args = ap.parse_args()

    for p in args.parallel_fraction:
        if not 0 <= p <= 1:
            ap.error("parallel fraction must be between 0 and 1")

    if args.max_procs > (os.cpu_count() or 1):
        print(f"WARNING: --max-procs {args.max_procs} exceeds the {os.cpu_count()} logical CPUs available; "
              "speedup will plateau at the CPU count.\n")
    counts = proc_counts(args.max_procs)
    results = {}
    print(f"Machine has {os.cpu_count()} logical CPUs | work = {args.work:,} iterations | best of {args.repeats}\n")

    for p in args.parallel_fraction:
        print(f"Parallel fraction p = {p:.2f}  (Amdahl ceiling = {amdahl(p, 10**9):.1f}x)")
        print(f"{'N':>4} {'time (s)':>10} {'measured':>10} {'Amdahl':>10} {'implied p':>10}")
        t1 = None
        rows = []
        for n in counts:
            t = measure(args.work, p, n, args.repeats)
            if t1 is None:
                t1 = t
            s = t1 / t
            rows.append((n, s))
            print(f"{n:>4} {t:>10.3f} {s:>9.2f}x {amdahl(p, n):>9.2f}x {estimate_p(s, n):>10.3f}")
        results[p] = rows
        print()

    print("Measured speedup falls a little below Amdahl's curve because real runs also pay for")
    print("process communication and shared hardware (hyper-threads, memory bandwidth, turbo).")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed; skipping plot (pip install matplotlib)")
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (p, rows) in enumerate(results.items()):
            color = f"C{i}"
            xs = [n for n, _ in rows]
            ax.plot(xs, [s for _, s in rows], "o-", color=color, label=f"measured p={p:.2f}")
            ax.plot(xs, [amdahl(p, n) for n in xs], "--", color=color, alpha=0.6, label=f"Amdahl p={p:.2f}")
        ax.set_xlabel("Processes (N)")
        ax.set_ylabel("Speedup (x)")
        ax.set_title("Amdahl's Law: measured vs. predicted")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("amdahl_local.png", dpi=150)
        print("\nSaved amdahl_local.png")


if __name__ == "__main__":
    main()
