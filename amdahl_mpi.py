#!/usr/bin/env python3
"""
amdahl_mpi.py - measure Amdahl's Law speedup ACROSS NODES with MPI.

Each launch uses however many MPI ranks you start (N). Rank 0 does the serial
part alone while the other ranks wait; then every rank does 1/N of the parallel
part. Rank 0 appends one line to a CSV. Run it for several N, then print the
report:

    # laptop (needs an MPI install + mpi4py)
    mpirun -n 4 python3 amdahl_mpi.py run -p 0.9
    # cluster, via SLURM (see run_amdahl.sbatch)
    srun -N 2 -n 32 python3 amdahl_mpi.py run -p 0.9

    python3 amdahl_mpi.py report results.csv          # no MPI needed

Note: for a fair comparison, make sure N=1 is run too (it is the baseline).
"""
import argparse
import csv
import os
import socket
import sys


def burn(iterations: int) -> int:
    x = 0
    for i in range(iterations):
        x += (i * i) % 7
    return x


def amdahl(p, n):
    return 1.0 / ((1.0 - p) + p / n)


def cmd_run(args):
    from mpi4py import MPI  # imported here so `report` works without MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    node = socket.gethostname()
    nodes = len(set(comm.gather(node, root=0) or [])) if rank == 0 else 0

    serial_work = int(args.work * (1 - args.p))
    parallel_work = args.work - serial_work
    my_share = parallel_work // size

    best = float("inf")
    for _ in range(args.repeats):
        comm.Barrier()
        t0 = MPI.Wtime()
        if rank == 0:
            burn(serial_work)               # serial part: rank 0 only
        comm.Barrier()                      # everyone waits for the serial part
        burn(my_share)                      # parallel part: each rank does 1/N
        comm.Barrier()
        elapsed = MPI.Wtime() - t0
        best = min(best, comm.allreduce(elapsed, op=MPI.MAX))

    if rank == 0:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["p", "work", "ranks", "nodes", "seconds"])
            w.writerow([args.p, args.work, size, nodes, f"{best:.6f}"])
        print(f"p={args.p:.2f} ranks={size} nodes={nodes} time={best:.3f}s -> {args.out}")


def cmd_report(args):
    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        sys.exit("no data in " + args.csv)
    groups = {}
    for r in rows:
        key = (float(r["p"]), int(r["work"]))
        groups.setdefault(key, {})[int(r["ranks"])] = (float(r["seconds"]), int(r["nodes"]))
    for (p, work), by_n in sorted(groups.items()):
        if 1 not in by_n:
            print(f"p={p:.2f}: no N=1 baseline in the data, skipping")
            continue
        t1 = by_n[1][0]
        print(f"\nParallel fraction p = {p:.2f}  (work = {work:,}; Amdahl ceiling = {amdahl(p, 10**9):.1f}x)")
        print(f"{'ranks':>6} {'nodes':>6} {'time (s)':>10} {'measured':>10} {'Amdahl':>10} {'efficiency':>11}")
        for n in sorted(by_n):
            t, nodes = by_n[n]
            s = t1 / t
            print(f"{n:>6} {nodes:>6} {t:>10.3f} {s:>9.2f}x {amdahl(p, n):>9.2f}x {s / n:>10.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="time one launch (use mpirun/srun)")
    r.add_argument("-p", type=float, default=0.90, help="parallel fraction (default 0.90)")
    r.add_argument("--work", type=int, default=200_000_000, help="total iterations (default 200M)")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--out", default="results.csv")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report", help="summarize a results CSV")
    q.add_argument("csv")
    q.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    if args.cmd == "run" and not 0 <= args.p <= 1:
        ap.error("p must be between 0 and 1")
    args.fn(args)


if __name__ == "__main__":
    main()
