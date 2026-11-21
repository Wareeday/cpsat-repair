import time
import itertools
from collections import defaultdict, deque

def solve(instance):
    start_time = time.time()
    TIME_LIMIT = 28.0  # safety margin

    jobs = instance["jobs"]
    n_machines = instance["n_machines"]
    n_jobs = len(jobs)

    # flatten operations, assign global ids
    ops = []  # (job, idx_in_job, machine, duration)
    job_next = []  # list of op ids for each job in order
    for j, job in enumerate(jobs):
        ids = []
        for k, (m, d) in enumerate(job):
            op_id = len(ops)
            ops.append((j, k, m, d))
            ids.append(op_id)
        job_next.append(ids)

    n_ops = len(ops)

    # job precedence edges (fixed)
    job_pred = [[] for _ in range(n_ops)]
    job_succ = [[] for _ in range(n_ops)]
    for ids in job_next:
        for a, b in zip(ids, ids[1:]):
            job_pred[b].append(a)
            job_succ[a].append(b)

    # operations per machine
    machine_ops = [[] for _ in range(n_machines)]
    for op_id, (_, _, m, _) in enumerate(ops):
        machine_ops[m].append(op_id)

    best_makespan = float('inf')
    best_starts = None

    # ---------- helper: compute earliest start times given extra edges ----------
    def compute_schedule(extra_edges):
        """extra_edges: list of (u,v) meaning u must finish before v starts."""
        indeg = [0] * n_ops
        adj = [[] for _ in range(n_ops)]
        # add job edges
        for v in range(n_ops):
            for u in job_pred[v]:
                adj[u].append(v)
                indeg[v] += 1
        # add extra edges
        for u, v in extra_edges:
            adj[u].append(v)
            indeg[v] += 1

        # topological order & longest path (earliest start)
        q = deque([i for i in range(n_ops) if indeg[i] == 0])
        est = [0] * n_ops
        processed = 0
        while q:
            u = q.popleft()
            processed += 1
            _, _, _, du = ops[u]
            finish_u = est[u] + du
            for v in adj[u]:
                if est[v] < finish_u:
                    est[v] = finish_u
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if processed != n_ops:  # cycle
            return None, None
        makespan = max(est[i] + ops[i][3] for i in range(n_ops))
        return makespan, est

    # ---------- exact enumeration when feasible ----------
    total_permutations = 1
    for m_ops in machine_ops:
        total_permutations *= math.factorial(len(m_ops))
    # if too huge, fall back to heuristic
    if total_permutations <= 5_000_000:
        # generate permutations per machine
        machine_perms = [list(itertools.permutations(m_ops)) for m_ops in machine_ops]

        # recursive product enumeration with pruning
        extra_edges = []

        def recurse(machine_idx):
            nonlocal best_makespan, best_starts
            # time check
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutError
            if machine_idx == n_machines:
                # evaluate full schedule
                makespan, starts = compute_schedule(extra_edges)
                if makespan is not None and makespan < best_makespan:
                    best_makespan = makespan
                    best_starts = starts
                return

            for perm in machine_perms[machine_idx]:
                # add edges for this permutation
                added = []
                for a, b in zip(perm, perm[1:]):
                    added.append((a, b))
                extra_edges.extend(added)

                # quick lower bound: longest path with current edges
                lb, _ = compute_schedule(extra_edges)
                if lb is not None and lb < best_makespan:
                    recurse(machine_idx + 1)

                # backtrack
                for _ in added:
                    extra_edges.pop()

        try:
            recurse(0)
        except TimeoutError:
            pass

    # ---------- heuristic (list scheduling) ----------
    if best_starts is None:
        # simple forward scheduling respecting ready times
        machine_ready = [0] * n_machines
        job_ready = [0] * n_jobs
        starts = [[0] * len(job) for job in jobs]

        for j in range(n_jobs):
            for k, (m, d) in enumerate(jobs[j]):
                ready = max(machine_ready[m], job_ready[j])
                starts[j][k] = ready
                finish = ready + d
                machine_ready[m] = finish
                job_ready[j] = finish

        # compute makespan
        best_makespan = max(job_ready)
        best_starts = []
        for j in range(n_jobs):
            best_starts.extend(starts[j])

        # convert to required nested list format
        nested = []
        idx = 0
        for j in range(n_jobs):
            opcnt = len(jobs[j])
            nested.append(best_starts[idx:idx+opcnt])
            idx += opcnt
        best_starts = nested

    else:
        # convert flat start list to nested list as required
        nested = []
        idx = 0
        for j in range(n_jobs):
            cnt = len(jobs[j])
            nested.append(best_starts[idx:idx+cnt])
            idx += cnt
        best_starts = nested

    return {"starts": best_starts}