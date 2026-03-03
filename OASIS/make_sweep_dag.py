# make_sweep_dag.py
from pathlib import Path

# ----------------- EDIT THESE -----------------
DATA_ROOT  = "/mercury/data3/jschueler1/OASIS_data/OASIS_tensors"
MODEL_ROOT = "/mercury/data3/jschueler1/OASIS_data/OASIS_models/oasis_ablation"
DF_PATH    = "/mercury/data3/jschueler1/OASIS_data/OASIS_dataframes/test_hybrid.feather"   # parquet/csv/pkl/feather accepted by eval_entry.py
SPECIES    = "hybrid"

SEED       = 42
EPOCHS     = 50
PATIENCE   = 10
BATCH_SIZE = 16
LR         = 2e-4
# ---------------------------------------------

RCS = [0.5, 1.0, 2.0]
RNS = [0.5, 1.0, 2.0]
ROS = [1.0, 2.0, 4.0, 8.0]

def fmt(x: float) -> str:
    s = str(x).replace(".", "p")
    if s.endswith("p0"):
        s = s[:-2]
    return s

def jid(run_id: str) -> str:
    # DAGMan node names must be simple: letters/numbers/underscore
    return "J_" + run_id.replace("-", "_")

def main():
    Path("condor_logs").mkdir(exist_ok=True)

    runs = []
    run_ids = []
    for rc in RCS:
        for rn in RNS:
            for ro in ROS:
                run_id = f"rc{fmt(rc)}_rn{fmt(rn)}_ro{fmt(ro)}"
                runs.append((rc, rn, ro, run_id))
                run_ids.append(run_id)

    # Write helper files (optional but handy)
    with open("runs.txt", "w") as f:
        for rc, rn, ro, run_id in runs:
            f.write(f"{rc} {rn} {ro} {run_id}\n")
    with open("run_ids.txt", "w") as f:
        for run_id in run_ids:
            f.write(f"{run_id}\n")

    # Write one big DAG: TRAIN -> EVAL per run
    dag_lines = []
    dag_lines.append("# sweep.dag (auto-generated)")
    dag_lines.append(f"# DATA_ROOT={DATA_ROOT}")
    dag_lines.append(f"# MODEL_ROOT={MODEL_ROOT}")
    dag_lines.append(f"# DF_PATH={DF_PATH}")
    dag_lines.append("")

    for rc, rn, ro, run_id in runs:
        train_node = jid("train_" + run_id)
        eval_node  = jid("eval_" + run_id)

        # TRAIN job
        dag_lines.append(f"JOB {train_node} train.sub")
        dag_lines.append(
            f'VARS {train_node} '
            f'run_id="{run_id}" rc="{rc}" rn="{rn}" ro="{ro}" '
            f'data_root="{DATA_ROOT}" model_root="{MODEL_ROOT}" '
            f'seed="{SEED}" epochs="{EPOCHS}" patience="{PATIENCE}" '
            f'batch_size="{BATCH_SIZE}" lr="{LR}"'
        )
        dag_lines.append("")

        # EVAL job
        dag_lines.append(f"JOB {eval_node} eval.sub")
        dag_lines.append(
            f'VARS {eval_node} '
            f'run_id="{run_id}" '
            f'data_root="{DATA_ROOT}" model_root="{MODEL_ROOT}" '
            f'df_path="{DF_PATH}" species="{SPECIES}"'
        )
        dag_lines.append("")

        # Dependency
        dag_lines.append(f"PARENT {train_node} CHILD {eval_node}")
        dag_lines.append("")

    with open("sweep.dag", "w") as f:
        f.write("\n".join(dag_lines))

    print(f"Wrote sweep.dag with {len(runs)} branches (TRAIN->EVAL).")
    print("Submit with: condor_submit_dag sweep.dag")

if __name__ == "__main__":
    main()
