export DATA_ROOT=/mercury/data3/jschueler1/OASIS_data/OASIS_tensors
export MODEL_ROOT=/mercury/data3/jschueler1/OASIS_data/OASIS_models/oasis_ablation
export RUN_ID=smoke_rc1_rn1_ro1
export DF_PATH=/mercury/data3/jschueler1/OASIS_data/OASIS_dataframes/hybrid_test.pkl

python -m OASIS.eval_entry \
  --run_id $RUN_ID \
  --model_root $MODEL_ROOT \
  --data_root $DATA_ROOT \
  --df_path $DF_PATH \
  --species hybrid
