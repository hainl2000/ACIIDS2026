# ACIIDS2026: lncRNA–disease association prediction

Final model: masked VGAE → LDAGM, blended with a gradient-boosting stack over split-consistently cross-fitted pair features.

```bash
pip install -r requirements.txt
python final_model.py all            # 6 evaluation splits x 5 seeds; summaries in logs/final_model/
python final_model.py dataset2_v2026_cs 1
```

- Code: `final_model.py` (evaluation), `ldagm_branch.py`, `stack.py`, `data.py`, `LDAGM.py`, `config.py`.
- Data build: `build_updated_datasets.py` → `data_preprocessing.ipynb` → `build_eval_splits.py`.
