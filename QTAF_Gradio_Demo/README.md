# QTAF-Net · Pathology VQA Gradio Demonstration

This package contains a Gradio research demo for **nine fine-tuned PyTorch checkpoints**, including all three QTAF-Net ablations. The model classes were extracted from the original capstone notebooks, and each checkpoint was checked with strict `load_state_dict` loading.

## Quick start (Python 3.10+)

```bash
pip install -r requirements.txt
python prepare_vocab.py
python app.py
```

Open the local address printed by Gradio, usually `http://127.0.0.1:7860`.

**Important:** `prepare_vocab.py` downloads the public `flaviagiammarino/path-vqa` dataset from Hugging Face and reconstructs the exact training-only question vocabulary and top-1,000 open-answer labels. It requires internet access, disk space for the dataset, and may take several minutes. The app will not produce correct text predictions without the matching vocabulary. The script checks the expected clean-train count (19,648), question-vocabulary size (3,972), and open-answer class count (1,000). If any check fails, do not bypass it: investigate the dataset version or preprocessing.

## Files

- `app.py`: Gradio interface, inference, model comparison, heatmaps, research metrics
- `prepare_vocab.py`: reconstruction of training mappings from PathVQA
- `models/*.py`: model classes extracted from original training notebooks
- `models/*.pth`: nine fine-tuned trained checkpoints
- `requirements.txt`: dependencies

## Hugging Face Spaces

1. Create a new Space and select **Gradio** (or Docker if you need to customize system packages).
2. Upload the project files, including the `models` directory. Git LFS is recommended for large `.pth` files.
3. **Before deploying**, run `python prepare_vocab.py` on your own computer or Kaggle with internet and upload the resulting `vocab.json` into the Space root. This avoids a costly dataset download at startup.
4. Use `app.py` as the entry point. On CPU, comparing all nine models may take significant time; GPU hardware improves responsiveness.
5. Test a few known PathVQA examples against the original notebooks before presenting publicly.

## Academic limitations

- This is a research demonstration, **not a medical device**.
- The user manually selects Yes/No or Open-ended answer type, matching the oracle-routing evaluation setting. The type-prediction experiment is distinct.
- The displayed published accuracy values are **held-out aggregate metrics**, not accuracy on the currently uploaded example.
- Softmax confidence is not calibrated.
- Attention overlays should not be interpreted as clinical explanations.
- The exploratory 2,000-answer model is not included in the nine-model final evaluation.
- Model class loading has been verified, but the entire live app has **not** been end-to-end validated because the dataset-derived vocabulary is not yet available in this environment.
