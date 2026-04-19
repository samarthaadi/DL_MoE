# Web Interface — Setup & Usage

The web interface consists of three static HTML pages that communicate with a local Flask API. No build step is required.

## Prerequisites

- Python 3.10+
- All dependencies installed: `pip install -r requirements.txt`
- At least one trained checkpoint in `results/checkpoints/` (B2–B4, R1–R9 are included in the repo)

## Step 1 — Start the API server

From the **repository root**:

```bash
python api.py
```

The server starts on `http://localhost:5000`. You should see:

```
 * Running on http://127.0.0.1:5000
```

Keep this terminal open while using the web pages.

## Step 2 — Open a web page

Open any of the three HTML files directly in your browser (no web server needed for the HTML):

| Page | File | What it does |
|------|------|--------------|
| **Analyzer** | `website/index.html` | Type a Hinglish sentence → see per-token LID/POS predictions and routing weights (α) |
| **Results** | `website/results.html` | Dashboard comparing all 17 experiments with mean ± std across seeds |
| **Analysis** | `website/analysis.html` | Interpretability dashboard — router α distributions, switch-point trajectory, CMI bucket stats, PNG figures |

Example (Linux):
```bash
xdg-open website/index.html
```

## Verifying the model is working

### Quick API check

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sentence": "mujhe bahut zyada hunger lag rahi hai", "task": "lid"}'
```

Expected response shape:
```json
{
  "words": ["mujhe", "bahut", "zyada", "hunger", "lag", "rahi", "hai"],
  "predictions": ["hi", "hi", "hi", "en", "hi", "hi", "hi"],
  "alpha": [0.72, 0.68, 0.71, 0.41, 0.65, 0.69, 0.70],
  "ensemble_weights": [...],
  "models_used": [...]
}
```

### Check available checkpoints

```bash
curl http://localhost:5000/checkpoints
```

Returns a list of all loaded checkpoints with their exp_id, task, seed, metric, and router type.

### Check aggregated metrics

```bash
curl http://localhost:5000/metrics
```

Returns the full `results/metrics/aggregated.json`.

### Check analysis stats (requires prior analysis run)

```bash
curl "http://localhost:5000/analysis/stats?exp=R1&task=lid"
```

Returns α statistics, switch-point trajectory, and CMI bucket data for the R1/LID checkpoint.

## Running analysis to populate the Analysis page

The Analysis page requires pre-computed stats. Run analysis on any MoE checkpoint:

```bash
python main.py --mode analysis \
  --checkpoint results/checkpoints/R2-lid-s42.pt \
  --exp_id R2
```

Stats and figures are saved to `results/figures/` and served automatically by the API.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on port 5000 | Make sure `python api.py` is running in another terminal |
| `No checkpoints found for task` | Check that `.pt` files exist in `results/checkpoints/` |
| Analysis page shows no experiments | Run `python main.py --mode analysis ...` first |
| CORS error in browser console | Ensure flask-cors is installed: `pip install flask-cors` |
| Models fail to load | Run `python setup_data.py` to pre-download HingBERT and RoBERTa |
