# AI Load Test Toolkit

This repository provides Python scripts for load testing AI model APIs, aggregating results, and visualizing performance. It is designed for vLLM servers and OpenAI-compatible endpoints.

## Setup

1. **Install dependencies**
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. **Configure environment**
   - Edit files in `envs/` to set API endpoints, keys, and other variables as needed.

---

## How to Use the Scripts

### 1. Run Load Tests
Use `scripts/text_load_test.py` to send prompts to your AI API and record response times.

```powershell
python scripts/text_load_test.py --env envs/text_gpt.env --input inputs/texts/prompts.txt --output results/test_run.csv
```
- `--env` specifies the environment file with API settings
- `--input` is the file with prompts
- `--output` is the CSV file to save results

---

### 2. Aggregate Results
Use `scripts/aggregate.py` to combine multiple result CSVs into a summary file.

```powershell
python scripts/aggregate.py --base_path results/
```
- Aggregates all CSVs in the specified folder
- Produces `all_times.csv` with combined data

---

### 3. Plot Results
Use `scripts/plot.py` to visualize aggregated results.

```powershell
python scripts/plot.py --input results/all_times.csv --output results/plot.png
```
- `--input` is the aggregated CSV
- `--output` is the image file for the plot

---

## Notes

- Adjust environment files in `envs/` for different models or endpoints
- Scripts are designed for OpenAI-compatible APIs but can be adapted for others
- Aggregation and plotting scripts help analyze and visualize test results

---

## License

See `LICENSE` for details.
