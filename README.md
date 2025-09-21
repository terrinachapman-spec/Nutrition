
# TGK Meal Builder (Streamlit) - Regenerated

This repository contains a Streamlit prototype to build client meals from a food database (Excel).
It automatically scales selected foods (gram quantities) to approximate target macronutrients (protein, carbs, fat).

## Files
- `app.py` - Streamlit app.
- `requirements.txt` - Python dependencies.
- `concise-14-edition.xlsx` - **SAMPLE** minimal food database included so the app runs out-of-the-box. Replace this with your full Excel file when ready.
- `README.md` - this file.

## How to run locally
1. Create a virtualenv and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run Streamlit:
```bash
streamlit run app.py
```

3. Upload your full `concise-14-edition.xlsx` via the sidebar uploader or replace the sample file in the repo root.

## Deploy to Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and connect your GitHub repo.
3. Deploy the app.

## Notes & next steps
- The solver uses continuous grams; if you require whole-unit constraints (e.g., eggs, slices of bread) we can add integer programming.
- We can also add portion limits, saving client profiles, weekly planner, PDF export, and branded-food lookup (Nutritrack / FoodSwitch).
