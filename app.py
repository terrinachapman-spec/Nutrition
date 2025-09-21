# TGK Meal Builder - Streamlit (Regenerated)
# Robust Streamlit prototype that loads an Excel food DB and computes gram portions
# to approximate target meal macros using a constrained least-squares solver.
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import io, os

# ---------- Config ----------
DEFAULT_EXCEL = "concise-14-edition.xlsx"  # default path (repo root). Can upload via UI.

# ---------- Helpers ----------
@st.cache_data
def load_foods_from_fileobj(fileobj):
    fileobj.seek(0)
    df = pd.read_excel(fileobj, sheet_name=0)
    return normalize_foods_df(df)

@st.cache_data
def load_foods_from_path(path):
    if not os.path.exists(path):
        st.error(f"Food database not found at: {path}")
        return pd.DataFrame(columns=['name','brand','serving_grams','calories','protein_g','carbs_g','fat_g','kcal_per_g','prot_per_g','carb_per_g','fat_per_g'])
    df = pd.read_excel(path, sheet_name=0)
    return normalize_foods_df(df)

def normalize_foods_df(df):
    # Normalize column names to expected canonical columns.
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}

    def find_column(possible):
        for p in possible:
            if p in lc:
                return lc[p]
        return None

    name_col = find_column(['name','foodname','food name','description']) or df.columns[0]
    calories_col = find_column(['calories','kcal','energy (kcal)','energy_kcal'])
    protein_col = find_column(['protein','protein_g','protein (g)','protein_g_per_serving'])
    carbs_col = find_column(['carbs','carbohydrate','carbs_g','carbs (g)','carbohydrate_g'])
    fat_col = find_column(['fat','fat_g','fat (g)'])
    serving_col = find_column(['serving_grams','servinggrams','serving_g','serving (g)','grams','serve_g'])

    # Build canonical df
    out = pd.DataFrame()
    out['name'] = df[name_col].astype(str) if name_col in df.columns else df.iloc[:,0].astype(str)
    out['brand'] = df[lc.get('brand', 'brand')] if 'brand' in lc and lc['brand'] in df.columns else ('brand' in df.columns and df['brand']) if False else ""
    # safer: if 'brand' in df.columns use it else empty string
    if 'brand' in df.columns:
        out['brand'] = df['brand'].astype(str).fillna("")
    else:
        out['brand'] = ""

    # Numeric columns (fill missing with 0)
    def get_num(colname):
        if colname and colname in df.columns:
            return pd.to_numeric(df[colname], errors='coerce').fillna(0.0)
        else:
            return pd.Series(0.0, index=df.index)

    out['calories'] = get_num(calories_col)
    out['protein_g'] = get_num(protein_col)
    out['carbs_g'] = get_num(carbs_col)
    out['fat_g'] = get_num(fat_col)
    out['serving_grams'] = get_num(serving_col)

    # Compute per-gram values: if serving_grams > 0 use it, else assume values are per 100g -> divide by 100
    def per_gram(val_series, serving_series):
        res = []
        for v, s in zip(val_series.fillna(0.0), serving_series.fillna(0.0)):
            try:
                if float(s) > 0:
                    res.append(float(v) / float(s))
                elif float(v) > 0:
                    res.append(float(v) / 100.0)
                else:
                    res.append(0.0)
            except Exception:
                res.append(0.0)
        return pd.Series(res, index=val_series.index)

    out['kcal_per_g'] = per_gram(out['calories'], out['serving_grams'])
    out['prot_per_g'] = per_gram(out['protein_g'], out['serving_grams'])
    out['carb_per_g'] = per_gram(out['carbs_g'], out['serving_grams'])
    out['fat_per_g'] = per_gram(out['fat_g'], out['serving_grams'])

    # Reset index and return
    return out.reset_index(drop=True)

def solve_quantities(selected_df, target_prot, target_carb, target_fat, max_gram_per_food=1000):
    """
    Solve for grams per selected food to match target macros (g) using non-negative least squares.
    selected_df must contain prot_per_g, carb_per_g, fat_per_g columns.
    Returns grams array same length as selected_df.
    """
    if selected_df.shape[0] == 0:
        return np.array([])
    prot = selected_df['prot_per_g'].to_numpy(dtype=float)
    carb = selected_df['carb_per_g'].to_numpy(dtype=float)
    fat = selected_df['fat_per_g'].to_numpy(dtype=float)
    # Build A with shape (3, n_foods)
    A = np.vstack([prot, carb, fat])  # shape (3, n)
    b = np.array([float(target_prot), float(target_carb), float(target_fat)], dtype=float)
    # Solve min ||A x - b|| with x >= 0 and x <= max_gram_per_food
    try:
        res = lsq_linear(A, b, bounds=(0, max_gram_per_food), max_iter=2000)
        x = res.x
        if np.any(np.isnan(x)):
            x = np.maximum(np.nan_to_num(x, 0.0), 0.0)
    except Exception as e:
        # fallback: use ordinary least squares and clip
        try:
            x_ls, *_ = np.linalg.lstsq(A.T, b, rcond=None)
            x = np.clip(x_ls, 0, max_gram_per_food)
        except Exception:
            x = np.zeros(A.shape[1])
    return x

def compute_meal_totals(selected_df, grams):
    grams = np.array(grams, dtype=float)
    kcal = (selected_df['kcal_per_g'].to_numpy(dtype=float) * grams).sum()
    prot = (selected_df['prot_per_g'].to_numpy(dtype=float) * grams).sum()
    carb = (selected_df['carb_per_g'].to_numpy(dtype=float) * grams).sum()
    fat = (selected_df['fat_per_g'].to_numpy(dtype=float) * grams).sum()
    return {'kcal': float(kcal), 'protein_g': float(prot), 'carbs_g': float(carb), 'fat_g': float(fat)}


# ---------- Streamlit UI ----------
st.set_page_config(page_title="TGK Meal Builder", layout="wide")
st.title("TGK Meal Builder — Streamlit prototype (Regenerated)")

# Sidebar: file upload or use default
st.sidebar.header("Data & client settings")
uploaded_file = st.sidebar.file_uploader("Upload concise-14-edition.xlsx (optional)", type=["xlsx"])
if uploaded_file is not None:
    foods = load_foods_from_fileobj(uploaded_file)
    st.sidebar.success("Loaded uploaded foods file")
else:
    foods = load_foods_from_path(DEFAULT_EXCEL)

st.sidebar.markdown("---")
client_name = st.sidebar.text_input("Client name", value="Test Client")
st.sidebar.subheader("Preset meal templates (from example plan)")
preset = st.sidebar.selectbox("Choose template", ["Manual", "Breakfast (~395 kcal, 30C/23P/13F)", "Lunch (~435 kcal, 35C/26P/9F)", "Dinner (~435 kcal, 30C/20P/9F)", "Snack (~200 kcal, 20C/20P/4F)"])

if preset == "Breakfast (~395 kcal, 30C/23P/13F)":
    target_prot, target_carb, target_fat = 23.0, 30.0, 13.0
elif preset == "Lunch (~435 kcal, 35C/26P/9F)":
    target_prot, target_carb, target_fat = 26.0, 35.0, 9.0
elif preset == "Dinner (~435 kcal, 30C/20P/9F)":
    target_prot, target_carb, target_fat = 20.0, 30.0, 9.0
elif preset == "Snack (~200 kcal, 20C/20P/4F)":
    target_prot, target_carb, target_fat = 20.0, 20.0, 4.0
else:
    target_prot = st.sidebar.number_input("Target protein (g)", value=25.0, step=1.0)
    target_carb = st.sidebar.number_input("Target carbs (g)", value=30.0, step=1.0)
    target_fat = st.sidebar.number_input("Target fat (g)", value=13.0, step=0.5)

max_gram_per_food = st.sidebar.number_input("Max grams per food (to cap portions)", min_value=50, max_value=2000, value=1000, step=50)

st.sidebar.markdown("---")
st.sidebar.write("Foods loaded:", len(foods))

# Main UI: search & select foods
st.header("1) Pick foods for this meal")
search = st.text_input("Search foods by name or brand (case-insensitive)")
df = foods.copy()
if search:
    mask = df['name'].str.contains(search, case=False, na=False) | df['brand'].str.contains(search, case=False, na=False)
    df = df[mask]
st.write(f"Showing {len(df)} foods (filtered)")

# Display a small table and allow selection
if len(df) == 0:
    st.warning("No foods found. Upload a valid Excel file named 'concise-14-edition.xlsx' or use the uploader in the sidebar.")
else:
    st.dataframe(df[['name','brand','calories','protein_g','carbs_g','fat_g','serving_grams']].head(200))

selected_idxs = st.multiselect("Select foods to include (1-8 foods recommended):", options=list(df.index), format_func=lambda i: f\"{df.loc[i,'name']} ({df.loc[i,'brand']})\")
selected_df = df.loc[selected_idxs].reset_index(drop=True)

st.markdown('---')
st.markdown("### 2) Compute optimized gram portions to meet target macros")

if selected_df.empty:
    st.info("Pick at least one food from the list above to compute portions.")
else:
    st.write("Selected foods:")
    st.dataframe(selected_df[['name','brand','calories','protein_g','carbs_g','fat_g','serving_grams']])
    if st.button("Compute quantities"):
        grams = solve_quantities(selected_df, target_prot, target_carb, target_fat, max_gram_per_food=max_gram_per_food)
        grams = np.round(grams, 1)
        selected_df['grams'] = grams
        totals = compute_meal_totals(selected_df, grams)
        # Build display
        disp = selected_df[['name','grams']].copy()
        disp['kcal'] = (selected_df['kcal_per_g'] * grams).round(0)
        disp['protein_g'] = (selected_df['prot_per_g'] * grams).round(1)
        disp['carbs_g'] = (selected_df['carb_per_g'] * grams).round(1)
        disp['fat_g'] = (selected_df['fat_per_g'] * grams).round(1)
        st.subheader("Resulting portions (grams) and macro breakdown")
        st.dataframe(disp.style.format({'grams':'{:.1f}','kcal':'{:.0f}','protein_g':'{:.1f}','carbs_g':'{:.1f}','fat_g':'{:.1f}'}))

        st.markdown("**Totals (actual):**")
        st.write(f\"Calories: {totals['kcal']:.0f} kcal  •  Protein: {totals['protein_g']:.1f} g  •  Carbs: {totals['carbs_g']:.1f} g  •  Fat: {totals['fat_g']:.1f} g\")

        st.markdown("**Difference (target - actual):**")
        diffs = {'protein_diff_g': target_prot - totals['protein_g'], 'carb_diff_g': target_carb - totals['carbs_g'], 'fat_diff_g': target_fat - totals['fat_g']}
        st.write(diffs)

        # CSV download
        csv_buf = io.StringIO()
        out_csv = disp[['name','grams','kcal','protein_g','carbs_g','fat_g']].copy()
        out_csv.to_csv(csv_buf, index=False)
        st.download_button("Download meal CSV", csv_buf.getvalue(), file_name=f\"{client_name}_meal.csv\", mime=\"text/csv\")

st.markdown('---')
st.write(\"Notes:\") 
st.write(\"- This is a prototype solver that returns non-negative gram quantities to approximate the target grams of macronutrients.\") 
st.write(\"- For best results select at least one protein source and at least one carb or fat source.\") 
st.write(\"- If your Excel contains values per-100g, leave serving_grams blank or zero. If your Excel uses a serving size (e.g., 30g tub), include serving_grams so the app converts correctly.\")
