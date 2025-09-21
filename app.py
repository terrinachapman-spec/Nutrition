# TGK Meal Builder - Streamlit (Corrected)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import io, os

DEFAULT_EXCEL = "concise-14-edition.xlsx"

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
        return pd.DataFrame(columns=['name','brand','serving_grams','calories','protein_g','carbs_g','fat_g',
                                     'kcal_per_g','prot_per_g','carb_per_g','fat_per_g'])
    df = pd.read_excel(path, sheet_name=0)
    return normalize_foods_df(df)

def normalize_foods_df(df):
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
    protein_col = find_column(['protein','protein_g','protein (g)'])
    carbs_col = find_column(['carbs','carbohydrate','carbs_g','carbs (g)'])
    fat_col = find_column(['fat','fat_g','fat (g)'])
    serving_col = find_column(['serving_grams','serving_g','grams','serve_g'])

    out = pd.DataFrame()
    out['name'] = df[name_col].astype(str)
    out['brand'] = df[lc.get('brand','brand')] if 'brand' in lc else ""

    def get_num(col):
        if col and col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            return pd.Series(0.0, index=df.index)

    out['calories'] = get_num(calories_col)
    out['protein_g'] = get_num(protein_col)
    out['carbs_g'] = get_num(carbs_col)
    out['fat_g'] = get_num(fat_col)
    out['serving_grams'] = get_num(serving_col)

    def per_gram(val, serving):
        vals = []
        for v,s in zip(val,serving):
            if s>0:
                vals.append(v/s)
            else:
                vals.append(v/100.0)
        return pd.Series(vals, index=val.index)

    out['kcal_per_g'] = per_gram(out['calories'], out['serving_grams'])
    out['prot_per_g'] = per_gram(out['protein_g'], out['serving_grams'])
    out['carb_per_g'] = per_gram(out['carbs_g'], out['serving_grams'])
    out['fat_per_g'] = per_gram(out['fat_g'], out['serving_grams'])

    return out.reset_index(drop=True)

def solve_quantities(selected_df, target_prot, target_carb, target_fat, max_gram_per_food=1000):
    if selected_df.empty:
        return np.array([])
    A = np.vstack([
        selected_df['prot_per_g'].to_numpy(),
        selected_df['carb_per_g'].to_numpy(),
        selected_df['fat_per_g'].to_numpy()
    ]).T
    b = np.array([target_prot,target_carb,target_fat],dtype=float)
    res = lsq_linear(A,b,bounds=(0,max_gram_per_food))
    return np.maximum(res.x,0.0)

def compute_meal_totals(selected_df, grams):
    grams = np.array(grams)
    kcal = (selected_df['kcal_per_g']*grams).sum()
    prot = (selected_df['prot_per_g']*grams).sum()
    carb = (selected_df['carb_per_g']*grams).sum()
    fat = (selected_df['fat_per_g']*grams).sum()
    return {'kcal':kcal,'protein_g':prot,'carbs_g':carb,'fat_g':fat}

# ---------- UI ----------
st.set_page_config(page_title="TGK Meal Builder", layout="wide")
st.title("TGK Meal Builder — Streamlit")

uploaded_file = st.sidebar.file_uploader("Upload Excel food DB", type=["xlsx"])
if uploaded_file:
    foods = load_foods_from_fileobj(uploaded_file)
else:
    foods = load_foods_from_path(DEFAULT_EXCEL)

st.sidebar.header("Targets")
preset = st.sidebar.selectbox("Preset",["Manual","Breakfast (30C/23P/13F)","Lunch (35C/26P/9F)","Dinner (30C/20P/9F)","Snack (20C/20P/4F)"])

if "Breakfast" in preset: targets = (23,30,13)
elif "Lunch" in preset: targets = (26,35,9)
elif "Dinner" in preset: targets = (20,30,9)
elif "Snack" in preset: targets = (20,20,4)
else:
    p=st.sidebar.number_input("Protein g",25.0); c=st.sidebar.number_input("Carbs g",30.0); f=st.sidebar.number_input("Fat g",13.0)
    targets=(p,c,f)

st.header("1) Pick foods")
search = st.text_input("Search")
df=foods.copy()
if search:
    df=df[df['name'].str.contains(search,case=False,na=False)]
st.dataframe(df[['name','brand','calories','protein_g','carbs_g','fat_g','serving_grams']].head(100))

selected_idxs = st.multiselect(
    "Select foods", 
    options=list(df.index), 
    format_func=lambda i: f"{df.loc[i,'name']} ({df.loc[i,'brand']})"
)
selected_df = df.loc[selected_idxs].reset_index(drop=True)

st.header("2) Compute portions")
if st.button("Compute") and not selected_df.empty:
    grams = solve_quantities(selected_df,*targets)
    selected_df['grams']=grams.round(1)
    totals = compute_meal_totals(selected_df,grams)
    disp = selected_df[['name','grams']].copy()
    disp['kcal']=(selected_df['kcal_per_g']*grams).round(0)
    disp['protein_g']=(selected_df['prot_per_g']*grams).round(1)
    disp['carbs_g']=(selected_df['carb_per_g']*grams).round(1)
    disp['fat_g']=(selected_df['fat_per_g']*grams).round(1)
    st.dataframe(disp)
    st.write("Totals:",totals)
    csv_buf=io.StringIO();disp.to_csv(csv_buf,index=False)
    st.download_button("Download CSV",csv_buf.getvalue(),"meal.csv","text/csv")
