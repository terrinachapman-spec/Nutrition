# TGK Weekly Nutrition Planner - Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import io, os, random

DEFAULT_EXCEL = "concise-14-edition.xlsx"

# ---------- Helpers ----------
def normalize_foods_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}
    def get(colnames, default=None):
        for c in colnames:
            if c in lc: return lc[c]
        return default
    out = pd.DataFrame()
    out["name"] = df[get(["name","foodname","description"],df.columns[0])].astype(str)
    out["brand"] = df[get(["brand"],None)] if get(["brand"],None) else ""
    def num(colnames):
        col=get(colnames)
        return pd.to_numeric(df[col],errors="coerce").fillna(0.0) if col else 0.0
    out["calories"]=num(["calories","kcal"])
    out["protein_g"]=num(["protein","protein_g"])
    out["carbs_g"]=num(["carbs","carbohydrate"])
    out["fat_g"]=num(["fat","fat_g"])
    out["serving_grams"]=num(["serving_grams","grams"])
    def per_gram(val,serv): return np.where(serv>0,val/serv,val/100.0)
    out["kcal_per_g"]=per_gram(out["calories"],out["serving_grams"])
    out["prot_per_g"]=per_gram(out["protein_g"],out["serving_grams"])
    out["carb_per_g"]=per_gram(out["carbs_g"],out["serving_grams"])
    out["fat_per_g"]=per_gram(out["fat_g"],out["serving_grams"])
    return out.reset_index(drop=True)

@st.cache_data
def load_foods(file=None, path=DEFAULT_EXCEL):
    if file: df=pd.read_excel(file, sheet_name=0)
    elif os.path.exists(path): df=pd.read_excel(path, sheet_name=0)
    else: return pd.DataFrame(columns=["name","brand","calories","protein_g","carbs_g","fat_g"])
    return normalize_foods_df(df)

def calc_bmr(weight, height, age, sex):
    if sex.lower().startswith("m"): return 10*weight+6.25*height-5*age+5
    else: return 10*weight+6.25*height-5*age-161

def activity_factor(level):
    return {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very active":1.9}[level]

def goal_adjust(cal,goal,week,total_weeks):
    if goal=="Shred":
        # gradually reduce
        return cal*(1-0.1 - 0.1*(week/total_weeks))
    if goal=="Gain" or goal=="Build muscle":
        return cal*(1+0.1+0.05*(week/total_weeks))
    if goal=="Performance": return cal*1.05
    return cal

def macro_split(weight,cal,style="Balanced"):
    protein=1.8*weight
    fat=(0.25*cal)/9
    carbs=(cal-(protein*4+fat*9))/4
    if style=="Low-carb": carbs=carbs*0.7; fat=(cal-(protein*4+carbs*4))/9
    if style=="High-protein": protein=2.2*weight; carbs=(cal-(protein*4+fat*9))/4
    return round(protein),round(carbs),round(fat)

def solve_portions(df,prot,carb,fat):
    if df.empty: return []
    A=np.vstack([df["prot_per_g"],df["carb_per_g"],df["fat_per_g"]]).T
    b=np.array([prot,carb,fat])
    res=lsq_linear(A,b,bounds=(0,1000))
    return np.round(res.x,1)

# ---------- UI ----------
st.set_page_config(page_title="TGK Weekly Nutrition Planner", layout="wide")
st.title("TGK Weekly Nutrition Planner")

# File
uploaded=st.sidebar.file_uploader("Upload Excel food DB",type="xlsx")
foods=load_foods(file=uploaded)

# Client intake
st.header("Client Intake")
col1,col2,col3=st.columns(3)
with col1:
    name=st.text_input("Name","Client")
    age=st.number_input("Age",18,80,30)
    sex=st.selectbox("Sex",["Male","Female"])
with col2:
    height=st.number_input("Height (cm)",140,220,170)
    weight=st.number_input("Weight (kg)",40,160,70)
with col3:
    activity=st.selectbox("Activity",["Sedentary","Light","Moderate","Active","Very active"])
    goal=st.selectbox("Goal",["Shred","Maintain","Gain","Build muscle","Performance"])
    style=st.selectbox("Diet style",["Balanced","High-protein","Low-carb","Plant-forward"])
    weeks=st.number_input("Timeframe (weeks)",4,52,8)

bmi=weight/((height/100)**2)
bmr=calc_bmr(weight,height,age,sex)
tdee=bmr*activity_factor(activity)

st.markdown(f"**BMI** {bmi:.1f}  |  **BMR** {bmr:.0f} kcal  |  **TDEE** {tdee:.0f} kcal")

# Preferences
st.header("Food Preferences")
excluded=st.multiselect("Exclude foods",list(foods["name"].unique()))
favoured=st.multiselect("Favoured foods",list(foods["name"].unique()))
filtered=foods[~foods["name"].isin(excluded)].reset_index(drop=True)

# Plan
st.header("Weekly Plan")
meal_targets=[("Breakfast",0.25),("Lunch",0.3),("Dinner",0.3),("Snack",0.15)]

weekly_plan={}
for week in range(1,weeks+1):
    week_days={}
    cal=goal_adjust(tdee,goal,week,weeks)
    prot,carb,fat=macro_split(weight,cal,style if style!="Plant-forward" else "Balanced")
    for day in range(1,8):
        day_meals={}
        for meal,frac in meal_targets:
            mp,mc,mf=int(prot*frac),int(carb*frac),int(fat*frac)
            sub=filtered.sample(min(3,len(filtered)))  # random foods
            grams=solve_portions(sub,mp,mc,mf)
            sub["grams"]=grams
            sub["kcal"]=(sub["kcal_per_g"]*grams).round(0)
            sub["protein_g"]=(sub["prot_per_g"]*grams).round(1)
            sub["carbs_g"]=(sub["carb_per_g"]*grams).round(1)
            sub["fat_g"]=(sub["fat_per_g"]*grams).round(1)
            day_meals[meal]=sub[["name","grams","kcal","protein_g","carbs_g","fat_g"]]
        week_days[f"Day {day}"]=day_meals
    weekly_plan[f"Week {week}"]=week_days

# Display
for week,days in weekly_plan.items():
    with st.expander(week,expanded=False):
        for day,meals in days.items():
            st.subheader(f"{day}")
            for meal,df in meals.items():
                st.write(meal)
                st.dataframe(df)

# Export
if st.button("Export CSV (Week 1)"):
    out=[]
    for day,meals in weekly_plan["Week 1"].items():
        for meal,df in meals.items():
            for _,row in df.iterrows():
                out.append([day,meal,row["name"],row["grams"],row["kcal"],row["protein_g"],row["carbs_g"],row["fat_g"]])
    outdf=pd.DataFrame(out,columns=["Day","Meal","Food","Grams","Kcal","Protein","Carbs","Fat"])
    csv_buf=io.StringIO()
    outdf.to_csv(csv_buf,index=False)
    st.download_button("Download Week 1 CSV",csv_buf.getvalue(),"week1_plan.csv","text/csv")
