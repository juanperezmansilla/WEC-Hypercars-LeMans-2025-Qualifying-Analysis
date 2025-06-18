# app.py

import pandas as pd
import numpy as np
from shiny import App, ui, render, reactive
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ALL_SESSIONS = pd.read_csv("C:/Users/juanp/Downloads/all_sessions.csv", delimiter=",")

features = [
    'LAP_NUMBER', 'S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT',
    'TOP_SPEED', 'TEAM', 'MANUFACTURER', 'DRIVER_NAME', 'SESSION', 'CLASS'
]
df = ALL_SESSIONS[features + ['LAP_IMPROVEMENT']].dropna()
df['IMPROVED'] = (df['LAP_IMPROVEMENT'] > 0).astype(int)

for col in ['TEAM', 'MANUFACTURER', 'DRIVER_NAME']:
    counts = df[col].value_counts()
    rare = counts[counts < 3].index
    df[col] = df[col].replace(rare, 'Other')

df_encoded = pd.get_dummies(df, columns=['TEAM', 'MANUFACTURER', 'DRIVER_NAME', 'SESSION', 'CLASS'])

X = df_encoded.drop(['LAP_IMPROVEMENT', 'IMPROVED'], axis=1)
y = df_encoded['IMPROVED']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# --- Shiny UI ---
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background: #f4f6fb; font-family: 'Segoe UI', Arial, sans-serif; }
        .shiny-input-container { margin-bottom: 18px; }
        h2 { color: #2d3a4a; font-weight: 700; letter-spacing: -1px; margin-bottom: 24px; }
        label { color: #2d3a4a; font-weight: 500; }
        .result { 
            background: #fff; 
            border-radius: 14px; 
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); 
            padding: 36px 24px; 
            margin-top: 32px; 
            text-align: center; 
            font-size: 1.4em; 
            color: #2d3a4a;
        }
        .prob { color: #4e6e8e; font-size: 1.1em; }
        .footer { margin-top: 40px; color: #aaa; font-size: 0.95em; text-align: center; }
    """),
    ui.h2("Predicción de Mejora de Vuelta (Random Forest)"),
    ui.input_numeric("lap_number", "Número de vuelta", 1, min=1, max=100),
    ui.input_numeric("s1_improvement", "Mejora S1", 0, min=-10, max=10),
    ui.input_numeric("s2_improvement", "Mejora S2", 0, min=-10, max=10),
    ui.input_numeric("s3_improvement", "Mejora S3", 0, min=-10, max=10),
    ui.input_numeric("top_speed", "Velocidad punta (km/h)", 300, min=100, max=400),
    ui.input_select("team", "Equipo", sorted(df['TEAM'].unique())),
    ui.input_select("manufacturer", "Fabricante", sorted(df['MANUFACTURER'].unique())),
    ui.input_select("driver", "Piloto", sorted(df['DRIVER_NAME'].unique())),
    ui.input_select("session", "Sesión", sorted(df['SESSION'].unique())),
    ui.input_select("car_class", "Clase", sorted(df['CLASS'].unique())),
    ui.hr(),
    ui.output_ui("prediction"),
    ui.div("App de ejemplo para predicción de mejora de vuelta. Desarrollado con Shiny para Python.", class_="footer")
)

def server(input, output, session):
    @reactive.Calc
    def pred():
        input_dict = {
            'LAP_NUMBER': input.lap_number(),
            'S1_IMPROVEMENT': input.s1_improvement(),
            'S2_IMPROVEMENT': input.s2_improvement(),
            'S3_IMPROVEMENT': input.s3_improvement(),
            'TOP_SPEED': input.top_speed(),
            'TEAM_' + input.team(): 1,
            'MANUFACTURER_' + input.manufacturer(): 1,
            'DRIVER_NAME_' + input.driver(): 1,
            'SESSION_' + input.session(): 1,
            'CLASS_' + input.car_class(): 1
        }
        input_df = pd.DataFrame([input_dict])
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]
        input_scaled = scaler.transform(input_df)
        rf_pred = rf.predict(input_scaled)[0]
        rf_proba = rf.predict_proba(input_scaled)[0][1]
        if rf_pred == 1:
            msg = "<div class='result'>🚀 <b>¡Mejora de vuelta!</b><br><span class='prob'>Probabilidad: {:.1f}%</span></div>".format(rf_proba*100)
        else:
            msg = "<div class='result'>⏱️ <b>No hay mejora</b><br><span class='prob'>Probabilidad: {:.1f}%</span></div>".format(rf_proba*100)
        return ui.HTML(msg)

    @output
    @render.ui
    def prediction():
        return pred()

app = App(app_ui, server)