# app.py (Gradio version)

import gradio as gr
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_bmi(height, weight):
    return weight / ((height / 100) ** 2)

def normalize(x, mean, std):
    if x is None or pd.isna(x):
        return 0.0
    z = (float(x) - mean) / (std if std > 0 else 1.0)
    return float(np.clip(z, -3, 3))

def risk_model(age, height, weight, systolic, diastolic, glucose, hba1c, chol, hdl, ldl, smoker, fam, activity):
    bmi = compute_bmi(height, weight)
    n_age = normalize(age, 45, 12)
    n_bmi = normalize(bmi, 26, 5)
    n_sys = normalize(systolic, 125, 15)
    n_dia = normalize(diastolic, 80, 10)
    n_fbg = normalize(glucose, 95, 12)
    n_a1c = normalize(hba1c, 5.4, 0.5)
    n_chol = normalize(chol, 190, 35)
    n_hdl = -normalize(hdl, 55, 12)
    n_ldl = normalize(ldl, 120, 30)
    n_act = -normalize(activity, 120, 60)

    d_score = 0.9*n_bmi + 1.2*n_fbg + 1.2*n_a1c + 0.5*n_age + 0.5*n_act + 0.6*fam + 0.3*smoker - 0.2
    h_score = 0.8*n_age + 0.8*n_chol + 0.7*n_ldl + 0.6*n_sys + 0.3*n_dia + 0.8*smoker + 0.6*fam + 0.5*n_bmi + 0.5*n_act + 0.8*n_hdl - 0.3
    ht_score = 1.2*n_sys + 0.8*n_dia + 0.6*n_bmi + 0.6*n_age + 0.4*smoker + 0.5*fam + 0.4*n_act - 0.2

    diabetes = round(float(sigmoid(d_score)), 3)
    heart = round(float(sigmoid(h_score)), 3)
    hypertension = round(float(sigmoid(ht_score)), 3)

    return f"BMI: {bmi:.1f}", f"Diabetes Risk: {diabetes*100:.1f}%", f"Heart Disease Risk: {heart*100:.1f}%", f"Hypertension Risk: {hypertension*100:.1f}%"

demo = gr.Interface(
    fn=risk_model,
    inputs=[
        gr.Slider(18, 90, value=40, label="Age"),
        gr.Number(value=170, label="Height (cm)"),
        gr.Number(value=70, label="Weight (kg)"),
        gr.Number(value=120, label="Systolic BP"),
        gr.Number(value=80, label="Diastolic BP"),
        gr.Number(value=95, label="Fasting Glucose"),
        gr.Number(value=5.3, label="HbA1c"),
        gr.Number(value=180, label="Cholesterol"),
        gr.Number(value=50, label="HDL"),
        gr.Number(value=110, label="LDL"),
        gr.Checkbox(label="Smoker"),
        gr.Checkbox(label="Family History"),
        gr.Slider(0, 600, value=60, label="Activity Minutes/Week"),
    ],
    outputs=["text", "text", "text", "text"],
    title="Healthcare Assistant (Gradio Demo)"
)

if __name__ == "__main__":
    demo.launch()