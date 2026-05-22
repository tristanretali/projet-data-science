import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Maintenance Prédictive - Dashboard",
    page_icon="⚙️",
    layout="wide",
)

# ── Chargement des données ──────────────────────────────────────────────────


@st.cache_resource
def load_all():
    with open("models/results.pkl", "rb") as f:
        results = pickle.load(f)
    with open("data/preprocessed_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("models/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/logistic_regression.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("models/gradient_boosting.pkl", "rb") as f:
        gb = pickle.load(f)
    df = pd.read_csv("dataset/predictive_maintenance_v3.csv")
    return results, data, preprocessor, rf, lr, gb, df


results, data, preprocessor, rf, lr, gb, df = load_all()

X_test = data["X_test"]
y_test = data["y_test"]
comparison = results["comparison"]
feature_names = results["feature_names"]
feature_importance = results["feature_importance"]

MODEL_LABELS = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "mlp": "Deep Learning (MLP)",
}
COLORS = {
    "Logistic Regression": "#636EFA",
    "Random Forest": "#00CC96",
    "Gradient Boosting": "#EF553B",
    "Deep Learning (MLP)": "#AB63FA",
}

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/EFREI_Visual_Identity.svg/320px-EFREI_Visual_Identity.svg.png",
    width=160,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Vue d'ensemble",
        "Comparaison des modèles",
        "Interprétabilité",
        "Prédiction en temps réel",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Meilleur modèle :** Random Forest  \n"
    "**Tâche :** Classification binaire  \n"
    "**Cible :** `failure_within_24h`"
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

if page == "Vue d'ensemble":
    st.title("⚙️ Maintenance Prédictive Industrielle")
    st.markdown("### Vue d'ensemble du dataset")

    # KPI cards
    total = len(df)
    n_failure = int(df["failure_within_24h"].sum())
    pct_failure = n_failure / total * 100
    n_machines = df["machine_id"].nunique()
    n_features = 6

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enregistrements", f"{total:,}")
    c2.metric("Pannes détectées", f"{n_failure:,}", f"{pct_failure:.1f}% du total")
    c3.metric("Machines uniques", n_machines)
    c4.metric("Capteurs utilisés", n_features)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Équilibre des classes")
        labels = ["No Failure (85.2%)", "Failure (14.8%)"]
        values = [
            int((df["failure_within_24h"] == 0).sum()),
            int((df["failure_within_24h"] == 1).sum()),
        ]
        fig_pie = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker_colors=["#00CC96", "#EF553B"],
                textinfo="label+percent",
            )
        )
        fig_pie.update_layout(showlegend=False, height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Répartition par type de machine")
        machine_counts = df["machine_type"].value_counts().reset_index()
        machine_counts.columns = ["Type", "Count"]
        fig_bar = px.bar(
            machine_counts,
            x="Type",
            y="Count",
            color="Type",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Distribution des capteurs par classe")

    sensors = [
        "vibration_rms",
        "temperature_motor",
        "pressure_level",
        "rpm",
        "current_phase_avg",
        "ambient_temp",
    ]
    sensor_choice = st.selectbox("Choisir un capteur", sensors)

    df_plot = df[[sensor_choice, "failure_within_24h"]].dropna()
    df_plot["Classe"] = df_plot["failure_within_24h"].map(
        {0: "No Failure", 1: "Failure"}
    )

    fig_hist = px.histogram(
        df_plot,
        x=sensor_choice,
        color="Classe",
        barmode="overlay",
        opacity=0.7,
        nbins=50,
        color_discrete_map={"No Failure": "#00CC96", "Failure": "#EF553B"},
        labels={sensor_choice: sensor_choice.replace("_", " ").title()},
    )
    fig_hist.update_layout(height=380, margin=dict(t=20, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPARAISON DES MODÈLES
# ════════════════════════════════════════════════════════════════════════════

elif page == "Comparaison des modèles":
    st.title("📊 Comparaison des modèles")

    # Tableau des métriques
    st.subheader("Tableau récapitulatif des performances")
    display_df = comparison.copy()
    display_df.index = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "Deep Learning (MLP)",
    ]
    display_df.columns = ["Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    display_df = display_df.round(4)

    def highlight_best(col):
        is_max = col == col.max()
        return [
            "background-color: #2e7d32; font-weight: bold" if v else "" for v in is_max
        ]

    st.dataframe(
        display_df.style.apply(highlight_best),
        use_container_width=True,
    )

    st.markdown("---")

    # Bar chart comparatif
    st.subheader("Comparaison visuelle des métriques")
    metric_choice = st.selectbox(
        "Métrique à afficher", ["Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    )
    metric_map = {
        "Recall": "recall",
        "F1-Score": "f1",
        "ROC-AUC": "roc_auc",
        "PR-AUC": "pr_auc",
    }
    col_name = metric_map[metric_choice]

    bar_data = pd.DataFrame(
        {
            "Modèle": display_df.index.tolist(),
            metric_choice: comparison[col_name].values,
        }
    )
    bar_data["Couleur"] = bar_data["Modèle"].map(COLORS)
    fig_bar2 = px.bar(
        bar_data,
        x="Modèle",
        y=metric_choice,
        color="Modèle",
        color_discrete_map=COLORS,
        text=bar_data[metric_choice].round(4),
        range_y=[0.6, 1.0],
    )
    fig_bar2.update_traces(textposition="outside")
    fig_bar2.update_layout(showlegend=False, height=380, margin=dict(t=20, b=10))
    st.plotly_chart(fig_bar2, use_container_width=True)

    st.markdown("---")

    col_roc, col_pr = st.columns(2)

    # Courbes ROC
    with col_roc:
        st.subheader("Courbes ROC")
        fig_roc = go.Figure()
        fig_roc.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(dash="dash", color="grey", width=1),
        )
        for key, label in MODEL_LABELS.items():
            y_proba = results["results"][key]["y_pred_proba"]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = results["results"][key]["roc_auc"]
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f"{label} ({roc_auc:.3f})",
                    line=dict(color=COLORS[label], width=2),
                )
            )
        fig_roc.update_layout(
            xaxis_title="Taux de faux positifs",
            yaxis_title="Taux de vrais positifs",
            height=400,
            margin=dict(t=20, b=10),
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Courbes Précision-Rappel
    with col_pr:
        st.subheader("Courbes Précision-Rappel")
        fig_pr = go.Figure()
        for key, label in MODEL_LABELS.items():
            y_proba = results["results"][key]["y_pred_proba"]
            precision, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            pr_auc_val = auc(recall_vals, precision)
            fig_pr.add_trace(
                go.Scatter(
                    x=recall_vals,
                    y=precision,
                    name=f"{label} ({pr_auc_val:.3f})",
                    line=dict(color=COLORS[label], width=2),
                )
            )
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Précision",
            height=400,
            margin=dict(t=20, b=10),
            legend=dict(x=0.02, y=0.05),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # Matrice de confusion du meilleur modèle
    st.subheader("Matrice de confusion — Random Forest (meilleur modèle)")
    y_pred_rf = results["results"]["rf"]["y_pred"]
    cm = confusion_matrix(y_test, y_pred_rf, normalize="true")
    fig_cm = px.imshow(
        cm,
        text_auto=".2%",
        color_continuous_scale="Blues",
        x=["Prédit: No Failure", "Prédit: Failure"],
        y=["Réel: No Failure", "Réel: Failure"],
    )
    fig_cm.update_layout(height=350, margin=dict(t=20, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INTERPRÉTABILITÉ
# ════════════════════════════════════════════════════════════════════════════

elif page == "Interprétabilité":
    st.title("🔍 Interprétabilité des modèles")

    st.subheader("Importance des variables — Random Forest")
    fi_df = (
        feature_importance.reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
        .sort_values("Importance", ascending=True)
    )
    fi_df.columns = ["Feature", "Importance"]

    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal",
        text=fi_df["Importance"].round(3),
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(
        coloraxis_showscale=False,
        height=420,
        margin=dict(t=20, b=10),
        yaxis_title="",
        xaxis_title="Importance",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.subheader("Analyse des variables dominantes")

    col1, col2 = st.columns(2)

    top_features = (
        feature_importance.sort_values(ascending=False).head(4).index.tolist()
    )
    numeric_top = [f for f in top_features if f in df.columns]

    with col1:
        st.markdown("**Top 4 variables les plus importantes**")
        for i, feat in enumerate(top_features):
            imp = feature_importance[feat]
            st.markdown(f"{i+1}. `{feat}` — **{imp:.1%}**")

    with col2:
        st.markdown("**Interprétation métier**")
        st.markdown(
            "- `temperature_motor` (25.4%) — surchauffe = signal fort de défaillance imminente  \n"
            "- `rpm` (24.9%) — régime anormal = indicateur mécanique clé  \n"
            "- `vibration_rms` (17.8%) — vibrations excessives = usure ou déséquilibre  \n"
            "- `current_phase_avg` (17.1%) — surintensité = anomalie électrique ou mécanique"
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION EN TEMPS RÉEL
# ════════════════════════════════════════════════════════════════════════════

elif page == "Prédiction en temps réel":
    st.title("🤖 Prédiction en temps réel")
    st.markdown(
        "Saisissez les valeurs des capteurs pour obtenir une prédiction "
        "de panne dans les 24 heures."
    )

    # Sélection du modèle
    model_choice = st.selectbox(
        "Modèle utilisé",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"],
    )

    st.markdown("---")
    st.subheader("Paramètres machine")

    col1, col2 = st.columns(2)

    with col1:
        vibration = st.slider("Vibration RMS", 0.35, 10.0, 1.62, 0.01)
        temperature = st.slider("Température moteur (°C)", 28.0, 95.0, 51.4, 0.1)
        current = st.slider("Courant moyen (A)", 2.2, 35.0, 8.8, 0.1)

    with col2:
        pressure = st.slider("Pression (bar)", 10.1, 206.5, 59.0, 0.1)
        rpm = st.slider("RPM", 124.0, 4099.0, 1145.0, 1.0)
        ambient = st.slider("Température ambiante (°C)", 8.0, 18.0, 13.0, 0.1)

    machine_type = st.selectbox(
        "Type de machine", ["CNC", "Compressor", "Pump", "Robotic Arm"]
    )

    st.markdown("---")

    if st.button("Lancer la prédiction", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [[vibration, temperature, current, pressure, rpm, ambient, machine_type]],
            columns=[
                "vibration_rms",
                "temperature_motor",
                "current_phase_avg",
                "pressure_level",
                "rpm",
                "ambient_temp",
                "machine_type",
            ],
        )

        X_input = preprocessor.transform(input_df)

        model_map = {
            "Random Forest": rf,
            "Logistic Regression": lr,
            "Gradient Boosting": gb,
        }
        model = model_map[model_choice]
        proba = model.predict_proba(X_input)[0][1]
        pred = int(proba >= 0.5)

        st.markdown("### Résultat")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if pred == 1:
                st.error(f"⚠️ **PANNE PROBABLE** dans les 24h")
            else:
                st.success(f"✅ **PAS DE PANNE** prévue dans les 24h")

            st.metric("Probabilité de panne", f"{proba:.1%}")

        with col_res2:
            # Jauge de risque
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#EF553B" if proba >= 0.5 else "#00CC96"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                    title={"text": "Niveau de risque"},
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(t=20, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Contribution des features (feature importance du RF normalisée)
        st.markdown("---")
        st.markdown("#### Variables ayant le plus influencé cette prédiction")
        st.caption("Basé sur l'importance globale du Random Forest.")

        numeric_feats = [
            "vibration_rms",
            "temperature_motor",
            "current_phase_avg",
            "pressure_level",
            "rpm",
            "ambient_temp",
        ]
        input_vals = dict(
            zip(
                numeric_feats, [vibration, temperature, current, pressure, rpm, ambient]
            )
        )

        contrib_df = pd.DataFrame(
            {
                "Feature": list(input_vals.keys()),
                "Valeur saisie": list(input_vals.values()),
                "Importance (%)": [
                    feature_importance.get(f, 0) * 100 for f in input_vals.keys()
                ],
            }
        ).sort_values("Importance (%)", ascending=False)

        fig_contrib = px.bar(
            contrib_df,
            x="Feature",
            y="Importance (%)",
            color="Importance (%)",
            color_continuous_scale="Reds",
            text=contrib_df["Importance (%)"].round(1),
        )
        fig_contrib.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_contrib.update_layout(
            coloraxis_showscale=False,
            height=320,
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
