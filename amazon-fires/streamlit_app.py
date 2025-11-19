# streamlit_app.py

"""
Objectifs de ton brief :
- Comprendre la tendance des feux dans le temps
- Identifier les États les plus touchés
- Repérer les périodes (mois) les plus fréquentes
- Regarder des corrélations simples utiles au récit

Points clés de cette version :
- Filtres (période + États)
- KPI cards (total, pic mensuel, mois de pic)
- Tendance annuelle (national + par État)
- Classement des États (bar chart)
- Heatmaps (Mois×Années, États×Années, États×Mois)
- Corrélations rapides (Spearman) + tableau
- Carte : Treemap (robuste sans GeoJSON). Optionnel : choropleth si tu fournis un GeoJSON local des États du Brésil
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# -------------------
# 1) Chargement des données
# -------------------
DATA_PATH = Path(__file__).with_name("amazon.csv")  # amazon.csv dans le même dossier

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"❌ Fichier introuvable : {path}")
        st.stop()

    df = pd.read_csv(path, encoding="latin1")

    # Normalisation des noms de colonnes si besoin
    df.columns = [c.strip().lower() for c in df.columns]

    # Mois PT → numéro
    month_map = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
        "outubro": 10, "novembro": 11, "dezembro": 12
    }
    # Harmonisation
    df["month"] = df["month"].astype(str).str.strip().str.lower()
    df["month_num"] = df["month"].map(month_map)

    # Types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["number"] = pd.to_numeric(df["number"], errors="coerce").fillna(0).astype(int)

    # Date au 1er du mois
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month_num"].fillna(1).astype(int).astype(str) + "-01",
        errors="coerce"
    )

    # Nettoyage basique
    df = df.dropna(subset=["year", "month_num", "state", "number"])  # garde les lignes valides
    df = df[df["number"] >= 0]

    return df


df = load_data(DATA_PATH)

# après le chargement de df
name_to_uf = {
    "Acre":"AC","Alagoas":"AL","Amapá":"AP","Amazonas":"AM","Bahia":"BA","Ceará":"CE","Distrito Federal":"DF",
    "Espírito Santo":"ES","Goiás":"GO","Maranhão":"MA","Mato Grosso":"MT","Mato Grosso do Sul":"MS",
    "Minas Gerais":"MG","Pará":"PA","Paraíba":"PB","Paraná":"PR","Pernambuco":"PE","Piauí":"PI",
    "Rio de Janeiro":"RJ","Rio Grande do Norte":"RN","Rio Grande do Sul":"RS","Rondônia":"RO","Roraima":"RR",
    "Santa Catarina":"SC","São Paulo":"SP","Sergipe":"SE","Tocantins":"TO"
}
df["state_clean"] = df["state"].astype(str).str.strip()
df["uf"] = df["state_clean"].map(name_to_uf)

missing = df.loc[df["uf"].isna(), "state_clean"].drop_duplicates().tolist()
if missing:
    st.warning(f"États non reconnus (à mapper) : {missing}")


import unicodedata

# --- Normalisation : supprime accents, espaces multiples, casse ---
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

df["state_raw"] = df["state"].astype(str).str.strip()
df["state_norm"] = (
    df["state_raw"]
    .str.lower()
    .apply(strip_accents)
    .str.replace(r"\s+", " ", regex=True)
)

# --- Corrections d'alias -> nom officiel IBGE ---
alias_to_proper = {
    # cas remontés par ton warning
    "amapa": "Amapá",
    "ceara": "Ceará",
    "espirito santo": "Espírito Santo",
    "goias": "Goiás",
    "maranhao": "Maranhão",
    "paraiba": "Paraíba",
    "piau": "Piauí",       # parfois tronqué dans le CSV
    "piaui": "Piauí",
    "rio": "Rio de Janeiro",   # par défaut, on choisit RJ
    "rondonia": "Rondônia",
    "sao paulo": "São Paulo",
    # bonus fréquents (au cas où)
    "para": "Pará",
    "santacatarina": "Santa Catarina",
    "rio grande do sul": "Rio Grande do Sul",
    "rio grande do norte": "Rio Grande do Norte",
    "minas gerais": "Minas Gerais",
    "mato grosso": "Mato Grosso",
    "mato grosso do sul": "Mato Grosso do Sul",
    "espirito-santo": "Espírito Santo",
}

# Applique l'alias → nom officiel si connu, sinon remonte l'original
df["state_clean"] = df["state_norm"].map(alias_to_proper).fillna(df["state_raw"])

# Puis map vers UF
name_to_uf = {
    "Acre":"AC","Alagoas":"AL","Amapá":"AP","Amazonas":"AM","Bahia":"BA","Ceará":"CE","Distrito Federal":"DF",
    "Espírito Santo":"ES","Goiás":"GO","Maranhão":"MA","Mato Grosso":"MT","Mato Grosso do Sul":"MS",
    "Minas Gerais":"MG","Pará":"PA","Paraíba":"PB","Paraná":"PR","Pernambuco":"PE","Piauí":"PI",
    "Rio de Janeiro":"RJ","Rio Grande do Norte":"RN","Rio Grande do Sul":"RS","Rondônia":"RO","Roraima":"RR",
    "Santa Catarina":"SC","São Paulo":"SP","Sergipe":"SE","Tocantins":"TO"
}
df["uf"] = df["state_clean"].map(name_to_uf)

missing = df.loc[df["uf"].isna(), "state_clean"].drop_duplicates().tolist()
if missing:
    st.warning(f"États non reconnus (à mapper) : {missing}")


# -------------------
# 2) UI
# -------------------
st.set_page_config(page_title="🔥 Feux de forêts – Brésil", layout="wide")
st.title("🔥 Feux de forêts au Brésil (1998–2017)")
st.markdown("""
Ce tableau de bord explore l'évolution des feux de forêts au Brésil (jeu de données Kaggle).
Utilise les filtres pour explorer par **année**, **État** et **période**.
""")

c1, c2 = st.columns([2, 1])
with c1:
    years = st.slider(
        "Période :",
        int(df["year"].min()),
        int(df["year"].max()),
        (2000, 2010),
        step=1
    )
with c2:
    default_states = ["Amazonas", "Mato Grosso"]
    states = st.multiselect(
        "États :",
        options=sorted(df["state"].unique()),
        default=[s for s in default_states if s in df["state"].unique()]
    )

# Filtrage
f = df[(df["year"] >= years[0]) & (df["year"] <= years[1])].copy()
if states:
    f = f[f["state"].isin(states)]

if f.empty:
    st.warning("Aucune donnée pour la période/états sélectionnés.")
    st.stop()

# -------------------
# 3) KPI Cards
# -------------------
agg_period = (f.groupby("date", as_index=False)["number"].sum().sort_values("date"))

total_fires = int(f["number"].sum())
peak_month_row = (agg_period.loc[agg_period["number"].idxmax()])
peak_month_label = peak_month_row["date"].strftime("%b %Y")
peak_month_value = int(peak_month_row["number"])
mean_monthly = float(agg_period["number"].mean())

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total feux (période filtrée)", f"{total_fires:,}".replace(",", " "))
with k2:
    st.metric("Pic mensuel (période filtrée)", f"{peak_month_value:,}".replace(",", " "), help=peak_month_label)
with k3:
    st.metric("Moyenne mensuelle", f"{mean_monthly:,.1f}".replace(",", " "))

# -------------------
# 4) Tendance
# -------------------
st.subheader("📈 Tendance des feux dans le temps")

# National (somme toutes régions)
annual_nat = (f.groupby("year", as_index=False)["number"].sum())
fig_trend_nat = px.line(annual_nat, x="year", y="number", markers=True,
                        title="Tendance annuelle (national)", labels={"number": "Nombre de feux", "year": "Année"})
st.plotly_chart(fig_trend_nat, use_container_width=True)

# Par État (facultatif) – utile si plusieurs États sélectionnés
if f["state"].nunique() > 1:
    annual_state = f.groupby(["year", "state"], as_index=False)["number"].sum()
    fig_trend_state = px.line(annual_state, x="year", y="number", color="state", markers=False,
                              title="Évolution annuelle par État", labels={"number": "Feux", "year": "Année"})
    st.plotly_chart(fig_trend_state, use_container_width=True)

# -------------------
# 5) États les plus touchés
# -------------------
st.subheader("🏆 États les plus touchés (période filtrée)")
state_rank = (f.groupby("state", as_index=False)["number"].sum().sort_values("number", ascending=False))
fig_states = px.bar(state_rank.head(20), x="number", y="state", orientation="h",
                    title="Top États par nombre total de feux", labels={"number": "Total feux", "state": "État"})
fig_states.update_yaxes(categoryorder="total ascending")
st.plotly_chart(fig_states, use_container_width=True)

# Treemap (robuste sans GeoJSON)
with st.expander("Alternative visuelle : Treemap par État"):
    fig_tree = px.treemap(state_rank, path=["state"], values="number",
                          title="Part des feux par État (période filtrée)")
    st.plotly_chart(fig_tree, use_container_width=True)

# -------------------
# 6) Saisonnalité & Heatmaps
# -------------------
st.subheader("🔥 Saisonnalité & Heatmaps")
month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

htype = st.radio(
    "Vue",
    ["Saisonnalité (Mois × Années)", "États × Années", "États × Mois (année choisie)"],
    horizontal=True,
)

if htype == "Saisonnalité (Mois × Années)":
    tmp = f.groupby(["year", "month_num"], as_index=False)["number"].sum()
    pivot = (tmp.pivot(index="year", columns="month_num", values="number")
                .reindex(columns=range(1, 13)).fillna(0))
    fig_hm = px.imshow(
        pivot.values,
        x=month_labels,
        y=pivot.index,
        origin="upper",
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(x="Mois", y="Année", color="Nombre de feux"),
        title="Saisonnalité des feux (Mois × Années)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

elif htype == "États × Années":
    tmp = f.groupby(["state", "year"], as_index=False)["number"].sum()
    order_states = tmp.groupby("state")["number"].sum().sort_values(ascending=False).index.tolist()
    pivot = tmp.pivot(index="state", columns="year", values="number").reindex(index=order_states).fillna(0)
    fig_hm = px.imshow(
        pivot.values,
        x=pivot.columns.astype(int),
        y=pivot.index,
        origin="upper",
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(x="Année", y="État", color="Nombre de feux"),
        title="Feux par État et par Année",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

else:  # États × Mois (année choisie)
    year_choice = st.select_slider("Choisis l'année", options=sorted(f["year"].unique()), value=int(f["year"].median()))
    data = f[f["year"] == year_choice]
    tmp = data.groupby(["state", "month_num"], as_index=False)["number"].sum()
    order_states = tmp.groupby("state")["number"].sum().sort_values(ascending=False).index.tolist()
    pivot = tmp.pivot(index="state", columns="month_num", values="number").reindex(index=order_states, columns=range(1, 13)).fillna(0)
    fig_hm = px.imshow(
        pivot.values,
        x=month_labels,
        y=pivot.index,
        origin="upper",
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(x="Mois", y="État", color="Nombre de feux"),
        title=f"Feux par État et par Mois — {year_choice}",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# -------------------
# 7) Corrélations utiles
# -------------------
st.subheader("🔗 Corrélations simples (Spearman)")

# 7.1 Corrélation Année ↔ Feux (niveau national mensuel)
nat_monthly = f.groupby("date", as_index=False)["number"].sum()
# Remplace NaT éventuels
nat_monthly = nat_monthly.dropna(subset=["date"])  

# Spearman corr year vs number
nat_monthly["year"] = nat_monthly["date"].dt.year
corr_year = nat_monthly[["year", "number"]].corr(method="spearman").iloc[0, 1]

# 7.2 Corrélation Mois ↔ Feux (saisonnalité au niveau national)
nat_monthly["month_num"] = nat_monthly["date"].dt.month
corr_month = nat_monthly[["month_num", "number"]].corr(method="spearman").iloc[0, 1]

# 7.3 Option : au niveau des États, entre année et feux (médiane des corrélations par État)
state_year_corrs = []
for s, dfg in f.groupby("state"):
    if dfg["year"].nunique() > 1:
        tmp = dfg.groupby("year")["number"].sum().reset_index()
        c = tmp[["year", "number"]].corr(method="spearman").iloc[0, 1]
        if not np.isnan(c):
            state_year_corrs.append(c)
median_state_year_corr = float(np.median(state_year_corrs)) if state_year_corrs else float("nan")

corr_df = pd.DataFrame({
    "Corrélation": ["Année ↔ Feux (national, mensuel)", "Mois ↔ Feux (national, mensuel)", "Année ↔ Feux (médiane des États)",],
    "Spearman ρ": [round(float(corr_year), 3) if pd.notnull(corr_year) else None,
                    round(float(corr_month), 3) if pd.notnull(corr_month) else None,
                    round(float(median_state_year_corr), 3) if pd.notnull(median_state_year_corr) else None]
})

st.dataframe(corr_df, use_container_width=True)

st.caption("Note : Spearman mesure une association monotone. Interprétation simple : ρ>0 tendance à augmenter, ρ<0 tendance à baisser. La corrélation 'Mois ↔ Feux' capte la saisonnalité (pics en fin d'hiver austral).")

# -------------------
# 8) (Optionnel) Carte choropleth par GeoJSON local
# -------------------
import json
from pathlib import Path
import plotly.express as px

st.subheader("🗺️ Carte choropleth – Feux totaux par État (période filtrée)")

geojson_path = Path("br_states.geojson")

map_df = f.groupby("state_clean", as_index=False)["number"].sum()
map_df["uf"] = map_df["state_clean"].map(name_to_uf)

# détection des clés du GeoJSON (sigla vs name) comme on l’a fait

if not geojson_path.exists():
    st.info("Place br_states.geojson à côté du script.")
else:
    # Agrège les feux sur le dataframe filtré f (celui après tes sliders)
    map_df = f.groupby("state_clean", as_index=False)["number"].sum()
    map_df["uf"] = map_df["state_clean"].map(name_to_uf)

    with open(geojson_path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)

    # Détecte automatiquement si le geojson a 'sigla' (UF) ou 'name' (nom complet)
    props = gj["features"][0]["properties"]
    keys_lower = {k.lower(): k for k in props.keys()}
    has_sigla = "sigla" in keys_lower
    has_name  = "name"  in keys_lower

    if has_sigla and map_df["uf"].notna().all():
        featureidkey = f"properties.{keys_lower['sigla']}"  # respecte la casse réelle du fichier
        locations_col = "uf"
    elif has_name:
        featureidkey = f"properties.{keys_lower['name']}"
        locations_col = "state_clean"
    else:
        st.error("Clé d’identifiant introuvable dans le GeoJSON (ni 'sigla' ni 'name').")
        st.stop()

    fig_map = px.choropleth(
        map_df,
        geojson=gj,
        locations=locations_col,
        featureidkey=featureidkey,
        color="number",
        color_continuous_scale="Reds",
        title="Feux totaux par État (période filtrée)",
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

# -------------------
# 9) Conclusion / pistes
# -------------------
st.markdown(
    """
    **Pistes d'enrichissement** : météo (précipitations, sécheresse), déforestation, surfaces agricoles, évènements El Niño/La Niña.
    """
)
