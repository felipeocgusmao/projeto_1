import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.analise import carregar_dados, limpar_dados, DATA_PATH
from src.modelo import treinar, preparar_features, FEATURES

st.set_page_config(page_title="Airbnb Barcelona", page_icon="🏠", layout="wide")


@st.cache_data
def carregar():
    return limpar_dados(carregar_dados())


@st.cache_resource
def carregar_modelo(df):
    return treinar(df)


df = carregar()
modelo, metricas = carregar_modelo(df)

# ---------------------------------------------------------------------------
# Sidebar — filtros
# ---------------------------------------------------------------------------
st.sidebar.title("Filtros")

zonas = sorted(df["Zona"].dropna().unique())
zonas_sel = st.sidebar.multiselect("Bairro", zonas, default=zonas)

tipos = sorted(df["TipoSimples"].unique())
tipos_sel = st.sidebar.multiselect("Tipo de alojamento", tipos, default=tipos)

preco_min, preco_max = int(df["Noche"].min()), int(df["Noche"].max())
preco_range = st.sidebar.slider("Preço por noite (€)", preco_min, preco_max, (preco_min, preco_max))

df_filtrado = df[
    df["Zona"].isin(zonas_sel) &
    df["TipoSimples"].isin(tipos_sel) &
    df["Noche"].between(*preco_range)
]

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("🏠 Análise de Alojamentos Airbnb — Barcelona")
st.caption(f"{len(df_filtrado)} alojamentos com os filtros aplicados")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Alojamentos", len(df_filtrado))
col2.metric("Preço médio/noite", f"{df_filtrado['Noche'].mean():.0f}€")
col3.metric("Valoração média", f"{df_filtrado['Valoracion'].mean():.2f} ⭐")
col4.metric("Avaliações médias", f"{df_filtrado['Evaluaciones'].mean():.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_eda, tab_modelo = st.tabs(["📊 Análise exploratória", "🤖 Previsão de preço"])

with tab_eda:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 bairros — preço médio")
        top = (
            df_filtrado.groupby("Zona")["Noche"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        if top.empty:
            st.info("Nenhum dado com os filtros aplicados.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}€"))
            ax.bar_label(bars, fmt="%.0f€", padding=3, fontsize=8)
            ax.set_xlabel("Preço médio (€)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_b:
        st.subheader("Distribuição por tipo")
        contagem = df_filtrado["TipoSimples"].value_counts()
        if contagem.empty:
            st.info("Nenhum dado com os filtros aplicados.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                contagem.values,
                labels=contagem.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader("Valoração vs. Preço por noite")
    subset = df_filtrado.dropna(subset=["Valoracion", "Noche"])
    if subset.empty:
        st.info("Nenhum dado com os filtros aplicados.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        sc = ax.scatter(subset["Noche"], subset["Valoracion"], alpha=0.5,
                        c=subset["Noche"], cmap="viridis", edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Preço (€)")
        ax.set_xlabel("Preço por noite (€)")
        ax.set_ylabel("Valoração")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab_modelo:
    st.subheader("Simule o preço de um alojamento")
    st.caption(f"Modelo: Random Forest — MAE {metricas['mae']:.1f}€ · R² {metricas['r2']:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        zona_sel = st.selectbox("Bairro", sorted(df["Zona"].dropna().unique()))
        tipo_sel = st.selectbox("Tipo", sorted(df["TipoSimples"].unique()))
        valoracao_inp = st.slider("Valoração", 1.0, 5.0, 4.5, step=0.01)
        avaliacoes_inp = st.number_input("Nº de avaliações", min_value=1, max_value=2000, value=50)
    with c2:
        hospedes_inp = st.number_input("Hóspedes", min_value=1, max_value=16, value=2)
        quartos_inp = st.number_input("Quartos", min_value=1, max_value=10, value=1)
        camas_inp = st.number_input("Camas", min_value=1, max_value=10, value=1)
        banheiros_inp = st.number_input("Banheiros", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    if st.button("Prever preço", type="primary"):
        entrada = pd.DataFrame([{
            "TipoSimples": tipo_sel,
            "Zona": zona_sel,
            "n_hospedes": hospedes_inp,
            "n_quartos": quartos_inp,
            "n_camas": camas_inp,
            "n_banheiros": banheiros_inp,
            "Valoracion": valoracao_inp,
            "Evaluaciones": avaliacoes_inp,
        }])
        X_entrada, _ = preparar_features(
            pd.concat([df[FEATURES], entrada], ignore_index=True)
        )
        preco = modelo.predict(X_entrada.iloc[[-1]])[0]
        st.success(f"### Preço estimado: {preco:.0f}€ / noite")

    st.divider()
    st.subheader("Importância das features")
    imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    imp.plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Importância média")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
