import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "alojamientos.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ---------------------------------------------------------------------------
# Carregamento e limpeza
# ---------------------------------------------------------------------------

def carregar_dados(caminho: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Valoracion: "4,60" → 4.60 (descarta nulos)
    df["Valoracion"] = (
        df["Valoracion"]
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Noche: "24€" → 24
    df["Noche"] = (
        df["Noche"]
        .str.replace("€", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Total: "47€ en total" → 47
    df["Total"] = (
        df["Total"]
        .str.extract(r"(\d+)")
        .squeeze()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Evaluaciones: " (25 evaluaciones)" → 25
    df["Evaluaciones"] = (
        df["Evaluaciones"]
        .str.extract(r"(\d+)")
        .squeeze()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Tipo: agrupa em categorias legíveis
    def _simplificar_tipo(t: str) -> str:
        t = t.lower()
        if "privada" in t:
            return "Habitación privada"
        if "hostal" in t:
            return "Habitación de hostal"
        if "hotel" in t:
            return "Habitación de hotel"
        if "habitación" in t or "habitacion" in t:
            return "Habitación"
        return "Alojamiento entero"

    df["TipoSimples"] = df["Tipo"].apply(_simplificar_tipo)

    # Remove linhas sem preço (não há como analisar)
    df = df.dropna(subset=["Noche"])

    return df


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def grafico_preco_por_bairro(df: pd.DataFrame, destino: str) -> None:
    top_zonas = (
        df.groupby("Zona")["Noche"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_zonas.index[::-1], top_zonas.values[::-1], color="#4C72B0")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}€"))
    ax.set_xlabel("Preço médio por noite")
    ax.set_title("Top 10 Bairros — Preço Médio por Noite")
    ax.bar_label(bars, fmt="%.0f€", padding=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close()


def grafico_tipos_alojamento(df: pd.DataFrame, destino: str) -> None:
    contagem = df["TipoSimples"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        contagem.values,
        labels=contagem.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Distribuição por Tipo de Alojamento")
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close()


def grafico_valoracao_vs_preco(df: pd.DataFrame, destino: str) -> None:
    subset = df.dropna(subset=["Valoracion", "Noche"])

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        subset["Noche"],
        subset["Valoracion"],
        alpha=0.5,
        c=subset["Noche"],
        cmap="viridis",
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="Preço por noite (€)")
    ax.set_xlabel("Preço por noite (€)")
    ax.set_ylabel("Valoração")
    ax.set_title("Valoração vs. Preço por Noite")
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def executar(caminho_dados: str = DATA_PATH, diretorio_saida: str = OUTPUT_DIR) -> pd.DataFrame:
    os.makedirs(diretorio_saida, exist_ok=True)

    df_raw = carregar_dados(caminho_dados)
    df = limpar_dados(df_raw)

    grafico_preco_por_bairro(df, os.path.join(diretorio_saida, "preco_por_bairro.png"))
    grafico_tipos_alojamento(df, os.path.join(diretorio_saida, "tipos_alojamento.png"))
    grafico_valoracao_vs_preco(df, os.path.join(diretorio_saida, "valoracao_vs_preco.png"))

    print(f"Registros carregados : {len(df_raw)}")
    print(f"Registros após limpeza: {len(df)}")
    print(f"\nPreço médio por noite : {df['Noche'].mean():.2f}€")
    print(f"Valoração média       : {df['Valoracion'].mean():.2f}")
    print(f"\nGráficos salvos em: {os.path.abspath(diretorio_saida)}")

    return df


if __name__ == "__main__":
    executar()
