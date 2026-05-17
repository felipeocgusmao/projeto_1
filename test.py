import os
import pandas as pd
import pytest

from src.analise import carregar_dados, limpar_dados, executar, DATA_PATH

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def test_arquivo_existe():
    assert os.path.isfile(DATA_PATH), f"Dataset não encontrado: {DATA_PATH}"


def test_carregamento_shape():
    df = carregar_dados()
    assert df.shape == (300, 10)


def test_carregamento_colunas():
    df = carregar_dados()
    esperadas = {"Tipo", "Zona", "Valoracion", "Noche", "Total", "Evaluaciones"}
    assert esperadas.issubset(df.columns)


# ---------------------------------------------------------------------------
# Limpeza — tipos e parsing
# ---------------------------------------------------------------------------

def test_noche_e_numerico():
    df = limpar_dados(carregar_dados())
    assert pd.api.types.is_numeric_dtype(df["Noche"])


def test_valoracao_e_numerica():
    df = limpar_dados(carregar_dados())
    assert pd.api.types.is_numeric_dtype(df["Valoracion"])


def test_total_e_numerico():
    df = limpar_dados(carregar_dados())
    assert pd.api.types.is_numeric_dtype(df["Total"])


def test_evaluaciones_e_numerico():
    df = limpar_dados(carregar_dados())
    assert pd.api.types.is_numeric_dtype(df["Evaluaciones"])


def test_parsing_valoracao_virgula():
    entrada = pd.DataFrame({"Valoracion": ["4,60", "3,85", None],
                             "Noche": ["24€", "30€", "10€"],
                             "Total": ["47€ en total", "60€ en total", "20€ en total"],
                             "Evaluaciones": [" (25 evaluaciones)", " (10 evaluaciones)", " (5 evaluaciones)"],
                             "Tipo": ["Habitación privada"] * 3})
    df = limpar_dados(entrada)
    assert df["Valoracion"].iloc[0] == pytest.approx(4.60)
    assert df["Valoracion"].iloc[1] == pytest.approx(3.85)


def test_parsing_preco_euro():
    entrada = pd.DataFrame({"Noche": ["24€", "150€", "9€"],
                             "Valoracion": ["4,5", "4,0", "3,9"],
                             "Total": ["48€ en total", "300€ en total", "18€ en total"],
                             "Evaluaciones": [" (1 evaluaciones)"] * 3,
                             "Tipo": ["Habitación privada"] * 3})
    df = limpar_dados(entrada)
    assert list(df["Noche"]) == [24, 150, 9]


def test_sem_nulos_em_noche_apos_limpeza():
    df = limpar_dados(carregar_dados())
    assert df["Noche"].isnull().sum() == 0


def test_tipo_simples_categorias():
    df = limpar_dados(carregar_dados())
    categorias_validas = {
        "Habitación privada",
        "Habitación de hostal",
        "Habitación de hotel",
        "Habitación",
        "Alojamiento entero",
    }
    assert set(df["TipoSimples"].unique()).issubset(categorias_validas)


# ---------------------------------------------------------------------------
# Valores de negócio
# ---------------------------------------------------------------------------

def test_preco_medio_razoavel():
    df = limpar_dados(carregar_dados())
    media = df["Noche"].mean()
    assert 10 <= media <= 500, f"Preço médio fora do esperado: {media}"


def test_valoracao_entre_0_e_5():
    df = limpar_dados(carregar_dados())
    validas = df["Valoracion"].dropna()
    assert (validas >= 0).all() and (validas <= 5).all()


# ---------------------------------------------------------------------------
# Pipeline completo (smoke test)
# ---------------------------------------------------------------------------

def test_pipeline_gera_graficos(tmp_path):
    executar(DATA_PATH, str(tmp_path))
    esperados = ["preco_por_bairro.png", "tipos_alojamento.png", "valoracao_vs_preco.png"]
    for nome in esperados:
        assert (tmp_path / nome).exists(), f"Gráfico não gerado: {nome}"
