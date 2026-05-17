import os
import pandas as pd
import pytest

from src.analise import carregar_dados, limpar_dados, executar, _parsear_nhab, DATA_PATH
from src.modelo import treinar, preparar_features

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


def _df_minimo(**kwargs) -> pd.DataFrame:
    defaults = {
        "Tipo": ["Habitación privada"] * 3,
        "Noche": ["24€", "30€", "10€"],
        "Valoracion": ["4,60", "3,85", "4,00"],
        "Total": ["47€ en total", "60€ en total", "20€ en total"],
        "Evaluaciones": [" (25 evaluaciones)", " (10 evaluaciones)", " (5 evaluaciones)"],
        "Nhab": ["2 huéspedes · 1 dormitorio · 1 cama · 1 baño compartido"] * 3,
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_parsing_valoracao_virgula():
    entrada = _df_minimo(Valoracion=["4,60", "3,85", None])
    df = limpar_dados(entrada)
    assert df["Valoracion"].iloc[0] == pytest.approx(4.60)
    assert df["Valoracion"].iloc[1] == pytest.approx(3.85)


def test_parsing_preco_euro():
    entrada = _df_minimo(Noche=["24€", "150€", "9€"],
                         Valoracion=["4,5", "4,0", "3,9"],
                         Total=["48€ en total", "300€ en total", "18€ en total"],
                         Evaluaciones=[" (1 evaluaciones)"] * 3)
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
# Parsing de Nhab
# ---------------------------------------------------------------------------

def test_parsear_nhab_completo():
    s = _parsear_nhab("2 huéspedes · 1 dormitorio · 1 cama · 1 baño compartido")
    assert s["n_hospedes"] == 2
    assert s["n_quartos"] == 1
    assert s["n_camas"] == 1
    assert s["n_banheiros"] == 1.0


def test_parsear_nhab_banheiro_decimal():
    s = _parsear_nhab("2 huéspedes · 1 dormitorio · 1 cama · 1,5 baños compartidos")
    assert s["n_banheiros"] == pytest.approx(1.5)


def test_parsear_nhab_sem_camas():
    s = _parsear_nhab("2 huéspedes · 1 dormitorio · 1 baño privado")
    assert s["n_hospedes"] == 2
    assert s["n_camas"] is None or (isinstance(s["n_camas"], float) and pd.isna(s["n_camas"]))


def test_parsear_nhab_nulo():
    s = _parsear_nhab(None)
    assert s["n_hospedes"] is None


def test_nhab_colunas_no_dataframe():
    df = limpar_dados(carregar_dados())
    for col in ("n_hospedes", "n_quartos", "n_camas", "n_banheiros"):
        assert col in df.columns, f"Coluna ausente: {col}"


def test_nhab_hospedes_positivos():
    df = limpar_dados(carregar_dados())
    validos = df["n_hospedes"].dropna()
    assert (validos > 0).all()


# ---------------------------------------------------------------------------
# Modelo de previsão de preço
# ---------------------------------------------------------------------------

def test_modelo_metricas_razoaveis():
    df = limpar_dados(carregar_dados())
    _, metricas = treinar(df)
    assert metricas["mae"] < 50, f"MAE alto demais: {metricas['mae']:.2f}"
    assert 0 <= metricas["r2"] <= 1, f"R² fora do intervalo: {metricas['r2']:.3f}"


def test_modelo_previsoes_positivas():
    df = limpar_dados(carregar_dados())
    _, metricas = treinar(df)
    assert (metricas["y_pred"] >= 0).all()


def test_preparar_features_sem_nulos():
    df = limpar_dados(carregar_dados())
    X, _ = preparar_features(df)
    assert X.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Pipeline completo (smoke test)
# ---------------------------------------------------------------------------

def test_pipeline_gera_graficos(tmp_path):
    executar(DATA_PATH, str(tmp_path))
    esperados = ["preco_por_bairro.png", "tipos_alojamento.png", "valoracao_vs_preco.png"]
    for nome in esperados:
        assert (tmp_path / nome).exists(), f"Gráfico não gerado: {nome}"


def test_modelo_gera_graficos(tmp_path):
    from src.modelo import executar as executar_modelo
    executar_modelo(DATA_PATH, str(tmp_path))
    for nome in ("importancia_features.png", "real_vs_previsto.png"):
        assert (tmp_path / nome).exists(), f"Gráfico do modelo não gerado: {nome}"
