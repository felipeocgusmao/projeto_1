import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.analise import carregar_dados, limpar_dados, DATA_PATH, OUTPUT_DIR


FEATURES = ["TipoSimples", "Zona", "n_hospedes", "n_quartos", "n_camas", "n_banheiros", "Valoracion", "Evaluaciones"]
TARGET = "Noche"


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = df[FEATURES].copy()
    encoders = {}
    for col in ("TipoSimples", "Zona"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("Desconhecido"))
        encoders[col] = le
    X = X.fillna(X.median(numeric_only=True))
    return X, encoders


def treinar(df: pd.DataFrame, seed: int = 42) -> tuple[RandomForestRegressor, dict]:
    subset = df.dropna(subset=[TARGET])
    X, encoders = preparar_features(subset)
    y = subset[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    modelo = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metricas = {
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2":   r2_score(y_test, y_pred),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return modelo, metricas


def grafico_importancia(modelo: RandomForestRegressor, destino: str) -> None:
    importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Importância das Features — Random Forest")
    ax.set_xlabel("Importância média")
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close()


def grafico_real_vs_previsto(metricas: dict, destino: str) -> None:
    y_test = metricas["y_test"]
    y_pred = metricas["y_pred"]
    lim = max(y_test.max(), y_pred.max()) + 10

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="none", color="#4C72B0")
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Preço real (€)")
    ax.set_ylabel("Preço previsto (€)")
    ax.set_title(f"Real vs. Previsto  |  MAE={metricas['mae']:.1f}€  R²={metricas['r2']:.2f}")
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close()


def executar(caminho_dados: str = DATA_PATH, diretorio_saida: str = OUTPUT_DIR) -> dict:
    os.makedirs(diretorio_saida, exist_ok=True)

    df = limpar_dados(carregar_dados(caminho_dados))
    modelo, metricas = treinar(df)

    grafico_importancia(modelo, os.path.join(diretorio_saida, "importancia_features.png"))
    grafico_real_vs_previsto(metricas, os.path.join(diretorio_saida, "real_vs_previsto.png"))

    print(f"MAE  : {metricas['mae']:.2f}€")
    print(f"RMSE : {metricas['rmse']:.2f}€")
    print(f"R²   : {metricas['r2']:.3f}")

    return metricas


if __name__ == "__main__":
    executar()
