# Airbnb Barcelona — Análise & Previsão de Preços

Dashboard interativo com análise exploratória e modelo preditivo de preços de alojamentos Airbnb em Barcelona.

🔗 **[Abrir app](https://projeto1-ow9ole3qp8uyiempm.streamlit.app/)**

---

## Funcionalidades

| Aba | O que mostra |
|---|---|
| Análise exploratória | Top 10 bairros por preço médio, distribuição por tipo de alojamento e scatter de valoração vs preço — tudo filtrável por bairro, tipo e faixa de preço |
| Previsão de preço | Formulário para simular o preço de um alojamento usando o modelo Random Forest (MAE ≈ 10€ · R² ≈ 0.80) |

## Dataset

- **Fonte:** Kaggle — Comparative Analysis of Airbnb Prices in Barcelona
- **Registros:** 300 alojamentos com preço, zona, tipo, valoração e número de hóspedes/quartos/camas/banheiros

## Principais conclusões

- Bairros mais caros: Complejo Residencial (129€), El Putxet i el Farró (96€), Eixample (84€)
- 65% dos alojamentos são quartos privados
- Preço alto não garante melhor valoração

## Estrutura do projeto

```
├── app.py                  # Dashboard Streamlit
├── src/
│   ├── analise.py          # Carregamento, limpeza e gráficos locais
│   └── modelo.py           # Treino e avaliação do Random Forest
├── data/
│   └── alojamientos.csv    # Dataset
├── notebooks/
│   └── analise_turismo.ipynb
├── outputs/                # Gráficos gerados localmente
├── test.py                 # 24 testes pytest
└── requirements.txt
```

## Como usar localmente

```bash
pip install -r requirements.txt
streamlit run app.py        # dashboard
python src/analise.py       # gera outputs/
python src/modelo.py        # métricas do modelo
pytest test.py -v           # testes
```

## Tecnologias

Python · Pandas · scikit-learn · Plotly · Streamlit · Matplotlib · pytest · GitHub Actions
