import pandas as pd

dados = {
    "cidade": ["Barcelona", "Madrid", "Valencia"],
    "turistas": [1200, 950, 700]
}

df = pd.DataFrame(dados)

print(df)