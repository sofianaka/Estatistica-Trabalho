import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Carregar os dados
df = pd.read_csv(r"c:\Users\mayum\OneDrive\Área de Trabalho\Estatistica Trabalho\treecover_loss__ha.csv")

# Exibir informações básicas dos dados
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

# Definir a coluna de análise
coluna_perda = "umd_tree_cover_loss__ha"

# Cálculo das estatísticas básicas
num_elementos = df[coluna_perda].count()
media_perda = df[coluna_perda].mean()
variancia_perda = df[coluna_perda].var()
desvio_padrao = df[coluna_perda].std()
moda_perda = df[coluna_perda].mode()[0]  # Pode haver mais de uma moda
mediana_perda = df[coluna_perda].median()

# Exibir estatísticas
print(f"Número de elementos: {num_elementos}")
print(f"Média: {media_perda:.2f}")
print(f"Variância: {variancia_perda:.2f}")
print(f"Desvio Padrão: {desvio_padrao:.2f}")
print(f"Moda: {moda_perda:.2f}")
print(f"Mediana: {mediana_perda:.2f}")

# Gráfico de histograma
plt.figure(figsize=(10, 5))
sns.histplot(df[coluna_perda], bins=30, kde=True)
plt.axvline(media_perda, color='red', linestyle='dashed', linewidth=2, label='Média')
plt.axvline(mediana_perda, color='green', linestyle='dashed', linewidth=2, label='Mediana')
plt.axvline(moda_perda, color='blue', linestyle='dashed', linewidth=2, label='Moda')
plt.legend()
plt.title("Distribuição de Frequências da Perda Florestal por Hectare")
plt.xlabel("Perda Florestal (ha)")
plt.ylabel("Frequência")
plt.show()

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df[coluna_perda])
plt.title("Boxplot da Perda Florestal por Hectare")
plt.show()

# Análise de simetria (skewness) e curtose
assimetria = stats.skew(df[coluna_perda])
curtose = stats.kurtosis(df[coluna_perda])
print(f"Coeficiente de Assimetria: {assimetria:.2f}")
print(f"Curtose: {curtose:.2f}")

# Análise da normalidade
if abs(assimetria) < 0.5:
    print("A distribuição é aproximadamente simétrica.")
elif assimetria > 0.5:
    print("A distribuição é assimétrica à direita (positiva).")
else:
    print("A distribuição é assimétrica à esquerda (negativa).")

# Teste de normalidade de Shapiro-Wilk
stat, p_value = stats.shapiro(df[coluna_perda])
print(f"Teste de Shapiro-Wilk: estatística={stat:.4f}, p-valor={p_value:.4f}")
if p_value > 0.05:
    print("Os dados seguem uma distribuição normal.")
else:
    print("Os dados não seguem uma distribuição normal.")




