import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv(r"c:\Users\mayum\OneDrive\Área de Trabalho\Estatistica Trabalho\treecover_loss__ha.csv")

# Definir a coluna a ser analisada
coluna_perda = "umd_tree_cover_loss__ha"

# Criar a figura com dois subplots
plt.figure(figsize=(14, 6))

# 🔹 Gráfico 1: Histograma (Distribuição de Frequência)
plt.subplot(1, 2, 1)
sns.histplot(df[coluna_perda], bins=10, kde=True, color="blue")
plt.xlabel("Perda de Cobertura Florestal (ha)")
plt.ylabel("Frequência")
plt.title("Distribuição de Frequência - Histograma")

# 🔹 Gráfico 2: Boxplot (Identificar Outliers)
plt.subplot(1, 2, 2)
sns.boxplot(y=df[coluna_perda], color="orange")
plt.ylabel("Perda de Cobertura Florestal (ha)")
plt.title("Boxplot - Detecção de Outliers")

# Ajustar espaçamento e exibir os gráficos
plt.tight_layout()
plt.show()
