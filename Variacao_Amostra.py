import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from scipy import stats  

# Carregar os dados  
df = pd.read_csv("treecover_loss__ha.csv")  

# perda de cobertura vegetal em hectares  
coluna_dados = "umd_tree_cover_loss__ha"  

# parâmetros da população  
populacao_media = df[coluna_dados].mean()  
populacao_variancia = df[coluna_dados].var()  
populacao_desvio = df[coluna_dados].std()  
populacao_mediana = df[coluna_dados].median()  
populacao_moda = stats.mode(df[coluna_dados], keepdims=True)[0][0]  

# Exibir os parâmetros da população  
print("\n=== Parâmetros da População ===")  
print(f"Média: {populacao_media:.2f}")  
print(f"Variância: {populacao_variancia:.2f}")  
print(f"Desvio Padrão: {populacao_desvio:.2f}")  
print(f"Mediana: {populacao_mediana:.2f}")  
print(f"Moda: {populacao_moda:.2f}")  

# As proporções de amostragem  
proporcoes = [0.0001, 0.001, 0.01, 0.05]  # 0,01%; 0,1%; 1%; 5%  
n_total = len(df)  

# Criar para os gráficos  
plt.figure(figsize=(12, 8))  

# Retirar amostras e calcular estatísticas  
for i, proporcao in enumerate(proporcoes):  
    n_amostra = max(1, int(n_total * proporcao))  # Garante pelo menos 1 elemento  
    amostra = df.sample(n=n_amostra)  # Removendo o random_state para variar as amostras  

    # Estimadores estatísticos  
    media = amostra[coluna_dados].mean()  
    variancia = amostra[coluna_dados].var()  
    desvio = amostra[coluna_dados].std()  
    mediana = amostra[coluna_dados].median()  
    moda = stats.mode(amostra[coluna_dados], keepdims=True)[0][0]  

    # Resultados  
    print(f"\n=== Amostra de {proporcao*100:.2f}% da População (n={n_amostra}) ===")  
    print(f"Média: {media:.2f} (Diff: {media - populacao_media:.2f})")  
    print(f"Variância: {variancia:.2f} (Diff: {variancia - populacao_variancia:.2f})")  
    print(f"Desvio Padrão: {desvio:.2f} (Diff: {desvio - populacao_desvio:.2f})")  
    print(f"Mediana: {mediana:.2f} (Diff: {mediana - populacao_mediana:.2f})")  
    print(f"Moda: {moda:.2f} (Diff: {moda - populacao_moda:.2f})")  

    # Gráficos de distribuição de frequência  
    plt.subplot(2, 2, i + 1)  
    sns.histplot(amostra[coluna_dados], bins=10, kde=True, color=np.random.rand(3,))  
    plt.title(f"Distribuição da Amostra ({proporcao*100:.2f}%)")  

# Ajustar layout dos gráficos  
plt.tight_layout()  
plt.show()  



