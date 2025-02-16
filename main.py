import pandas as pd

df = pd.read_csv(r"c:\Users\mayum\OneDrive\Área de Trabalho\Estatistica Trabalho\treecover_loss__ha.csv")


print(df.head())  # Verifica as primeiras linhas  
print(df.info())  # Informações sobre tipos de dados  
print(df.describe())  # Estatísticas básicas  
print(df.columns)  # Verificar os nomes das colunas  


coluna_perda = "umd_tree_cover_loss__ha"

# Média da perda florestal por hectar 
media_perda = df[coluna_perda].mean()

# Desvio padrão da perda florestal por hectar
desvio_padrao = df[coluna_perda].std()

# Mediana da perda florestal por hectar
mediana_perda = df[coluna_perda].median()

print(f"Média: {media_perda:.2f}, Desvio Padrão: {desvio_padrao:.2f}, Mediana: {mediana_perda:.2f}")




