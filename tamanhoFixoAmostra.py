import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro

# Carregar dados
df = pd.read_csv("treecover_loss__ha.csv")
dados = df['umd_tree_cover_loss__ha'].dropna().values

# Estatísticas populacionais
media_pop = np.mean(dados)
variancia_pop = np.var(dados, ddof=0)
desvio_padrao_pop = np.std(dados, ddof=0)

# Ajuste no tamanho da amostra e número de amostras
tamanho_amostra = max(10, int(0.01 * len(dados)))
num_amostras = min(100 if tamanho_amostra == int(0.01 * len(dados)) else 200, len(dados) // tamanho_amostra)

# Gerar amostras aleatórias
amostras = [np.random.choice(dados, size=tamanho_amostra, replace=True) for _ in range(num_amostras)]

# Estatísticas das médias amostrais
medias_amostrais = np.array([np.mean(amostra) for amostra in amostras])
somas_amostrais = np.array([np.sum(amostra) for amostra in amostras])

# Cálculo da esperança das estatísticas amostrais
esperanca_medias = np.mean(medias_amostrais)
esperanca_variancias = np.mean([np.var(amostra, ddof=1) for amostra in amostras])
esperanca_desvio_padrao = np.mean([np.std(amostra, ddof=1) for amostra in amostras])

# Estatísticas das somas das amostras
media_somas = np.mean(somas_amostrais)
variancia_somas = np.var(somas_amostrais, ddof=1)
desvio_somas = np.std(somas_amostrais, ddof=1)

# Comparação com soma das medidas das amostras
soma_medias = esperanca_medias * tamanho_amostra
soma_variancias = esperanca_variancias * tamanho_amostra
soma_desvio_padrao = esperanca_desvio_padrao * np.sqrt(tamanho_amostra)

# Intervalo de Confiança
nivel_confianca = 0.95
t = stats.t.ppf(1 - (1 - nivel_confianca) / 2, df=tamanho_amostra-1)
erro_padrao = desvio_padrao_pop / np.sqrt(tamanho_amostra)
margem_erro = t * erro_padrao
intervalo_confianca = (media_pop - margem_erro, media_pop + margem_erro)

# Verificando se as médias amostrais estão dentro do intervalo de confiança
contagem_dentro_ic = sum(1 for media in medias_amostrais if intervalo_confianca[0] <= media <= intervalo_confianca[1])
proporcao_dentro_ic = (contagem_dentro_ic / num_amostras) * 100

# Teste de Normalidade das Médias Amostrais
estatistica, p_valor = shapiro(medias_amostrais)

# Gráficos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(medias_amostrais, bins=20, density=True, alpha=0.6, color='b', edgecolor='black')
x = np.linspace(min(medias_amostrais), max(medias_amostrais), 100)
plt.plot(x, stats.norm.pdf(x, np.mean(medias_amostrais), np.std(medias_amostrais)), color='black')
plt.xlabel('Média Amostral')
plt.ylabel('Densidade')
plt.title('Distribuição das Médias Amostrais')

plt.subplot(1, 2, 2)
plt.hist(somas_amostrais, bins=20, density=True, alpha=0.6, color='g', edgecolor='black')
x = np.linspace(min(somas_amostrais), max(somas_amostrais), 100)
plt.plot(x, stats.norm.pdf(x, np.mean(somas_amostrais), np.std(somas_amostrais)), color='black')
plt.xlabel('Soma das Amostras')
plt.ylabel('Densidade')
plt.title('Distribuição das Somas das Amostras')

plt.tight_layout()
plt.show()

# Exibir Resultados
print(f"Intervalo de Confiança (t-Student): {intervalo_confianca}")
print(f"Proporção de Médias dentro do IC: {proporcao_dentro_ic:.2f}%")
print(f"Esperança das Médias Amostrais: {esperanca_medias:.2f}")
print(f"Esperança das Variâncias Amostrais: {esperanca_variancias:.2f}")
print(f"Esperança dos Desvios Padrão Amostrais: {esperanca_desvio_padrao:.2f}")
print(f"Média das Somas: {media_somas:.2f}")
print(f"Variância das Somas: {variancia_somas:.2f}")
print(f"Desvio Padrão das Somas: {desvio_somas:.2f}")
print("\nComparação com a soma das medidas das amostras:")
print(f"Soma das Médias Amostrais: {soma_medias:.2f}")
print(f"Soma das Variâncias Amostrais: {soma_variancias:.2f}")
print(f"Soma dos Desvios Padrão Amostrais: {soma_desvio_padrao:.2f}")
print("\nTeste de Normalidade Shapiro-Wilk para as Médias Amostrais:")
print(f"Estatística: {estatistica:.4f}, p-valor: {p_valor:.4f}")
if p_valor > 0.05:
    print("Não rejeitamos a hipótese nula: As médias amostrais seguem uma distribuição normal.")
else:
    print("Rejeitamos a hipótese nula: As médias amostrais não seguem uma distribuição normal.")










