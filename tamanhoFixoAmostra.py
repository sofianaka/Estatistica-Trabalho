import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


df = pd.read_csv("treecover_loss__ha.csv")

print("Colunas disponíveis:", df.columns)

dados = df['umd_tree_cover_loss__ha'].dropna().values

# (01) Definir tamanho da amostra (0,5% ou 1% da população) e garantir pelo menos 2 elementos
tamanho_populacao = len(dados)
tamanho_amostra = max(2, int(0.005 * tamanho_populacao))  # 0,5% da população, mínimo 2
num_amostras = min(200, tamanho_populacao // tamanho_amostra)  # (02) Garante 100% da população

print(f"Amostras de tamanho fixo com reposição, com número de elementos de {tamanho_amostra}")
print(f"Número total de amostras geradas: {num_amostras}")

# (01) Gerar amostras com reposição
amostras = [np.random.choice(dados, size=tamanho_amostra, replace=True) for _ in range(num_amostras)]

# Calcular estatísticas populacionais
media_pop = np.mean(dados)
variancia_pop = np.var(dados, ddof=0)
desvio_padrao_pop = np.std(dados, ddof=0)

print(f"Média Populacional: {media_pop:.2f}")
print(f"Variância Populacional: {variancia_pop:.2f}")
print(f"Desvio Padrão Populacional: {desvio_padrao_pop:.2f}")

# (03) Calcular estatísticas amostrais
medias_amostrais = np.array([np.mean(amostra) for amostra in amostras])
variancias_amostrais = np.array([np.var(amostra, ddof=1) for amostra in amostras])
desvios_amostrais = np.array([np.std(amostra, ddof=1) for amostra in amostras])

print(f"Esperança das Médias Amostrais: {np.mean(medias_amostrais):.2f}")
print(f"Esperança das Variâncias Amostrais: {np.mean(variancias_amostrais):.2f}")
print(f"Esperança dos Desvios Padrão Amostrais: {np.mean(desvios_amostrais):.2f}")

# (04) Soma das amostras e estatísticas
somas_amostras = np.array([np.sum(amostra) for amostra in amostras])
print(f"Média das Somas: {np.mean(somas_amostras):.2f}")
print(f"Variância das Somas: {np.var(somas_amostras, ddof=1):.2f}")
print(f"Desvio Padrão das Somas: {np.std(somas_amostras, ddof=1):.2f}")

# (05) Gráfico da distribuição das médias amostrais
plt.hist(medias_amostrais, bins=15, density=True, alpha=0.6, color='b', edgecolor='black')
x = np.linspace(min(medias_amostrais), max(medias_amostrais), 100)
plt.plot(x, stats.norm.pdf(x, np.mean(medias_amostrais), np.std(medias_amostrais)), color='black')
plt.xlabel('Média Amostral')
plt.ylabel('Densidade')
plt.title('Distribuição das Médias Amostrais (Teorema Central do Limite)')
plt.show()

# (06) Intervalo de confiança para a média amostral
nivel_confianca = 0.95
z = stats.norm.ppf(1 - (1 - nivel_confianca) / 2)
margem_erro = z * (desvio_padrao_pop / np.sqrt(tamanho_amostra))
intervalo_confianca = (float(media_pop - margem_erro), float(media_pop + margem_erro))
print(f"Intervalo de Confiança: {intervalo_confianca}")

# (07) Contar quantas médias amostrais estão dentro do intervalo de confiança
dentro_ic = sum(1 for media in medias_amostrais if intervalo_confianca[0] <= media <= intervalo_confianca[1])
proporcao_dentro_ic = (dentro_ic / num_amostras) * 100

# (08) Verificar se o intervalo contém a média populacional
print(f"Proporção de Médias dentro do IC: {proporcao_dentro_ic:.2f}%")

# (09) Comparação com o nível de confiança desejado
diferenca_conf = abs(proporcao_dentro_ic - (nivel_confianca * 100))
print(f"Diferença entre a proporção observada e o nível de confiança: {diferenca_conf:.2f}%")








