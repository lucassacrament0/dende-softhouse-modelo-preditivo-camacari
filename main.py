import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregando os dados
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Visualizando o balanceamento da variável alvo (winner)
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='winner', palette='viridis')
plt.title('Distribuição de Vitórias (0 = Não, 1 = Sim)')
plt.show()

print(f"Total de linhas no Treino: {len(train_df)}")
print(f"Valores nulos encontrados: {train_df.isnull().sum().sum()}")

# Separando as variáveis (X) do Alvo (y)
X = train_df.drop('winner', axis=1)
y = train_df['winner']

# Transformando textos em números (Variáveis Categóricas)
# Selecionando as colunas de texto: team_name, country_code e confederation
X_encoded = pd.get_dummies(X, drop_first=True)

# Divisão de 80% para Treino e 20% para Teste (Para medirmos nossa precisão)
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"Dados para ensinar a máquina: {X_train.shape[0]} linhas")
print(f"Dados para validar a máquina: {X_val.shape[0]} linhas")

# Inicializando o Modelo (O Coração da nossa IA)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando a máquina com 80% dos dados
modelo_rf.fit(X_train, y_train)

# Pedindo para a máquina fazer as predições nos 20% que separamos
previsoes = modelo_rf.predict(X_val)

# Avaliando o quão bem ela foi
acuracia = accuracy_score(y_val, previsoes)
print(f"Acurácia (Taxa de Acerto) do Modelo: {acuracia * 100:.2f}%\n")

# Gerando a Matriz de Confusão visual
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_val, previsoes), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Validação')
plt.ylabel('Realidade (O que realmente aconteceu)')
plt.xlabel('Predição (O que a máquina achou)')
plt.show()


# Aplicando a mesma transformação de texto para número no test_df
X_test_encoded = pd.get_dummies(test_df, drop_first=True)

# ALINHAMENTO DE COLUNAS
# Garante que o teste tenha as mesmas colunas que o modelo aprendeu no treino,
# preenchendo com 0 caso alguma coluna falte.
X_test_aligned = X_test_encoded.reindex(columns=X_train.columns, fill_value=0)

# Extraindo as probabilidades
# O metodo predict_proba retorna duas colunas: [Chance de Perder(0), Chance de Ganhar(1)]
# Queremos a chance de Ganhar, então pegamos a coluna 1 ( [:, 1] )
probabilidades_vitoria = modelo_rf.predict_proba(X_test_aligned)[:, 1]

# Montando o arquivo Bonito para a visualisação
resultado_final = pd.DataFrame({
    'Selecao': test_df['team_name'],
    'Chance_de_Vitoria (%)': (probabilidades_vitoria * 100).round(2) # Transforma em % com 2 casas decimais
})

# Salvando no formato CSV
nome_do_arquivo = 'entrega_final_copa.csv'
resultado_final.to_csv(nome_do_arquivo, index=False)

print(f"✅ SUCESSO! Arquivo '{nome_do_arquivo}' foi gerado na mesma pasta do seu projeto.")
print("\nVeja as 5 primeiras predições:")
print(resultado_final.head())