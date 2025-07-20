import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB




caminho_arquivo = '/Users/brunaneiva/Desktop/ANDOC/train.csv'
dados = pd.read_csv(caminho_arquivo)

#Visualizar os nomes das variáveis
#print(dados.head())
#print(dados.info())
#print(dados.shape)

# média, desvio padrão, max, min, quartis das variaveis numericas
estatisticas_descritivas = dados.describe()
#print(estatisticas_descritivas)

# Verificar se há valores ausentes em cada coluna
valores_ausentes = dados.isnull().sum()
#print(valores_ausentes)

# Exibir informações sobre o DataFrame, incluindo contagem de não nulos 
#print(dados.info())
# não há valores ausentes

# Exibe os tipos de dados das variáveis no DataFrame
tipos_de_dados = dados.dtypes
#print(tipos_de_dados)

duplicate_entries = dados[dados.duplicated()]
#print(duplicate_entries.shape)
# número de linhas duplicadas:
#print("Número de Colunas Duplicadas :",dados.duplicated().sum())


# Vamos identificar as variáveis numéricas
variaveis_numericas = dados.select_dtypes(include=['int', 'float']).columns

# Defina o número de colunas e linhas para organizar os subplots
num_cols = 4
num_rows = len(variaveis_numericas) // num_cols + 1

# Crie a figura e os subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))

# Ajuste a aparência da figura
fig.subplots_adjust(hspace=0.5)

# Preencha os subplots com os histogramas
for i, coluna in enumerate(variaveis_numericas):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].hist(dados[coluna], bins=30, color='skyblue', edgecolor='black')
    axs[row, col].set_title(f'Histograma de {coluna}')
    axs[row, col].set_xlabel(coluna)
    axs[row, col].set_ylabel('Frequência')

# Exibir a imagem
plt.savefig('/Users/brunaneiva/Desktop/ANDOC/histogramas.png')
#plt.show()

num_linhas = len(variaveis_numericas) // 4 + (len(variaveis_numericas) % 4 > 0)

# Crie a figura e os subplots
fig, axs = plt.subplots(nrows=num_linhas, ncols=4, figsize=(20, 6 * num_linhas))

# Identifique as variáveis que têm outliers
variaveis_com_outliers = []

# Itere sobre as variáveis numéricas e crie um boxplot para cada uma
for i, coluna in enumerate(variaveis_numericas):
    row = i // 4
    col = i % 4
    sns.boxplot(x=dados[coluna], ax=axs[row, col], color='skyblue')
    axs[row, col].set_title(f'Boxplot de {coluna}')

print(f'Variáveis com outliers: {variaveis_com_outliers}')
# Ajuste o layout da figura
plt.savefig('/Users/brunaneiva/Desktop/ANDOC/boxplot.png')
plt.tight_layout()

# Exiba a figura
plt.show()

# Selecione apenas as variáveis numéricas
dados_numericos = dados.select_dtypes(include=['float64', 'int64'])

# Crie a matriz de correlação
matriz_correlacao = dados_numericos.corr()

# Configure o estilo da visualização
sns.set(style="white") 

# Crie um mapa de calor para visualização
plt.figure(figsize=(15, 12))
sns.heatmap(matriz_correlacao, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Matriz de Correlação - Variáveis Numéricas")
plt.savefig('/Users/brunaneiva/Desktop/ANDOC/matrizcorr.png')
plt.show()

# visualizar quais sao as variaveis categoricas
categoricas = dados.select_dtypes(include=['object']).columns
dados['Loan Title'] = dados['Loan Title'].str.lower()

#print(categoricas)
# visualizar quais as categorias de cada variavel categorica e quantas são
#for coluna in categoricas:
    #print(f'Categorias em {coluna}:\n{dados[coluna].value_counts()}\n')
    #print(f'{coluna} tem {dados[coluna].nunique()} categorias únicas.')

# Gráfico para ver as variaveis categoricas     
 #for coluna in categoricas:
   # plt.figure(figsize=(10, 6))
    #dados[coluna].value_counts().plot(kind='bar', color='skyblue')
   # plt.title(f'Contagem de Categorias em {coluna}')
   # plt.xlabel(coluna)
   # plt.ylabel('Contagem')
   # plt.show()
       

# Escolha as três variáveis categóricas com mais categorias
variaveis_categoricas = ['Batch Enrolled', 'Sub Grade', 'Loan Title']


# Itere sobre cada variável categórica
for variavel in variaveis_categoricas:
    # Calcule o top 10 de categorias
    top_10_categorias = dados[variavel].value_counts().head(10).index

    # Substitua as categorias fora do top 10 por "Outros" para cada variável
    dados.loc[~dados[variavel].isin(top_10_categorias), variavel] = 'Outros'

# Crie os gráficos de barras para as top 10 categorias de cada variável
for variavel in variaveis_categoricas:
    # Crie o gráfico de barras
    plt.figure(figsize=(12, 6))
    sns.countplot(x=variavel, data=dados, palette='viridis')
    plt.title(f'Top 10 Categorias em {variavel}')
    plt.xlabel(variavel)
    plt.ylabel('Contagem')
    plt.xticks(rotation=45, ha='right')  # Ajuste a rotação dos rótulos do eixo x

    # Exiba o gráfico
    #plt.show()
print(f"As top 10 categorias em {variaveis_categoricas} são:\n{top_10_categorias}")

#Teste Qui-Quadrado

# Lista de variáveis categóricas
variaveis_categoricas = dados.select_dtypes(include='object').columns

# Função para aplicar o teste qui-quadrado entre pares de variáveis
def teste_chi_quadrado(variavel1, variavel2):
    tabela_contingencia = pd.crosstab(dados[variavel1], dados[variavel2])
    _, p_valor, _, _ = chi2_contingency(tabela_contingencia)
    return p_valor

# Nível de significância ajustado para Bonferroni
niveis_significancia_ajustados = 0.05 / (len(variaveis_categoricas) * (len(variaveis_categoricas) - 1) / 2)

# Iteração sobre pares de variáveis categóricas
for i in range(len(variaveis_categoricas)):
    for j in range(i+1, len(variaveis_categoricas)):
        variavel1 = variaveis_categoricas[i]
        variavel2 = variaveis_categoricas[j]
        
        # Aplicar o teste qui-quadrado
        p_valor = teste_chi_quadrado(variavel1, variavel2)
        
        # Exibir resultados
        print(f'Teste Qui-Quadrado entre {variavel1} e {variavel2}: p-valor = {p_valor:.4f}')

        # Verificar significância após ajuste de Bonferroni
        if p_valor < niveis_significancia_ajustados:
            print('Associação significativa após ajuste de Bonferroni')
        else:
            print('Não há associação significativa após ajuste de Bonferroni')
        print()



# Data Preparation

#Excluir as variaveis ID, Payment Plan e Accounts Deliquent
variaveis_a_excluir = ['ID', 'Payment Plan', 'Accounts Delinquent']

# Excluir as colunas do DataFrame original
dados.drop(variaveis_a_excluir, axis=1, inplace=True)


# Normalização dos dados 
# Separar variáveis contínuas e categóricas automaticamente
variaveis_continuas = dados.select_dtypes(include=['float64', 'int64']).columns
variaveis_categoricas = dados.select_dtypes(include=['object']).columns

# Excluir a variável dependente das variáveis contínuas
variaveis_continuas = variaveis_continuas.difference(['Loan Status'])

# Padronizar variáveis contínuas
scaler = StandardScaler()
dados[variaveis_continuas] = scaler.fit_transform(dados[variaveis_continuas])

# Normalizar variáveis categóricas usando Label Encoding
label_encoder = LabelEncoder()
dados[variaveis_categoricas] = dados[variaveis_categoricas].apply(lambda x: label_encoder.fit_transform(x.astype(str)))

####MODELOS


# Separar as features (X) e a variável alvo (y)
X = dados.drop('Loan Status', axis=1)
y = dados['Loan Status']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Certificar se os rótulos são inteiros
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Modelos
modelos = {
    'Regressão Logística': LogisticRegression(solver='liblinear'),
    'Árvore de Decisão': DecisionTreeClassifier(),
    'Floresta Aleatória': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Máquina de Vetores de Suporte (SVM)': SVC(probability=True),
    'Redes Neurais': MLPClassifier(max_iter=1000),
    'Naive Bayes': GaussianNB()
}

# Treinar e avaliar cada modelo
for nome, modelo in modelos.items():
    print(f"\n{nome}:")
    modelo.fit(X_train, y_train)
    previsoes = modelo.predict(X_test)
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, previsoes))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, previsoes, zero_division=1))
    

# Calcular as probabilidades previstas
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    # Calcular a curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, probabilidades)

    # Calcular a área sob a curva ROC (AUC-ROC)
    roc_auc = auc(fpr, tpr)

    # Plotar a curva ROC
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC para {nome}')
    plt.legend(loc='lower right')
    plt.show()