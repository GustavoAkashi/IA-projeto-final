
from tree import Tree, Node
import pandas as pd
import numpy as np
import math

def vetorize(data):
    vector = []
    for i in range(22):
        vector.append(0)
    frequencies = {}
    for row in data["label"]:
        if row in frequencies:
            frequencies[row]+=1
        else:
            frequencies[row] = 1
    
    for key, value in frequencies.items():
        vector[key] = value
    
    print(vector)

def calc_entropia(data):
    total = len(data)
    frequencies = {}
    entropia = 0
    for row in data["label"]:
        if row in frequencies:
            frequencies[row]+=1
        else:
            frequencies[row] = 1
    
    for key, value in frequencies.items():
        pr = value/total
        entropia += pr*math.log2(pr)
    
    return entropia*-1


def calc_ganho(entropia_pai, node_pai, node_filho_esq, node_filho_dir):
    ganho = entropia_pai

    entropia_filho_esq = calc_entropia(node_filho_esq.data)
    entropia_filho_dir = calc_entropia(node_filho_dir.data)

    ganho -= entropia_filho_esq*(len(node_filho_esq.data)/len(node_pai.data))
    ganho -= entropia_filho_dir*(len(node_filho_dir.data)/len(node_pai.data))

    return ganho

def build(node):
    entropia_pai = calc_entropia(node.data)
    # Nesse caso chegamos em um nó folha, pois todos os elementos pertencentes
    # a esse nó são da mesma classe.
    if entropia_pai == 0:
        node.answer = node.data["label"].unique()[0]
        return node
    
    # Variáveis que serão usadas para guardar o nó que obteve
    # o melhor ganho
    columns = node.data.columns.values.tolist()
    best_gain = -1
    best_condition = ""
    best_filho_dir = None
    best_filho_esq = None

    for column_name in columns[:-1]:
        # Percorre todos os elementos de todas as colunas e testa
        # um a um, guardando quem obtiver o melhor ganho.
        for possible_splitter in node.data[column_name].unique():
            possible_value_to_split = possible_splitter

            # Define uma condição de separação baseada no elemento
            # da coluna atual
            curr_condition = f"{column_name} <= {possible_value_to_split}"

            # Se a condicao for verdadeira
            data_filho_dir = node.data[node.data[column_name] <= possible_value_to_split]
            node_dir = Node(data_filho_dir, "")

            # Se a condicao nao for verdadeira
            data_filho_esq = pd.concat([data_filho_dir,node.data]).drop_duplicates(keep=False)
            node_esq = Node(data_filho_esq, "")

            # Calcula o ganho caso a condição de separação
            # seja escolhida.
            curr_gain = calc_ganho(entropia_pai, node, node_esq, node_dir)

            # Caso o ganho seja maior que nosso melhor ganho,
            # guarda todas as variaveis necessárias pra criar
            # a ramificação no nó.
            if curr_gain > best_gain:
                best_gain = curr_gain
                best_condition = curr_condition
                best_filho_dir = node_dir
                best_filho_esq = node_esq
    

    # Define a condição do nó atual.
    node.condition = best_condition

    # Repete o processo de criação para os nós
    # filhos.
    node.left = build(best_filho_esq)
    node.right = build(best_filho_dir)
    return node
    

def discretiza_dados(data: pd.DataFrame):
    mapeamento_tipo = {
        "rice": 0,
        "maize": 1,
        "jute": 2,
        "cotton": 3,
        "coconut": 4,
        "papaya": 5,
        "orange": 6,
        "apple": 7,
        "muskmelon": 8,
        "watermelon": 9,
        "grapes": 10,
        "mango": 11,
        "banana": 12,
        "pomegranate": 13,
        "lentil": 14,
        "blackgram": 15,
        "mungbean": 16,
        "mothbeans": 17,
        "pigeonpeas": 18,
        "kidneybeans": 19,
        "chickpea": 20,
        "coffee": 21,
    }

    for key, value in mapeamento_tipo.items():
        data["label"] = data["label"].replace(key, value)
    
    data["label"] = data["label"].astype(int)
    return data

def divide_dados_treino(data, porcentagem):
    data = data.iloc[np.random.permutation(len(data))]
    tamanho_total = len(data)
    tamanho_treino = math.floor(tamanho_total*porcentagem)
    return data.iloc[:tamanho_treino], data.iloc[tamanho_treino:]

def testa_entrada(tree, entrada):
    return traverse_tree(tree.root, entrada)

def traverse_tree(actual_node, entrada):
    if actual_node.answer != "":
        if float(entrada["label"].unique()[0]) == float(actual_node.answer):
            return True
        else:
            return False
    
    condition_splitted = actual_node.condition.split("<=")
    if float(entrada[condition_splitted[0].strip()]) <= float(condition_splitted[1].strip()):
        return traverse_tree(actual_node.right, entrada)
    else:
        return traverse_tree(actual_node.left, entrada)

if __name__ == "__main__":
    data = pd.read_csv('dataset.csv')
    data = discretiza_dados(data)
    xizes = []
    yizes = []
    data_treino, data_teste = divide_dados_treino(data, 0.7)
    tree = Tree(data)
    build(tree.root)
    columns = data.columns.values.tolist()
    print(tree.root.condition)
    correct = 0
    wrong = 0
    for row in data_teste.itertuples():
        valid = testa_entrada(tree, pd.DataFrame([row[1:]], columns=columns))
        if valid == True:
            correct+=1
        else:
            wrong+=1
    print(f"""
        There were {correct} correct test cases and {wrong} wrong test cases.
        Accuracy: {correct/(correct+wrong)}
    """)
    
    print(xizes)
    print(yizes)
