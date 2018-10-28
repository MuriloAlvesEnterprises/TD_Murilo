# -*- coding: utf-8 -*-
"""
Análise do desempenho de algoritmos de aprendizagem por reforço na solução do problema de roteamento de veículos capacitados utilizando traço de elegibilidade

Base de dados: CVRPLIB
http://vrp.atd-lab.inf.puc-rio.br/index.php/en/

@author: Murilo Alves
"""

import time
from random import uniform, choice
from math import pow, sqrt
import numpy
import matplotlib.pyplot as plt
import statistics

def Recompensa(distancia, demanda, capacidadeVeiculo):
    """
    Recompensa da ação escolhida.
    
    Entrada:
        distancia: Custo de sair do estado atual e ir para o próximo
        demanda: Demanda do próximo estado
        capacidadeVeiculo: Capacidade máxima do veículo
        
    Retorno:
        Valor da recompensa (negativa)
    """
    return - distancia/(capacidadeVeiculo - demanda)

def ValidaAcoes (acoesPossiveis, listEstadosVisitados, demandaRotaAcumulada, ambiente, VeiculosDinamicos):
    """
    Válida ações para serem escolhidas no estado atual, caso não for um estado visitado e no caso de veículos dinâmicos 
    a demanda não torna a rota inválida.
    
    Entrada:
        acoesPossiveis: Todas as ações que o estado atual possui
        listEstadosVisitados: Lista de estados já visitados
        demandaRotaAcumulada: Demanda acumulada pela rota
        ambiente: Informações sobre o ambiente
        VeiculosDinamicos: Condição para considerar se a demanda não torna a rota inválida
        
    Retorno:
        acoes: Valores das ações, sendo '-inf' para ações inválidas
        permissao: Lista informando quais ações são válidias, 0 para inválida e 1 para o contrário
    """
    acoes = [] 
    permissao = [] 
    
    for i in range(len(acoesPossiveis)):
        if (listEstadosVisitados[i] == True or (demandaRotaAcumulada + ambiente["Estados"][i]["Demanda"] > ambiente["Capacidade"] and VeiculosDinamicos == True)):
            acoes.append(float('-inf'))
            permissao.append(0)
        else:
            acoes.append(acoesPossiveis[i])
            permissao.append(1)
    
    return acoes, permissao

def Politica(epsilon, acoes, permissao): 
    """
    Escolhe uma ação, aleatoriamente ou pelo maior valor.
    
    Entrada:
        epsilon: Parâmetro para probalidade de decição do metódo de escolha da ação
        acoes: Valores das ações, sendo '-inf' para ações inválidas
        permissao: Lista informando quais ações são válidias, 0 para inválida e 1 para o contrário
        
    Retorno:
        Posição (index) da ação na lista de ações 
    """
    if (uniform(0, 1) > epsilon): # Aleatoriedade
        probabilidade = []
        somatoria = sum(permissao)
        for i in range(len(permissao)):
            probabilidade.append(permissao[i]/somatoria)

        acao = numpy.random.choice(acoes,1,p=probabilidade)
            
        return acoes.index(acao) 
    else: # Maior valor
        return acoes.index(max(acoes)) 
    
def MaxQ (acoes):
    """
    Escolhe uma ação pelo maior valor.
    
    Entrada:
        acoes: Valores das ações, sendo '-inf' para ações inválidas
        
    Retorno:
        Posição (index) da ação na lista de ações 
    """
    return acoes.index(max(acoes)) 

def TaxaAprendizagem (visitas):
    """
    Taxa de aprendizagem baseada em visitas feitas ao par Q(estado, ação)
    
    Entrada:
        visitas: Quantidades de visitas ao par Q(estado, ação)
        
    Retorno:
        Cálculo da taxa
    """
    return 1/(1 + visitas)

def CriaMatriz(quantidadeEstados):
    """
    Inicializa a matriz com tamanho estados x estados ou Q(s,a)
    
    Entrada:
        quantidadeEstados: Quantidades de estados do ambiente
        
    Retorno:
        Matriz zerada com tamanho estados x estados
    """
    matriz = [0]*quantidadeEstados
    
    for i in range(quantidadeEstados):
        matriz[i] = [0]*quantidadeEstados
        
    return matriz
 
def EscolheRota(rotas):
    """
    Seleciona a rota com a menor demanda
    
    Entrada:
        rotas: Informações sobre todas as rotas
        
    Retorno:
        Posição (index) da rota com menor demanda
    """
    veiculos = []
    
    for i in range(len(rotas)):
        veiculos.append(rotas[i]["Demanda"])

    return veiculos.index(min(veiculos))

def CriaRotas(quantidadeVeiculos):
    """
    Inicializa uma lista (tamanho K veículos) com informações de demanda, custo e uma lista de estados visitados 
    
    Entrada:
        quantidadeVeiculos: Quantidades de veículos do ambiente
        
    Retorno:
        Matriz com demanda e custo zeradas, lista de consumidores inicializa com 0 (depósito)
    """
    rotas = []
    
    for i in range(quantidadeVeiculos):
        rotas.append({"Demanda": 0, "Custo": 0, "Consumidores": [0]})

    return rotas

def AtualizaRota(rotaAtual, estado, acao, ambiente):
    """
    Cálcula a distância euclidiana entre o estado e acão, além disso atualiza o custo, a demanda e a lista visitados da rota
    
    Entrada:
        rotaAtual: Rota atual para ser atualizada
        estado: Posição do estado atual
        acao: Posição da ação atual
        ambiente: Informações sobre o ambiente
        
    Retorno:
        Rota atualizada e a distância euclidiana
    """
    # Cálcula a distancia euclidiana entre dois pontos
    x = pow(ambiente["Estados"][estado]["CoordX"] - ambiente["Estados"][acao]["CoordX"], 2)
    y = pow(ambiente["Estados"][estado]["CoordY"] - ambiente["Estados"][acao]["CoordY"], 2)
    distancia = sqrt(x + y)
    
    # Atualiza o custo da rota
    rotaAtual["Custo"] = rotaAtual["Custo"] + distancia
    
    # Atualiza a demanda da rota
    rotaAtual["Demanda"] = rotaAtual["Demanda"] + ambiente["Estados"][acao]["Demanda"]
    
    # Atualiza a sequência de consumidores da rota
    rotaAtual["Consumidores"].append(acao)

    return rotaAtual, distancia
    
# Algoritmo Q-Learning com traço de elegibilidade (veículos dinâmicos)
def Q_Learning_VeiculosDinamicos(ambiente, lambd, taxaDesconto = 0.1, epsilon = 0.9, epocas = 1000):
    # Quantidade de estados do ambiente
    quantidadeEstados = len(ambiente["Estados"])
    
    # Matriz Q(s,a)
    Q = CriaMatriz(quantidadeEstados)
    QVisitas = CriaMatriz(quantidadeEstados)
    
    #Armazenar os resultados
    resultados = []

    for i in range(epocas):
        # Lista de estados visitados (consumidores)
        listEstadosVisitados = quantidadeEstados*[False]
    
        # Estado inicial do ambiente (depósito)
        listEstadosVisitados[0] = True

        # Metricas da rota
        rotas = []
        rotas.append({"Demanda": 0, "Custo": 0, "Consumidores": [0]})
        veiculo = 0
        
        # Estado inicial
        estado = 0
        
        QElegibilidade = CriaMatriz(quantidadeEstados)
        
        while (listEstadosVisitados.count(False) != 0):
            # Validar ações
            acoes, permissao = ValidaAcoes(Q[estado], listEstadosVisitados, rotas[veiculo]["Demanda"], ambiente, True)
            
            # Escolhe uma ação
            acao = Politica(epsilon, acoes, permissao)
            
            # Cálcula a distância euclidiana e atualiza a rota
            rotas[veiculo], distancia = AtualizaRota(rotas[veiculo], estado, acao, ambiente)
            
            # Marca a ação para não se escolhida de novo e atualiza a lista de ações válidas
            listEstadosVisitados[acao] = True
            acoes[acao] = float('-inf')
            
            # Atualiza a elegibilidade ao par (s,a)
            QElegibilidade[estado][acao] = QElegibilidade[estado][acao] + 1
            
            # Atualiza a quantidade de visitas ao par (s,a)
            QVisitas[estado][acao] = QVisitas[estado][acao] + 1
            
            # Valor da possível próxima ação
            if (acoes.count(float('-inf')) == len(acoes)): # Se ação é o depósito
                valorAcaoFutura = 0
            else:
                valorAcaoFutura = Q[acao][MaxQ(acoes)]
            
            if (lambd == 0):
                Q[estado][acao] = Q[estado][acao] + TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q[estado][acao])
            else:
                for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                    u = rotas[veiculo]["Consumidores"][j-1]
                    v = rotas[veiculo]["Consumidores"][j]
                    # Atualiza Q com elegibilidade
                    Q[u][v] = Q[u][v] + TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q[estado][acao])*QElegibilidade[u][v]
                    QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
            
            # Continua ou cria uma nova rota
            if (acao != 0):
                listEstadosVisitados[0] = False
            else:
                if (listEstadosVisitados.count(False) != 0):
                    rotas.append({"Demanda": 0, "Custo": 0, "Consumidores": [0]})
                    veiculo = veiculo + 1
            
            # Atualiza o estado com a ação
            estado = acao
    
        # Cálculo da distância total das rotas geradas
        distanciaTotal = 0
        
        for veiculo in range(len(rotas)):
            if (len(rotas) > ambiente["Veiculos"]):
                distanciaTotal = float('inf')
            
            distanciaTotal = distanciaTotal + rotas[veiculo]["Custo"]
        
        resultados.append(distanciaTotal)
        if (i == 0):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
        if (distanciaTotal < menorDistancia):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
    return menorDistancia, menorRotas, resultados
        
# Algoritmo Double Q-Learning com traço de elegibilidade (veículos dinâmicos)
def DoubleQ_Learning_VeiculosDinamicos(ambiente, lambd, taxaDesconto = 0.1, epsilon = 0.9, epocas = 1000):
    # Quantidade de estados do ambiente
    quantidadeEstados = len(ambiente["Estados"])
    
    # Matriz Q(s,a)
    Q1 = CriaMatriz(quantidadeEstados)
    Q2 = CriaMatriz(quantidadeEstados)
    QVisitas = CriaMatriz(quantidadeEstados)
    
    #Armazenar os resultados
    resultados = []

    for i in range(epocas):
        # Lista de estados visitados (consumidores)
        listEstadosVisitados = quantidadeEstados*[False]
    
        # Estado inicial do ambiente (depósito)
        listEstadosVisitados[0] = True

        # Metricas da rota
        rotas = []
        rotas.append({"Demanda": 0, "Custo": 0, "Consumidores": [0]})
        veiculo = 0
        
        # Estado inicial
        estado = 0
        
        QElegibilidade = CriaMatriz(quantidadeEstados)
        
        while (listEstadosVisitados.count(False) != 0):
            # Validar ações
            acoes, permissao = ValidaAcoes([elemA + elemB for elemA, elemB in zip(Q1[estado], Q2[estado])], listEstadosVisitados, rotas[veiculo]["Demanda"], ambiente, True)
            
            # Escolhe uma ação
            acao = Politica(epsilon, acoes, permissao)

            # Cálcula a distância euclidiana e atualiza a rota
            rotas[veiculo], distancia = AtualizaRota(rotas[veiculo], estado, acao, ambiente)
            
            # Marca a ação para não se escolhida de novo e atualiza a lista de ações válidas
            listEstadosVisitados[acao] = True
            acoes[acao] = float('-inf')
            
            # Atualiza a elegibilidade ao par (s,a)
            QElegibilidade[estado][acao] = QElegibilidade[estado][acao] + 1
            
            # Atualiza a quantidade de visitas ao par (s,a)
            QVisitas[estado][acao] = QVisitas[estado][acao] + 1
            
            if (choice(["1","2"]) == "1"):
                # Valor da possível próxima ação
                if (acoes.count(float('-inf')) == len(acoes)): # Se ação é o depósito
                    valorAcaoFutura = 0
                else:
                    valorAcaoFutura = Q2[acao][MaxQ(acoes)]
                
                if (lambd == 0):
                    Q1[estado][acao] = Q1[estado][acao]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q1[estado][acao])
                else:
                    for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                        u = rotas[veiculo]["Consumidores"][j-1]
                        v = rotas[veiculo]["Consumidores"][j]
                        # Atualiza Q com elegibilidade
                        Q1[u][v] = Q1[u][v]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q1[estado][acao])*QElegibilidade[u][v]
                        QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
            else:
                # Valor da possível próxima ação
                if (acoes.count(float('-inf')) == len(acoes)): # Se ação é o depósito
                    valorAcaoFutura = 0
                else:
                    valorAcaoFutura = Q1[acao][MaxQ(acoes)]
                
                if (lambd == 0):
                    Q2[estado][acao] = Q2[estado][acao]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q2[estado][acao])
                else:
                    for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                        u = rotas[veiculo]["Consumidores"][j-1]
                        v = rotas[veiculo]["Consumidores"][j]
                        # Atualiza Q com elegibilidade
                        Q2[u][v] = Q2[u][v]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q2[estado][acao])*QElegibilidade[u][v]
                        QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]

            # Continua ou cria uma nova rota
            if (acao != 0):
                listEstadosVisitados[0] = False
            else:
                if (listEstadosVisitados.count(False) != 0):
                    rotas.append({"Demanda": 0, "Custo": 0, "Consumidores": [0]})
                    veiculo = veiculo + 1
                
            # Atualiza o estado com a ação
            estado = acao

        # Cálculo da distância total das rotas geradas
        distanciaTotal = 0
        
        for veiculo in range(len(rotas)):
            if (len(rotas) > ambiente["Veiculos"]):
                distanciaTotal = float('inf')
            
            distanciaTotal = distanciaTotal + rotas[veiculo]["Custo"]
        
        resultados.append(distanciaTotal)
        if (i == 0):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
        if (distanciaTotal < menorDistancia):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
    return menorDistancia, menorRotas, resultados

# Algoritmo Q-Learning com traço de elegibilidade (veículos fixos)
def Q_Learning_VeiculosFixos(ambiente, lambd, taxaDesconto = 0.1, epsilon = 0.9, epocas = 1000):
    # Quantidade de estados do ambiente
    quantidadeEstados = len(ambiente["Estados"])
    
    # Matriz Q(s,a)
    Q = CriaMatriz(quantidadeEstados)
    QVisitas = CriaMatriz(quantidadeEstados)
    
    #Armazenar os resultados
    resultados = []

    for i in range(epocas):
        # Lista de estados visitados (consumidores)
        listEstadosVisitados = quantidadeEstados*[False]
    
        # Estado inicial do ambiente (depósito)
        listEstadosVisitados[0] = True

        # Metricas da rota
        rotas = CriaRotas(ambiente["Veiculos"])
        
        QElegibilidade = CriaMatriz(quantidadeEstados)
        
        while (listEstadosVisitados.count(False) != 0):
            # Escolhe um veiculo
            veiculo = EscolheRota(rotas)
            
            # O último estado da rota escolhida
            estado = rotas[veiculo]["Consumidores"][-1]

            # Validar ações
            acoes, permissao = ValidaAcoes(Q[estado], listEstadosVisitados, rotas[veiculo]["Demanda"], ambiente, False)
            
            # Escolhe uma ação
            acao = Politica(epsilon, acoes, permissao)

            # Cálcula a distância euclidiana e atualiza a rota
            rotas[veiculo], distancia = AtualizaRota(rotas[veiculo], estado, acao, ambiente)
            
            # Marca a ação para não se escolhida de novo e atualiza a lista de ações válidas
            listEstadosVisitados[acao] = True
            acoes[acao] = float('-inf')
            
            # Atualiza a elegibilidade ao par (s,a)
            QElegibilidade[estado][acao] = QElegibilidade[estado][acao] + 1
            
            # Atualiza a quantidade de visitas ao par (s,a)
            QVisitas[estado][acao] = QVisitas[estado][acao] + 1
            
            # Valor da possível próxima ação
            if (acoes.count(float('-inf')) == len(acoes)): # Se ação é o depósito
                valorAcaoFutura = Q[acao][0]
            else:
                valorAcaoFutura = Q[acao][MaxQ(acoes)]
            if (lambd == 0):
                Q[estado][acao] = Q[estado][acao] + TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q[estado][acao])
            else:
                for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                    u = rotas[veiculo]["Consumidores"][j-1]
                    v = rotas[veiculo]["Consumidores"][j]
                    # Atualiza Q com elegibilidade
                    Q[u][v] = Q[u][v] + TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q[estado][acao])*QElegibilidade[u][v]
                    QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
        
        distanciaTotal = 0
        
        for veiculo in range(len(rotas)):
            # Calcular as metricas para o retorno ao deposito
            estado = rotas[veiculo]["Consumidores"][-1]
            acao = 0
            
            # Cálcula a distância euclidiana e atualiza a rota
            rotas[veiculo], distancia = AtualizaRota(rotas[veiculo], estado, acao, ambiente)
            
            # Atualiza a elegibilidade ao par (s,a)
            QElegibilidade[estado][acao] = QElegibilidade[estado][acao] + 1
            
            # Atualiza a quantidade de visitas ao par (s,a)
            QVisitas[estado][acao] = QVisitas[estado][acao] + 1
            
            if (lambd == 0):
                Q[estado][acao] = Q[estado][acao] + TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + 0 - Q[estado][acao])
            else:
                for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                    u = rotas[veiculo]["Consumidores"][j-1]
                    v = rotas[veiculo]["Consumidores"][j]
                    # Atualiza Q com elegibilidade
                    Q[u][v] = Q[u][v] + TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + 0 - Q[estado][acao])*QElegibilidade[u][v]
                    QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
            
            if (rotas[veiculo]["Demanda"] > ambiente["Capacidade"]):
                distanciaTotal = float('inf') # inválido
            
            distanciaTotal = distanciaTotal + rotas[veiculo]["Custo"]
        
        resultados.append(distanciaTotal)
        if (i == 0):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
        if (distanciaTotal < menorDistancia):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
    return menorDistancia, menorRotas, resultados

# Algoritmo Double Q-Learning com traço de elegibilidade (veículos fixos)
def DoubleQ_Learning_VeiculosFixos(ambiente, lambd, taxaDesconto, epsilon = 0.9, epocas = 1000):
    # Quantidade de estados do ambiente
    quantidadeEstados = len(ambiente["Estados"])
    
    # Matriz Q(s,a)
    Q1 = CriaMatriz(quantidadeEstados)
    Q2 = CriaMatriz(quantidadeEstados)
    QVisitas = CriaMatriz(quantidadeEstados)
    
    #Armazenar os resultados
    resultados = []

    for i in range(epocas):
        # Lista de estados visitados (consumidores)
        listEstadosVisitados = quantidadeEstados*[False]
    
        # Estado inicial do ambiente (depósito)
        listEstadosVisitados[0] = True

        # Metricas da rota
        rotas = CriaRotas(ambiente["Veiculos"])
        
        QElegibilidade = CriaMatriz(quantidadeEstados)
        
        while (listEstadosVisitados.count(False) != 0):
            # Escolhe um veiculo
            veiculo = EscolheRota(rotas)
            
            # O último estado da rota escolhida
            estado = rotas[veiculo]["Consumidores"][-1]
            
            # Validar ações
            acoes, permissao = ValidaAcoes([elemA + elemB for elemA, elemB in zip(Q1[estado], Q2[estado])], listEstadosVisitados, rotas[veiculo]["Demanda"], ambiente, False)
            
            # Escolhe uma ação
            acao = Politica(epsilon, acoes, permissao)

            # Cálcula a distância euclidiana e atualiza a rota
            rotas[veiculo], distancia = AtualizaRota(rotas[veiculo], estado, acao, ambiente)
            
            # Marca a ação para não se escolhida de novo e atualiza a lista de ações válidas
            listEstadosVisitados[acao] = True
            acoes[acao] = float('-inf')
            
            # Atualiza a elegibilidade ao par (s,a)
            QElegibilidade[estado][acao] = QElegibilidade[estado][acao] + 1
            
            # Atualiza a quantidade de visitas ao par (s,a)
            QVisitas[estado][acao] = QVisitas[estado][acao] + 1
            
            if (choice(["1","2"]) == "1"):
                # Valor da possível próxima ação
                if (acoes.count(float('-inf')) == len(acoes)): # Se ação é o depósito
                    valorAcaoFutura = Q2[acao][0]
                else:
                    valorAcaoFutura = Q2[acao][MaxQ(acoes)]
                    
                if (lambd == 0):
                    Q1[estado][acao] = Q1[estado][acao]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q1[estado][acao])
                else:
                    for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                        u = rotas[veiculo]["Consumidores"][j-1]
                        v = rotas[veiculo]["Consumidores"][j]
                        # Atualiza Q com elegibilidade
                        Q1[u][v] = Q1[u][v]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q1[estado][acao])*QElegibilidade[u][v]
                        QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
            else:
                # Valor da possível próxima ação
                if (acoes.count(float('-inf')) == len(acoes)): # Se ação é o depósito
                    valorAcaoFutura = Q1[acao][0]
                else:
                    valorAcaoFutura = Q1[acao][MaxQ(acoes)]
                if (lambd == 0):
                    Q2[estado][acao] = Q2[estado][acao]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q2[estado][acao])
                else:  
                    for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                        u = rotas[veiculo]["Consumidores"][j-1]
                        v = rotas[veiculo]["Consumidores"][j]
                        # Atualiza Q com elegibilidade
                        Q2[u][v] = Q2[u][v]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + taxaDesconto*valorAcaoFutura - Q2[estado][acao])*QElegibilidade[u][v]
                        QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
                
        distanciaTotal = 0
        
        for veiculo in range(len(rotas)):
            # Calcular as metricas para o retorno ao deposito
            estado = rotas[veiculo]["Consumidores"][-1]
            acao = 0
            
            # Cálcula a distância euclidiana e atualiza a rota
            rotas[veiculo], distancia = AtualizaRota(rotas[veiculo], estado, acao, ambiente)
            
            # Atualiza a elegibilidade ao par (s,a)
            QElegibilidade[estado][acao] = QElegibilidade[estado][acao] + 1
            
            # Atualiza a quantidade de visitas ao par (s,a)
            QVisitas[estado][acao] = QVisitas[estado][acao] + 1
            
            if (choice(["1","2"]) == "1"):
                if (lambd == 0):
                    Q1[estado][acao] = Q1[estado][acao]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + 0 - Q1[estado][acao])
                else:
                    for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                        u = rotas[veiculo]["Consumidores"][j-1]
                        v = rotas[veiculo]["Consumidores"][j]
                        # Atualiza Q com elegibilidade
                        Q1[u][v] = Q1[u][v]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + 0 - Q1[estado][acao])*QElegibilidade[u][v]
                        QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
            else:
                if (lambd == 0):
                    Q2[estado][acao] = Q2[estado][acao]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + 0 - Q2[estado][acao])
                else:
                    for j in range(len(rotas[veiculo]["Consumidores"]) - 1, 0, -1):
                        u = rotas[veiculo]["Consumidores"][j-1]
                        v = rotas[veiculo]["Consumidores"][j]
                        # Atualiza Q com elegibilidade
                        Q2[u][v] = Q2[u][v]+ TaxaAprendizagem(QVisitas[estado][acao])*(Recompensa(distancia, ambiente["Estados"][acao]["Demanda"], ambiente["Capacidade"]) + 0 - Q2[estado][acao])*QElegibilidade[u][v]
                        QElegibilidade[u][v] = lambd*taxaDesconto*QElegibilidade[u][v]
            
            if (rotas[veiculo]["Demanda"] > ambiente["Capacidade"]):
                distanciaTotal = float('inf') # inválido
            
            distanciaTotal = distanciaTotal + rotas[veiculo]["Custo"]
        
        resultados.append(distanciaTotal)
        if (i == 0):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
        if (distanciaTotal < menorDistancia):
            menorDistancia = distanciaTotal
            menorRotas = rotas
        
    return menorDistancia, menorRotas, resultados

def LerArquivo (cpvlib):
    """
    Carrega informações de instâncias da biblioteca CVRPLIB
    
    Entrada:
        cpvlib: Dados do arquivo que contém as informações sobre a instância do problema
        
    Retorno:
        Informações sobre o ambiente
    """
    fh = open(cpvlib, 'r')
    linhas = fh.readlines() 
    
    parametro = 0 # 0 - informações, 1 - coordenadas, 2- demanda
    
    # Nós = deposito(0) + consumidores (1..N)
    estados = []
    
    for linha in linhas:
        valores = linha.split()
        if linha.find("NAME") != -1:
            nome = valores[2]
            
        if linha.find("trucks:") != -1:
            veiculos = int(valores[valores.index('trucks:') + 1].replace(',',''))
        
        if linha.find("CAPACITY") != -1:
            capacidade = int(valores[2])
    
        if linha.find("NODE_COORD_SECTION") != -1:
            parametro = 1
            
        if linha.find("DEMAND_SECTION") != -1:
            parametro = 2
        
        if linha.find("DEPOT_SECTION") != -1:
            parametro = 0
        
        if parametro == 1 and valores[0].isdigit():
            estados.append({"CoordX":int(valores[1]), "CoordY":int(valores[2])})
    
        if parametro == 2 and valores[0].isdigit():
            estados[int(valores[0])-1]['Demanda'] = int(valores[1])

    ambiente = {"Estados":estados, "Nome": nome, "Capacidade": capacidade, "Veiculos": veiculos}
        
    return ambiente

def Biblioteca ():
    """
    Carrega todas as instâncias escolhidas da biblioteca CVRPLIB
    
    Entrada:
        
    Retorno:
        Lista com os dados do arquivo das instâncias
    """
    cpvlib = []
    cpvlib.append("Benchmark/A-n32-k5.vrp")     # 0 - Ideal é 784
    cpvlib.append("Benchmark/A-n63-k10.vrp")    # 1 - Ideal é 1314 
    cpvlib.append("Benchmark/A-n64-k9.vrp")     # 2 - Ideal é 1401
    cpvlib.append("Benchmark/B-n31-k5.vrp")     # 3 - Ideal é 672
    cpvlib.append("Benchmark/E-n22-k4.vrp")     # 4 - Ideal é 375
    cpvlib.append("Benchmark/E-n51-k5.vrp")     # 5 - Ideal é 521
    cpvlib.append("Benchmark/P-n16-k8.vrp")     # 6 - Ideal é 450
    cpvlib.append("Benchmark/X-n106-k14.vrp")   # 7 - Ideal é 26362
    
    return cpvlib

def GraficoCustoEpisodio(resultados, algoritmo):
    plt.plot(resultados)
    plt.xlabel(algoritmo)
    plt.title("Custos por episódios")
    plt.show()
    
def GraficoRotas(menorRotas, menorDistancia, algoritmo, ambiente):
    cor = ['blue', 'red', 'orange', 'green', 'purple', 'cyan', 'pink', 'lightgreen', 'crimson','navy']
    
    for i in range(len(menorRotas)):
        x = []
        y = []
        for j in range(len(menorRotas[i]["Consumidores"])):
            valor = menorRotas[i]["Consumidores"][j]
            x.append(ambiente["Estados"][valor]["CoordX"])
            y.append(ambiente["Estados"][valor]["CoordY"])
        plt.plot(x, y, cor[i], label="R"+str(i))
    
    plt.title("Rotas geradas com distância total de " + str(round(menorDistancia)))
    plt.xlabel(algoritmo)
    plt.legend(loc='upper left')
    plt.show()
    
def ExibeResultados(opcao, ambiente):
    if opcao == 0:
        algoritmo = "Algoritmo Q-Learning com traço de elegibilidade (veículos fixos) \n"
    elif opcao == 1:
        algoritmo = "Algoritmo Double Q-Learning com traço de elegibilidade (veículos fixos) \n"
    elif opcao == 2:
        algoritmo = "Algoritmo Q-Learning com traço de elegibilidade (veículos dinâmicos) \n"
    elif opcao == 3:
        algoritmo = "Algoritmo Double Q-Learning com traço de elegibilidade (veículos dinâmicos) \n"
    else:
       print("Escolha inválida de algoritmo!")
       return False
   
    print(algoritmo)
    
    for Desconto in [0.1, 0.01]: # Taxa de desconto
        print("Taxa de Desconto: ", Desconto)
        for Lambda in [1, 0.99, 0.975, 0.95, 0.9, 0.8, 0.4, 0]: # Valor de lambda
            valores = []
            custoComputacional = []
            for j in range(30):
                time_start = time.clock()
                
                if opcao == 0:
                    menorDistancia, menorRotas, resultados = Q_Learning_VeiculosFixos(ambiente, Lambda, Desconto)
                elif opcao == 1:
                    menorDistancia, menorRotas, resultados = DoubleQ_Learning_VeiculosFixos(ambiente, Lambda, Desconto)
                elif opcao == 2:
                    menorDistancia, menorRotas, resultados = Q_Learning_VeiculosDinamicos(ambiente, Lambda, Desconto)
                else:
                    menorDistancia, menorRotas, resultados = DoubleQ_Learning_VeiculosDinamicos(ambiente, Lambda, Desconto)
                
                time_elapsed = (time.clock() - time_start)
                
                valores.append(menorDistancia)
                custoComputacional.append(time_elapsed)
                #Rotas escolhidas
                #for i in range(menorRotas):
                #    print("R"+ str(i) + "  ", menorRotas[i])
                #GraficoCustoEpisodio(resultados, algoritmo)
                #GraficoRotas(menorRotas, menorDistancia, algoritmo, ambiente)
            
            media = statistics.mean(valores)
            desvio = statistics.pstdev(valores)
            if (media == float('inf') or desvio == float('inf')):
                print("Inválido")
            else:
                print("Distância: ", round(media), " Desvio: ", round(desvio), " Custo Computacional: ", round(statistics.mean(custoComputacional), 2))
    
    return True
    
"""
Começo do código
"""
# Carrega arquivos
cpvlib = Biblioteca()

# Escolher algoritmo
opcao = 0 # Algoritmo Q-Learning com traço de elegibilidade (veículos fixos)
'''
opcao = 1 #Algoritmo Double Q-Learning com traço de elegibilidade (veículos fixos)
opcao = 2 #Algoritmo Q-Learning com traço de elegibilidade (veículos dinâmicos) 
opcao = 3 #Algoritmo Double Q-Learning com traço de elegibilidade (veículos dinâmicos) 
'''

# Carrega informações do ambiente de um arquivo
for k in range(0,8):
    ambiente = LerArquivo (cpvlib[k])
    print("Ambiente: ", ambiente["Nome"])
    
    # Execução dos algoritmos
    ExibeResultados(opcao, ambiente)

            