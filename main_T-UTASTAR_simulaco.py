import numpy as np
import pandas as pd
import sys

from T_UTASTAR_mathematical_model import T_UTASTAR_modelo
from funcoes import *
from T_UTASTAR_obter_ws import Valores_do_w
from T_UTASTAR_pos_optimization_weighted_sum import T_UTASTAR_pos_optmization_weighted_sum
#from TESTE_T_UTASTAR_pos_optimization_weighted_sum import T_UTASTAR_pos_optmization_weighted_sum

#np.set_printoptions(suppress=True)
#np.set_printoptions(precision=16)


# o ordem_pref_critérios tem dois valores, o primeiro diz se o vetor de pesos deve considerar (True) a ordem de preferência dos critérios ou deve ser aleatório
# o segundo é um vetor que diz a ordem de preferencia dos critérios sendo 0 o MENOS preferível (que terá menor peso) e (n-1) (número de critérios) o mais preferivel (que terá maior peso)
ordem_pref_critérios = [False, [2, 1, 0, 3]]


num_iteracoes_multiobjetivo = 1000
m = 3
n = 4
T = 10

min_val = 10
max_val = 20
tensor_temp = np.random.uniform(min_val, max_val, (m, n, T))

#print(tensor_temp)
#sys.exit(0)



#Parâmetros:
delta = 0.05
#preferivel = [True] * 10
preferivel = [True] * len(tensor_temp)
sig = ["+", "-"]



#Chamo a função que gera o tensor de atributos a partir do tensor de tempo
tensor_atributos = trans_espaco_atributos(tensor_temp.T)
tensor_decisao = tensor_atributos.transpose(0, 2, 1)



#TODO:DELETAR ABAIXO

"""
#TESTANDO, DELETAR ABAIXO:
tensor_decisao = np.array([
    [[3, 10, 1],
     [4, 20, 2],
     [2, 11, 3]],
    [[4, 10, 2],
     [5, 15, 1],
     [3, 11, 3]] # TR
])
"""





# a função abaixo é para calcular os alfas (quantidade de diferentes valores de cada critério)
alfa = generate_alfa(tensor_decisao)


# Obtendo valores de num alternativa, critério e criterios de max
num_alternatives = len(tensor_decisao[0])
num_criterios = (tensor_decisao[0].shape)[1]
c_max = np.array([True] * num_criterios)

#arredondar pra 3 casas decimais:
tensor_decisao = np.round(tensor_decisao, 3)
#print(tensor_decisao)

# Abaixo chamo a classe T_UTASTAR_obter_ws, ela serve para representar a utilidade das alternativas em termos de w
valores_w = Valores_do_w(tensor_decisao, c_max)
m_w, k_vetores_dict_values_ui, wij_das_utilizadades_de_cada_alternativa = valores_w.run()


#Chamo a classe modelo T_UTASTAR pra obter o resultado do modelo matemático. ela retorna um vetor com todas as soluções dos w_ij, uma solução com os w_ij separados por característica e por critério e o tamanho do w_ij
T_UTASTAR = T_UTASTAR_modelo(tensor_decisao, m_w, num_alternatives, num_criterios, alfa, sig, preferivel, delta)
w_ij_todo_junto_vetor, w_ij_split, len_wij = T_UTASTAR.run()



#Calculo a pontuação de cada alternativa
vetor_pontuacao_alternativas_calculado_pelos_w_ij = calucular_pontuacoes_alternativas_apos_obter_pontuacao(wij_das_utilizadades_de_cada_alternativa, w_ij_todo_junto_vetor)

#Verifico se o vetor está em ordem descrescente, respeitando a pref do decisor
resultado_correto = verificar_se_pontuacao_alternativas_estao_descrescentes(vetor_pontuacao_alternativas_calculado_pelos_w_ij)


print('as pontuações obtidas pelo modelo matemático estão em ordem decrescente:')
print(resultado_correto) # imprime "True"


#Chamo a função para associar a cada utilidade, um valor
utilidade_associada_a_valores = associar_a_cada_utiliadde_um_valor(len_wij, w_ij_split, k_vetores_dict_values_ui)





#PARA GERAR OS GRÁFICOS
title = ["Média", "Coeficiente angular"]

y_label = [[r'$u_{11}(c_{11})$', r'$u_{12}(c_{12})$', r'$u_{13}(c_{13})$'], [r'$u_{21}(c_{21})$', r'$u_{22}(c_{22})$', r'$u_{23}(c_{23})$']]
#x_label = [r'LE at birth ($c_1$)', r'Education ($c_2$)', r'GNI per capita ($c_3$)']
x_label = [r'$c_1$', r'$c_2$', r'$c_3$']
#gerar_grafico(utilidade_associada_a_valores, tensor_decisao, title, x_label, y_label)




#MULTIOBJETIVO

print("\n Começa o multiobjetivo \n")
T_UTASTAR_weighted_sum = T_UTASTAR_pos_optmization_weighted_sum(tensor_decisao, m_w, num_alternatives, num_criterios, alfa, sig, preferivel, delta, wij_das_utilizadades_de_cada_alternativa, ordem_pref_critérios)
result_multiobjetivo_com_dict, dict_ordenado_w_kjl_e_valor_media_ponderada, dict_ordenado_w_kjl_e_valor_media_simples, w_ij_split_media_ponderada, len_wij_mp = T_UTASTAR_weighted_sum.run(num_iteracoes_multiobjetivo)

print(dict_ordenado_w_kjl_e_valor_media_ponderada)
#Chamo a função para associar a cada utilidade, um valor da média ponderada
utilidade_associada_a_valores_media_ponderada = associar_a_cada_utiliadde_um_valor(len_wij_mp, w_ij_split_media_ponderada, k_vetores_dict_values_ui)



#gerar_grafico(utilidade_associada_a_valores_media_ponderada, tensor_decisao, title, x_label, y_label)


#print("Solucões multiobjetivo: ", result_multiobjetivo_com_dict)

print("Média ponderada: ", dict_ordenado_w_kjl_e_valor_media_ponderada)
print("Média simples: ", dict_ordenado_w_kjl_e_valor_media_simples)

print("\n")



# O código abaixo é para preparar a Tabela com os valores não negativos dos w_{kj\ell}
variaveis_nao_negativas = []
for dicionario in result_multiobjetivo_com_dict:
    variaveis_nao_negativas.append([(chave, valor, dicionario['Ocorrências']) for chave, valor in dicionario['Vetor'].items() if valor != 0])

#abaixo faço o mesmo para a média ponderada MP, só que o código é diferente porque o dict da média é diferente
variaveis_nao_zero_MP = []
for key, value in dict_ordenado_w_kjl_e_valor_media_ponderada.items():
    if value != 0:
        variaveis_nao_zero_MP.append((key, value))


#Obter os w_kj\ell que foram não negativos em pelo menos uma ocorrência (não é em termos de valor, apenas a representação dos w)
sorted_w_set = obter_w_kjl_nao_negativos(variaveis_nao_negativas)

print('variaveis_nao_negativas')
print(variaveis_nao_negativas)
#separar dos outros dados os vetores que foram diferentes dentro da solução, independente da ocorrencia
vetores_diferentes_w = obter_vetores_da_solucao_diferentes_entre_si(variaveis_nao_negativas, sorted_w_set)

#faço o mesmo para média ponderada (MP):
vetores_diferentes_w_MP = obter_vetores_da_solucao_diferentes_entre_si([variaveis_nao_zero_MP], sorted_w_set)


print('vetor diferernte')
print(vetores_diferentes_w)
print('vetor diferernte')
print(vetores_diferentes_w_MP)

#w_ij_todo_junto_MP = np.array(w_ij_split_media_ponderada).flatten()
#print(w_ij_todo_junto_MP)

#preparando para printar a tabela com os diferentes soluções e suas ocorrências
dict_para_latex = dict(zip(sorted_w_set, np.transpose(vetores_diferentes_w)))
df_para_tabela_latex = pd.DataFrame.from_dict(dict_para_latex, orient='index')




ocorrencia = [a[0][2] for a in variaveis_nao_negativas]

df_para_tabela_latex.columns = ocorrencia





#colocando em vermelho os dados não negativos
df_para_tabela_latex = df_para_tabela_latex.round(3)
for col in df_para_tabela_latex.columns:
    for row in df_para_tabela_latex.index:
        if df_para_tabela_latex.at[row,col] not in [0.000, 0.00]:
            df_para_tabela_latex.at[row,col] = "\textcolor{red}{"+str(df_para_tabela_latex.at[row,col])+"}"

values = [format(val, '.2f') for val in vetores_diferentes_w_MP[0]]

df_para_tabela_latex['MP'] = np.transpose(values)

print(df_para_tabela_latex.to_latex(escape=False))









#preparando para printar a tabela DEITADA com os diferentes soluções e suas ocorrências
ocorrencia = [a[0][2] for a in variaveis_nao_negativas]

df_para_tabela_latex = pd.DataFrame(vetores_diferentes_w, columns=sorted_w_set)


df_para_tabela_latex.columns = zip(sorted_w_set)
df_indice = pd.DataFrame(ocorrencia, columns=['Ocorrência'])
df_para_tabela_latex = pd.concat([df_indice, df_para_tabela_latex], axis=1)






values = [format(val, '.2f') for val in vetores_diferentes_w_MP[0]]

#df_para_tabela_latex['MP'] = np.transpose(values)

print(df_para_tabela_latex.to_latex(escape=False))




