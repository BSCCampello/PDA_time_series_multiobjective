import numpy as np
import pandas as pd

from T_UTASTAR_mathematical_model import T_UTASTAR_modelo
from funcoes import *
from T_UTASTAR_obter_ws import Valores_do_w
from T_UTASTAR_pos_optimization_weighted_sum import T_UTASTAR_pos_optmization_weighted_sum
#from TESTE_T_UTASTAR_pos_optimization_weighted_sum import T_UTASTAR_pos_optmization_weighted_sum

#np.set_printoptions(suppress=True)
#np.set_printoptions(precision=16)

# o ordem_pref_critérios tem dois valores, o primeiro diz se o vetor de pesos deve considerar (True) a ordem de preferência dos critérios ou deve ser aleatório
# o segundo é um vetor que diz a ordem de preferencia dos critérios sendo 0 o MENOS preferível (que terá menor peso) e (n-1) (número de critérios) o mais preferivel (que terá maior peso)

ordem_pref_critérios = [True, [2, 1, 0]]


num_iteracoes_multiobjetivo = 1000

tensor_temp = np.array([
    [[70.7, 71.8, 72.8, 73.6, 74.1, 74.8], [8.1, 8.9, 10.25, 10.15, 11.35, 11.35],
     [9772, 13439, 14500, 17157, 19725, 23712]],  # MY
    [[68, 66, 65.1, 65.8, 68.6, 70.3], [10.95, 10.85, 11.85, 12.6, 13.1, 13.35],
     [19461, 12011, 12933, 17797, 21075, 22094]],  # RU
    [[64.3, 67, 70, 72.5, 74.2, 75.6], [6.7, 7.2, 8.3, 8.95, 10.55, 11.05],
     [10494, 11317, 12807, 14987, 16506, 18976]],  # TR
    [[65.3, 67.6, 70.1, 71.9, 73.3, 74.8], [8, 8.95, 9.95, 10.15, 11.05, 11.6],
     [10065, 10959, 11161, 12032, 14420, 15062]],  # BR
    [[69, 69.9, 71.7, 73.7, 75, 76], [6.8, 7.25, 7.85, 8.75, 9.85, 10.3],
     [1520, 2508, 3632, 5632, 9387, 13347]],  # CN
    [[57.9, 60.4, 62.6, 64.5, 66.5, 68.4], [5.35, 5.9, 6.45, 7.35, 8.25, 8.55],
     [1754, 2046, 2522, 3239, 4499, 5814]],  # IN
    [[63.3, 65, 66.3, 67.2, 68.1, 69.1], [6.75, 7.2, 8.7, 9.3, 9.95, 10.3],
     [4337, 5930, 5308, 6547, 8267, 10130]],  # ID
    [[70.8, 72.8, 74.4, 75.3, 76.1, 77], [8.05, 8.55, 9.15, 9.85, 10.5, 10.8],
     [12074, 12028, 14388, 14693, 15395, 16249]],  # MX
    [[65.3, 66.1, 66.7, 67.2, 67.7, 68.3], [8.7, 8.95, 9.5, 9.75, 9.75, 10.2],
     [3962, 4111, 4994, 6058, 7478, 8232]],  # PH
    [[62.1, 61.4, 55.9, 51.6, 54.5, 57.9], [8.95, 10.65, 11, 11.15, 11.55, 11.75],
     [9987, 9566, 9719, 10935, 11833, 12110]],  # ZA
])







#Parâmetros:
delta = 0.05
#preferivel = [True] * 10
preferivel = [True] * len(tensor_temp)
sig = ["+", "-"]



#Chamo a função que gera o tensor de atributos a partir do tensor de tempo
tensor_atributos = trans_espaco_atributos(tensor_temp.T)
print(tensor_atributos)
tensor_decisao = tensor_atributos.transpose(0, 2, 1)
print(tensor_decisao)


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
gerar_grafico(utilidade_associada_a_valores, tensor_decisao, title, x_label, y_label)




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




