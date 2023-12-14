import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy



# Atenção: passar tensor transposto
def trans_espaco_atributos(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    var = np.var(X, axis=0)
    cv = stats.variation(X, axis=0)

    # O coeficiente angular (ca) depende do critério pois se o critério é de custo, é de mínimo, se é de benefício, é de máximo
    ca = []
    for i in X.T:
        aux = []
        for j in i:
            slope, intercept, r_value, p_value, std_err = stats.linregress((np.arange(len(j)), j))
            aux.append(slope)
        ca.append(aux)
    ca = np.array(ca).T

    tensor_atributos = np.array([mean, cv, ca])
    tensor_atributos = np.array([mean, ca])

    return np.array(tensor_atributos)


#Essa função só serve para gerar o alfa
def generate_alfa(tensor_decisao):
    alfa = []
    for k in tensor_decisao:
        auxi_alfa = []
        for i in k.T:
            (unique, counts) = np.unique(i, return_counts=True)
            auxi_alfa.append(len(counts))
        alfa.append(auxi_alfa)

    return alfa


def feature_vectors_occurences_across_solutions(solutions=[]):
    if solutions.size == 0:
        return None

    result = []
    feature_vectors_in_solution = solutions[0]
    for feature_idx, _ in enumerate(feature_vectors_in_solution):
        uniques, counts = np.unique(solutions[:, feature_idx, :], return_counts=True, axis=0)
        tmp = [{'nth_vect_in_solution': u, 'reps': c} for (u, c) in zip(uniques, counts)]
        result.append(tmp)

    return result



#As utilidades totais de cada alternativa é obtida multiplicando o resultado dos w_ij (de todas as caracteristicas e todos os critérios) pelos valores em wij_das_utilizadades_de_cada_alternativa (que representa quais w_ij são multiplicados por 1 e quais por zero)
def calucular_pontuacoes_alternativas_apos_obter_pontuacao(wij_das_utilizadades_de_cada_alternativa, w_ij_todo_junto_vetor):
    vetor_pontuacao_alternativas_calculado_pelos_w_ij = []
    for utilidade_alternativa in wij_das_utilizadades_de_cada_alternativa:
        u_c_a_i = np.dot(w_ij_todo_junto_vetor, utilidade_alternativa)
        vetor_pontuacao_alternativas_calculado_pelos_w_ij.append(u_c_a_i)
    return vetor_pontuacao_alternativas_calculado_pelos_w_ij


#O cálculo abaixo é necessário para verificar se as pontuações finais obtidas pelas alternativas estão de fato ordenadas descrescente, que é como tem que ser, já que eu começo o tensor com a ordem das alternativas de melhor para pior
def verificar_se_pontuacao_alternativas_estao_descrescentes(vetor_pontuacao_alternativas_calculado_pelos_w_ij):
    verificar_se_pontuacao_esta_ordenada_descrescente = np.array(vetor_pontuacao_alternativas_calculado_pelos_w_ij)
    result = np.all(np.diff(verificar_se_pontuacao_esta_ordenada_descrescente) < 0)
    return result



def associar_a_cada_utiliadde_um_valor(len_wij, w_ij_split, k_vetores_dict_values_ui):
    # Abaixo organizo o dict para que cada utilidade vá associada a um valor
    k_vetores_dict_values_ui_atualizado = deepcopy(k_vetores_dict_values_ui)
    for k in range(len_wij):
        for ite, w in enumerate(w_ij_split[k]):
            k_vetores_dict_values_ui_atualizado[k][ite].update( (x, np.dot(y, w)) for x, y in k_vetores_dict_values_ui_atualizado[k][ite].items())

    return k_vetores_dict_values_ui_atualizado




def gerar_grafico(utilidade_associada_a_valores, tensor_decisao, title, x_label, y_label):

    for ite, i in enumerate(utilidade_associada_a_valores):
        mat = tensor_decisao[ite].T
        for ite2, k in enumerate(i):

            y = list(k.values())

            x = np.arange(len(y))

            # x_ticks_labels = [a for a in k.keys()]

            x_ticks_labels = np.round(mat[ite2], 1)
            x_ticks_labels = np.sort(x_ticks_labels)


            max = np.amax(y)
            if max != 0:
                y = y / max

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.scatter(x, y)
            plt.title(title[ite], fontsize=30)

            # Set number of ticks for x-axis
            ax.set_xticks(x)
            # Set ticks labels for x-axis
            ax.set_xticklabels(x_ticks_labels, rotation=20, fontsize=15)
            # ax.set_xticklabels(x_ticks_labels, fontsize=20)
            # ax.yaxis.label.set_size(40)
            plt.yticks(fontsize=15)
            plt.ylim(top=1.1)  # adjust the top leaving bottom unchanged
            plt.ylim(bottom=-0.1)  # adjust the bottom leaving top unchanged
            plt.ylabel(y_label[ite][ite2], fontsize=35)
            plt.xlabel(x_label[ite2], fontsize=25)
            plt.savefig("fig%d%d" % (ite, ite2))
            plt.show()


def gerar_dict_com_valores_dos_w(w_ij_todo_junto_vetor, num_caracteristica, num_criterio, alfa):
    dicionario = OrderedDict()
    prefixo = "$w_"


    chave = []
    for k in range(0, num_caracteristica):
        for j in range(0, num_criterio):
            for l in range(0, (alfa[k][j] - 1)):
                valor = int(str(k+1) + str(j+1) + str(l+1))
                chave.append(prefixo + "{%d}$"%(valor))


    for i, w in enumerate(w_ij_todo_junto_vetor):
        dicionario[chave[i]] = w

    return dicionario




def obter_w_kjl_nao_negativos(variaveis_nao_negativas):
    w_set = set()
    for sublist in variaveis_nao_negativas:
        for item in sublist:
            w_string = item[0]
            if w_string not in w_set:
                w_set.add(w_string)
    sorted_w_set = sorted(w_set)

    return sorted_w_set




def obter_vetores_da_solucao_diferentes_entre_si(variaveis_nao_negativas, sorted_w_set):
    vetores_diferentes_w = []
    for i, matriz in enumerate(variaveis_nao_negativas):

        vetor_w = [i[0] for i in matriz]

        novo_vetor = []
        for ind, ele in enumerate(sorted_w_set):

            if ele in vetor_w:
                local = vetor_w.index(ele)
                novo_vetor.append(matriz[local][1])
            else:
                novo_vetor.append(0)
        # print(novo_vetor)
        vetores_diferentes_w.append(novo_vetor)

    return vetores_diferentes_w