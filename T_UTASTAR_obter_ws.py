import numpy as np, numpy.random
from sympy import *
from scipy import stats
from collections import OrderedDict




class Valores_do_w():

    def __init__(self, tensor, c_max):
        self.tensor = tensor
        self.c_max = c_max



    # A função abaixo é apenas para gerar os símbolos u_jk(s_ijk)
    def generate_symbols_uij(self):
        simbolos = []
        for t in self.tensor:
            s = []
            for i, criterio in enumerate(np.transpose(t).astype(float)):
                auxi_s = []
                if self.c_max[i]:
                    criterio = np.sort(criterio)
                else:
                    criterio = -np.sort(-criterio)

                for j in criterio:
                    if symbols('ui(%.5f)' % j) not in auxi_s:
                        t = symbols('ui(%.5f)' % j)
                        auxi_s.append(t)

                # auxi_s[0] = 0
                s.append(auxi_s)
            simbolos.append(s)

        return simbolos


    # A função abaixo é apenas para gerar os símbolos w_ij
    def generate_symbols_wjk(self, symbols_ug_ia):
        s2 = []
        for k in symbols_ug_ia:
            auxi2_s2 = []
            for i in k:
                auxi_s2 = []
                for j in range(len(i) - 1):
                    t = symbols('w_j_%d' % ((j + 1)))
                    auxi_s2.append(t)
                auxi2_s2.append(auxi_s2)
            s2.append(auxi2_s2)
        return s2



    # A função abaixo é da equação 18, sendo que ela já coloca os u_i em função dos w. ver exemplo no slide chamado PASSO 1: CALCULA FUNÇÃO DE UTILIDADE
    def equation_7_18(self, symbols_ug_ia, symbols_wij):
        k_vetores_dict_values_ui = []
        for k, attribute in enumerate(symbols_ug_ia):
            vetor_dict_values_ui = []
            for c, criterio in enumerate(attribute):
                # dict_valor_ui = {}
                dict_valor_ui = OrderedDict()
                for j, ui in enumerate(criterio):
                    if j == 0:
                        dict_valor_ui[ui] = np.zeros(len(symbols_wij[k][c]))
                    else:
                        v = np.zeros(len(symbols_wij[k][c]))
                        for a in range(j):
                            v[a] = 1
                        dict_valor_ui[ui] = v
                vetor_dict_values_ui.append(dict_valor_ui)

            k_vetores_dict_values_ui.append(vetor_dict_values_ui)

        return k_vetores_dict_values_ui



    # A função abaixo determina as utilidades de cada alternativa, é o u_k[c(a_i)], sendo que já em função dos w_ijk
    def generate_utility_func_of_alt(self, symbols_ug_ia, k_vetores_dict_values_ui):
        tensor_utilidade_alt = []
        for m, matriz in enumerate(self.tensor):
            mat_utilidades_altern = []
            for q, colun in enumerate(matriz.T):
                aux_mat_utilidades = []
                for element in colun:
                    t = symbols('ui(%.5f)' % element)
                    aux_mat_utilidades.append(k_vetores_dict_values_ui[m][q][t])

                mat_utilidades_altern.append(aux_mat_utilidades)
            tensor_utilidade_alt.append(mat_utilidades_altern)
        #print('tensor_utilidade_alt')
        #print(tensor_utilidade_alt)
        return np.array(tensor_utilidade_alt)


    # A função abaixo é o PASSO 3 (4?? sera que errei?)  dos slides. serve para subtrair as utilidades par a par das alternativas
    def equation_7_11(self, tensor_utilidade_alt):
        m_w_por_periodo = []
        conc_arr = []
        for d in tensor_utilidade_alt:
            conc_arr.append(np.concatenate(d, axis=1))

        conc_arra_terceira_dimensao = np.concatenate(conc_arr, axis=1)
        #print('conc_arra_terceira_dimensao')
        #print(conc_arra_terceira_dimensao)
        aux_m_w = []
        #print(len(conc_arra_terceira_dimensao))
        for j in range(len(conc_arra_terceira_dimensao) - 1):
            if j < len(conc_arra_terceira_dimensao):
                aux_m_w.append(conc_arra_terceira_dimensao[j] - conc_arra_terceira_dimensao[j + 1])

        #A função também retorna o conc_arra_terceira_dimensao porque nele temos todos os  w_ij de cada alternativas juntos, isso é útil para no final multiplicar pelo vetor da solução dos valores dos w_ij
        #no caso, esse conc_arra_terceira_dimensao é como se representasse os zeros ou 1 do slide passo 3
        return aux_m_w, conc_arra_terceira_dimensao





    def run(self):
        # Chamando as funções:
        symbols_ug_ia = self.generate_symbols_uij()

        symbols_wij = self.generate_symbols_wjk(symbols_ug_ia)

        k_vetores_dict_values_ui = self.equation_7_18(symbols_ug_ia, symbols_wij)

        tensor_utilidade_alt = self.generate_utility_func_of_alt(symbols_ug_ia, k_vetores_dict_values_ui)

        m_w, conc_arra_terceira_dimensao = self.equation_7_11(tensor_utilidade_alt)


        return m_w, k_vetores_dict_values_ui, conc_arra_terceira_dimensao


