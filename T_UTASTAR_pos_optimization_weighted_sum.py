import sys

from docplex.mp.model import Model
import numpy as np
from funcoes import *
import pandas as pd


# Pós otimização

class T_UTASTAR_pos_optmization_weighted_sum(Model):
    def __init__(self, tensor_decisao, m_w, num_alternatives, num_criterios, alfa, sig, preferivel, delta, wij_das_utilizadades_de_cada_alternativa, ordem_pref_critérios):
        Model.__init__(self, 'T_UTASTAR_pos')
        self.tensor_decisao = tensor_decisao
        self.m_w = m_w
        self.num_alternatives = num_alternatives
        self.num_criterios = num_criterios
        self.alfa = alfa
        self.sig = sig
        self.preferivel = preferivel
        self.delta = delta
        self.wij_das_utilizadades_de_cada_alternativa = wij_das_utilizadades_de_cada_alternativa
        self.ordem_pref_critérios = ordem_pref_critérios


    def generate_weights_multiobjective(self):


        # O código abaixo é para gerar um peso para cada critério e depois "dividir" o peso de cada critério para cada w_kjl.
        # Então, no final, estou dando um peso para cada critério independente da característica
        pesos_por_carac = []
        peso_por_criterio = np.random.dirichlet(np.ones(self.num_criterios), size=1)

        if self.ordem_pref_critérios[0]:
            sorted_indices = np.argsort(peso_por_criterio[0])
            vetor_ordem_crescente = peso_por_criterio[0][sorted_indices]
            # o order diz a ordem de preferencia do critério sendo 0 o menos preferível (que terá menor peso) e (n-1) (número de critérios) o mais preferivel (que terá maior peso)
            order = self.ordem_pref_critérios[1]
            new_array = vetor_ordem_crescente[order]

            peso_por_criterio = [new_array]



        pesos_por_carac = []
        for carac in range(len(self.tensor_decisao)):
            n = 0

            aux2 = []
            for ite, l in enumerate(self.alfa[carac]):
                n += (l - 1)
                aux = []
                aux.append(peso_por_criterio[0][ite]/(l-1))

                aux2.append(aux*(l-1))


            pesos_por_carac.append(list(np.array(aux2).flat))

        return pesos_por_carac


    def setup_variables(self):
        # Variáveis
        self.w_kje = []
        for k in range(len(self.tensor_decisao)):
            self.w_kje.append([self.continuous_var(name='w_{}{}{}'.format((k + 1), (i + 1), (j + 1)))
                               for i in range(0, self.num_criterios)
                               for j in range(0, (self.alfa[k][i] - 1))])

        self.sigma_ik = [self.continuous_var(name='s({}){}'.format(mais_menos, (k + 1)))
                    for k in range(0, self.num_alternatives) for mais_menos in self.sig]

#TODO:tenho que adicionar a restrição que a função objetivo da solução do modelo vai na restrição
#TODO: também tenho que adicionar as variaveis erro conforme valores do modelo
    def setup_constraints(self):
        # Restrições

        for alt in range(0, len(self.m_w)):
            if self.preferivel[alt]:
                restricoes = self.add_constraint(
                    self.sum(np.dot(self.m_w[alt], np.concatenate(self.w_kje))) >= self.delta)
            else:
                restricoes = self.add_constraint(
                    self.sum(np.dot(self.m_w[alt], np.concatenate(self.w_kje))) == 0)

        for k in range(len(self.tensor_decisao)):
            self.add_constraint(self.sum(self.w_kje[k]) == 1)



    def setup_objective(self, pesos):
        # Função objetivo
        #print([(k, (j+1)*(e+1), self.w_kje[k][j+e]) for k in range(len(self.tensor_decisao)) for j in range(0, self.num_criterios-1) for e in range(self.alfa[k][j] - 2)])


        #self.maximize(self.sum(pesos[j] * self.w_kje[k][j] for k in range(len(self.tensor_decisao)) for j in range(0, self.num_criterios) for e in range(self.alfa[k][j] - 1)))
        self.maximize(self.sum(pesos[k][a] * self.w_kje[k][a] for k in range(len(self.tensor_decisao)) for a in range(len(pesos[k]))))





#TODO: tenho que ver como está sendo a comparação da solução final, porque aqui estou trabalhando com tensor e tenho o codigo como matriz, ver o for do comparacao
    def run(self, num_iteracoes_multiobjetivo):
        #vect_solucoes pega as soluções separadas por características
        vect_solucoes = []
        # vect_solucoes_todas_juntas pega as soluções juntas em um único vetor
        vect_solucoes_todas_juntas = []
        #vect_resultado_correto junta verificar se os resultados realmente mantem a pontuação na ordem correta
        vect_resultado_correto = []
        for iteracao in range(num_iteracoes_multiobjetivo):
            pesos = self.generate_weights_multiobjective()
            self.clear()
            self.setup_variables()
            self.setup_constraints()
            self.setup_objective(pesos)
            status = self.solve()
            if not status:
                print
                "O problema nao encontrou solucao viavel"
                return False

            #self.print_solution()

            solucao = [[self.w_kje[k][i].solution_value for i in range(len(self.w_kje[k]))] for k in
                       range(len(self.w_kje))]


            #Abaixo verifico se a solução mantem a ordem de preferência das alternativas
            w_ij_todo_junto_vetor = np.array(solucao).flatten()
            # Calculo a pontuação de cada alternativa
            vetor_pontuacao_alternativas_calculado_pelos_w_ij = calucular_pontuacoes_alternativas_apos_obter_pontuacao(self.wij_das_utilizadades_de_cada_alternativa, w_ij_todo_junto_vetor)
            # Verifico se o vetor está em ordem descrescente, respeitando a pref do decisor
            resultado_correto = verificar_se_pontuacao_alternativas_estao_descrescentes(vetor_pontuacao_alternativas_calculado_pelos_w_ij)
            vect_resultado_correto.append(resultado_correto)



            #gero um ordered dict que associa o valor do w_kjl para cada solução
            #dict_ordenado_w_kjl_e_valor = gerar_dict_com_valores_dos_w(w_ij_todo_junto_vetor, len(self.tensor_decisao), self.num_criterios, self.alfa)
            #print(dict_ordenado_w_kjl_e_valor)



            # abaixo pego as soluções com as características separadas
            vect_solucoes.append(solucao)
            #abaixo pego as soluções com as características todas juntas, em um único vetor
            vect_solucoes_todas_juntas.append(w_ij_todo_junto_vetor)
            #print(vect_solucoes_todas_juntas)


            # O for abaixo é pra "quebrar" o vetor solução da mesma forma de vetor que é gerado os w_ij na função generate_symbols_wjk
            w_ij_split = []
            for k in range(len(self.w_kje)):

                n = []
                a = 0
                for i in range(len(self.alfa[k]) - 1):
                    a += self.alfa[k][i] - 1
                    n.append(a)
                w_ij_split.append(np.split(solucao[k], n))


        #tamanho_vetor_solucao_por_k = [len(t) for t in solucao]



        if np.array(vect_resultado_correto).all():
            print('as pontuações obtidas pelo multiobjetivo estão em ordem decrescente:')
            print("SIM \n")
        else:
            print('as pontuações obtidas pelo multiobjetivo estão em ordem decrescente:')
            print("NÃO \n")



        # Encontrando os vetores únicos
        vetores_unicos, indices, counts = np.unique(vect_solucoes_todas_juntas, axis=0, return_index=True, return_counts=True)


        # Convertendo os vetores únicos e ocorrências em listas
        vetores_unicos_list = vetores_unicos.tolist()
        counts_list = counts.tolist()



        #Transformo o vetores_unicos_list em um dict_ordenado que indica o w_kjl e o valor dele
        vec_com_dict_ordenado_do_vetor_unico = []
        for vetor_solucoes in vetores_unicos:
            dict_ordenado_w_kjl_e_valor = gerar_dict_com_valores_dos_w(vetor_solucoes, len(self.tensor_decisao), self.num_criterios, self.alfa)
            vec_com_dict_ordenado_do_vetor_unico.append(dict_ordenado_w_kjl_e_valor)


        # Criando o dicionário
        result = [{"Vetor": v, "Ocorrências": c} for v, c in zip(vetores_unicos_list, counts_list)]
        result_com_dict = [{"Vetor": v, "Ocorrências": c} for v, c in zip(vec_com_dict_ordenado_do_vetor_unico, counts_list)]




        # calculando a média ponderada
        soma_ocorrencias = sum(d['Ocorrências'] for d in result)
        soma_ponderada = np.zeros(len(result[0]['Vetor']))
        soma_simples = np.zeros(len(result[0]['Vetor']))
        for d in result:
            soma_ponderada += np.array(d['Vetor']) * d['Ocorrências']
            soma_simples += np.array(d['Vetor'])
        media_ponderada = soma_ponderada / soma_ocorrencias


        media_simples = soma_simples/len(result)

        dict_ordenado_w_kjl_e_valor_media_ponderada = gerar_dict_com_valores_dos_w(media_ponderada, len(self.tensor_decisao), self.num_criterios, self.alfa)
        dict_ordenado_w_kjl_e_valor_media_simples = gerar_dict_com_valores_dos_w(media_simples, len(self.tensor_decisao), self.num_criterios, self.alfa)

        # O for abaixo é pra "quebrar" o vetor solução da mesma forma de vetor que é gerado os w_ij na função generate_symbols_wjk
        media_ponderada_para_cada_k = np.split(media_ponderada, len(self.tensor_decisao))

        w_ij_split = []
        for k in range(len(self.w_kje)):
            n = []
            a = 0
            for i in range(len(self.alfa[k]) - 1):
                a += self.alfa[k][i] - 1
                n.append(a)
            w_ij_split.append(np.split(media_ponderada_para_cada_k[k], n))

        #np.set_printoptions(suppress=True)
        #np.set_printoptions(precision=3)
        #CASO EU PRECISE DO DIC SEPARADO POR CARACTERÍSTICA:
        #np.set_printoptions(threshold=sys.maxsize)
        #print(w_ij_split)
        #print('vect_solucoes')
        #print(np.array(vect_solucoes))
        #algoResult = feature_vectors_occurences_across_solutions(np.array(vect_solucoes))
        #print('resultado final')
        #print(algoResult)

        return result_com_dict, dict_ordenado_w_kjl_e_valor_media_ponderada, dict_ordenado_w_kjl_e_valor_media_simples, w_ij_split, len(self.w_kje)








""""


print('iteração', i)
            if vect_solucoes:
                print('entrou')
                print(solucao)
                compare = []
                for c, comp in enumerate(solucao):
                    print('comp')
                    print(comp)
                    print(vect_solucoes[0][c])
                    for x in vect_solucoes[0][c]:
                        print('versao numpu')
                        print(x)
                        print(np.array_equal(comp, x))
                        if x == vect_solucoes[0][c]:
                            print('são iguais')
                            compare.append(True)
                        else:
                            print('são diferentes')
                            compare.append(False)

                    #compare.append([np.array_equal(comp, x) for x in vect_solucoes[0][c]])
                    print('compare', compare)

            else:
                compare = False

            if not (np.any(compare)):
                vect_solucoes.append(solucao)
                print('sol')
                print(vect_solucoes[0])







   
    # Variáveis
    pos_w_ij = [model_pos.continuous_var(name='pos_w_{}{}'.format((i + 1), (j + 1)))
                for i in range(0, num_criterios)
                for j in range(0, (alfa[i] - 1))]

    # sigma_ik = [model.continuous_var(name='s({}){}'.format(mais_menos, (k+1)))
    #                for k in range(0, num_alternatives) for mais_menos in sig]

    # Função objetivo
    model_pos.maximize(model.sum(pesos[0] * pos_w_ij[0] + pesos[1] * pos_w_ij[1] + pesos[2] * pos_w_ij[2]))
    # model_pos.maximize(model.sum(1*pos_w_ij[0]+ 0*pos_w_ij[1] + 0*pos_w_ij[2]))
    # model_pos.minimize(model.sum(pos_w_ij[0]))

    # Restrições

    for k in range(0, len(m_w)):
        if preferivel[k]:
            restricoes = model_pos.add_constraint(model_pos.sum(np.dot(m_w[k], pos_w_ij)) >= delta)
            # print(restricoes)
        else:
            restricoes = model_pos.add_constraint(model_pos.sum(np.dot(m_w[k], pos_w_ij)) == 0)
            # print(restricoes)

    model_pos.add_constraint(model_pos.sum(pos_w_ij) == 1)

    model_pos.solve()
    # model_pos.print_solution()
    solucao = [pos_w_ij[i].solution_value for i in range(len(pos_w_ij))]

    # print('solu atual')
    # print(solucao)
    # print('vec all')
    # print(vect_solucoes)
    # print("v ou f")
    compare = [np.array_equal(solucao, x) for x in vect_solucoes]
    # print(compare)

    if not (np.any(compare)):
        vect_solucoes.append(solucao)

    # print("")

vect_solucoes = np.array(vect_solucoes)
print(vect_solucoes)

"""