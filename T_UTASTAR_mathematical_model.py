from docplex.mp.model import Model
import numpy as np





class T_UTASTAR_modelo(Model):
    def __init__(self, tensor_decisao, m_w, num_alternatives, num_criterios, alfa, sig, preferivel, delta):
        Model.__init__(self, 'T_UTASTAR')
        self.tensor_decisao = tensor_decisao
        self.m_w = m_w
        self.num_alternatives = num_alternatives
        self.num_criterios = num_criterios
        self.alfa = alfa
        self.sig = sig
        self.preferivel = preferivel
        self.delta = delta




    # Agora preciso acrescentar o erro que aparece no passo 4 dos slides
    # get the matrix sigma (errors): m_sigma
    def matrix_variables_sigma(self):
        # get the matrix sigma (errors): m_sigma
        self.m_sigma = np.zeros((self.num_alternatives - 1, self.num_alternatives * 2))

        j = 0
        for i, vec in enumerate(self.m_sigma):
            vec[j:j + 4] = -1, 1, 1, -1
            j += 2

        return self.m_sigma


    # SETUPS

    def setup_variables(self):
        # Variáveis
        self.w_ij = []
        for k in range(len(self.tensor_decisao)):
            self.w_ij.append([self.continuous_var(name='w_{}{}{}'.format((k + 1), (i + 1), (j + 1)))
                         for i in range(0, self.num_criterios)
                         for j in range(0, (self.alfa[k][i] - 1))])

        self.sigma_ik = [self.continuous_var(name='s({}){}'.format(mais_menos, (k + 1)))
                    for k in range(0, self.num_alternatives) for mais_menos in self.sig]



    def setup_constraints(self):
        # Restrições

        for alt in range(0, len(self.m_w)):
            if self.preferivel[alt]:
                restricoes = self.add_constraint(
                    self.sum(np.dot(self.m_w[alt], np.concatenate(self.w_ij)) + np.dot(self.m_sigma[alt], self.sigma_ik)) >= self.delta)
            else:
                restricoes = self.add_constraint(
                    self.sum(np.dot(self.m_w[alt], np.concatenate(self.w_ij)) + np.dot(self.m_sigma[alt], self.sigma_ik)) == 0)

        for k in range(len(self.tensor_decisao)):
            self.add_constraint(self.sum(self.w_ij[k]) == 1)

    def setup_objective(self):
        # Função objetivo
        self.minimize(self.sum(self.sigma_ik))



    def run(self):
        self.matrix_variables_sigma()
        self.clear()
        self.setup_variables()
        self.setup_constraints()
        self.setup_objective()

        status = self.solve()
        if not status:
            print
            "O problema nao encontrou solucao viavel"
            return False

        self.print_solution()

        solucao = [[self.w_ij[k][i].solution_value for i in range(len(self.w_ij[k]))] for k in range(len(self.w_ij))]


        # O for abaixo é pra "quebrar" o vetor solução da mesma forma de vetor que é gerado os w_ij na função generate_symbols_wjk
        w_ij_split = []

        for k in range(len(self.w_ij)):
            n = []
            a = 0
            for i in range(len(self.alfa[k]) - 1):
                a += self.alfa[k][i] - 1
                n.append(a)
            w_ij_split.append(np.split(solucao[k], n))

        w_ij_todo_junto_vetor = np.array(solucao).flatten()

        return w_ij_todo_junto_vetor, w_ij_split, len(self.w_ij)