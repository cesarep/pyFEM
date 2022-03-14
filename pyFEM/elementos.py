# -*- coding: utf-8 -*-
"""
Módulo de Elementos

@author: César Eduardo Petersen

@date: Mon Sep  6 17:31:10 2021

Inclui as definições de nós e elementos.

"""

import numpy as np
from .materiais import *
from .integrador import gauss2d2, gauss2d3, gauss2dn

class Node:
    def __init__(self, i, x, y = None):
        """
        Define um nó

        Args:
            i (int): numero de id do nó
            x (float): Vetor de coordenadas ou coordenada x.
            y (float, optional): coordenada y.

        Returns:
            None.

        """
        self.id = i

        if isinstance(x, (tuple, list)):
            self.coord = np.array(x)
        else:
            self.coord = np.array([x, y])

        self.forcas = np.array([0., 0.])

        self.apoio = np.array([False, False])

        self.u = np.array([0., 0.])

        self.du = np.array([[0., 0.]])

    def __repr__(self):
        return "Node #%i @ [%.2f, %.2f], F=[%.2f, %.2f] %s%s" % (self.id+1, self.x, self.y, *self.forcas, "x"*self.apoio[0], "y"*self.apoio[1])


class Elemento:
    def __init__(self, i, mat: Material, nos):
        """
        Definição genérica de elemento.

        Args:
            i (int): id do elemento.
            mat (Material): Material do elemento.
            nos (Nodes[]): Lista de nós do elemento.

        Returns:
            None.

        """
        self.id = i
        self.nos = nos
        # lista de graus de liberdade do elemento
        self.gls = [2*no.id + i for no in nos for i in range(0,2)]

        # área do elemento
        self.Ae = None

        # tipo de célula, para exportação no VTK
        self.vtkcelltype = None

        # matriz de coordenadas nodais
        self.xe = np.array([no.coord for no in nos])

        # definiçao do material
        self.mat = mat
        self.E = mat.E
        self.nu = mat.nu

        # dicionario de parametros extras para leis constitutivas avançadas
        self.matprops = {}

        self.stress = np.array([0., 0., 0.])


    def __init2__(self):
        """
        Chamada após a inicialização do elemento, para armazenar os valores das
        matrizes calculadas nas classes especializadas.

        Returns:
            None.

        """
        self.D = self.De()
        self.K = self.Ke()

        if self.vtkcelltype is None:
            raise NotImplementedError("Tipo de célula VTK não definida")
        self.K = self.Ke()

        if self.Ae is None:
            # Fórmula de área
            # x1  x2  x3 ... xn
			# y1  y2  y3 ... yn
			# A = 1/2 * [ (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + ... + (xn*y1 - x1*yn) ]
            x = self.xe[0]
            y = self.xe[1]
            self.Ae = (x @ np.roll(y, -1) - np.roll(x, -1) @ y)/2
            #raise NotImplementedError("Área da célula não definida")


    def ue(self):
        """
        Retornar os deslocamentos nodais do elemento [u1, v1, u2, v2, ...]

        Returns:
            float[]: vetor de deslocamentos nodais.

        """
        return np.array([no.u[i] for no in self.nos for i in range(0,2)])


    def xe2(self):
        """
        Retonar as coordenadas nodais do elemento [x1, y1, x2, y2, ...]

        Returns:
            float[]: vetor de coordenadas nodais.

        """
        return np.array([no.coord[i] for no in self.nos for i in range(0,2)])


    def De(self, **kwargs):
        """
        Matriz constitutiva do Elemento Bidimensional

        Args:
            **kwargs (dict): dicionario de parametros extras para o modelo, opcional.

        Returns:
            float[][]: Matriz constitutiva.

        """
        return self.mat.De(**kwargs)


    def Ji(self, xi = 0, yi = 0):
        """
        Definição genérica da matriz Jacobiana

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        Returns:
            float[][]: Matriz Jacobiana.

        """
        return np.diag((1,1))


    def detJi(self, xi = 0, yi = 0):
        """
        Determinante Jacobiano

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        Returns:
            float: Determinante da Matriz Jacobiana.

        """
        J = self.Ji(xi,yi)
        return J[0,0]*J[1,1] - J[0,1]*J[1,0]

    def invJi(self, xi = 0, yi = 0):
        """
        Inversa da matriz Jacobiana

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        Returns:
            float[][]: Matriz Jacobiana invertida.

        """
        J = self.Ji(xi,yi)
        return np.array([[J[1,1], -J[0,1]], [-J[1, 0], J[0, 0]]])/self.detJi(xi, yi)

    def Ne(self, x = 0, y = 0):
        """
        Definição genérica de um vetor de funções interpoladores.

        Args:
            x, y (float): coordenadas globais x e y.

        """
        raise NotImplementedError("Função Interpoladora global não definida")

    def Ni(self, xi = 0, yi = 0):
        """
        Definição genérica de um vetor de funções interpoladores.

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        """
        raise NotImplementedError("Função Interpoladora local não definida")

    def __localParaGlobal(self, xi, yi):
        """
        Converte coordenadas locais para globais.

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        Returns:
            float[2]: Vetor de coordenadas globais [x, y].

        """

        return self.Ni(xi, yi) @ self.xe2()

    def Be(self, x = 0, y = 0):
        """
        Definição genérica da matriz de derivadas das funções interpoladores

        Args:
            x, y (float): coordenadas globais x e y.

        """
        raise NotImplementedError("Matriz de derivadas globais não definida")

    def Bi(self, xi = 0, yi = 0):
        """
        Definição genérica da matriz de derivadas das funções interpoladores

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        """
        raise NotImplementedError("Matriz de derivadas locais não definida")


    def Ke(self):
        """
        Definição genérica da matriz de rigidez do elemento
        """
        raise NotImplementedError("Matriz de rigidez não definida")


    def K_inc(self, n):
        """
        Definição genéria da matriz de rigidez do elemento na formulação incremental

        Args:
            n (int): passo do incremento

        """
        raise NotImplementedError("Matriz de rigidez incremental não definida")


    def Si(self, xi = 0, yi = 0):
        """
        Definição genérica do vetor de tensões.

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        """
        return self.D @ self.Bi(xi, yi) @ self.ue()

    def Se(self, x = 0, y = 0):
        """
        Definição genérica do vetor de tensões.

        Args:
            x, y (float): coordenadas globais.

        """
        return self.D @ self.Be(x, y) @ self.ue()


class CST(Elemento):
    def __init__(self, i, mat, nos):
        """
        Define um elemento triangular linear de deformação constante CST.

        Args:
            i (int): id do elemento.
            mat (Material): Material do elemento.
            nos (Nodes[3]): Lista com 3 nós para o elemento.

        """
        # inicializa o elemento genérico
        Elemento.__init__(self, i, mat, nos)

        # representação no padrão VTK do triangular linear
        self.vtkcelltype = 5

        # calcula a área do triangulo
        x = self.xe.transpose()[0]
        y = self.xe.transpose()[1]
        self.Ae = (x[1]*y[2] + x[0]*y[1] + x[2]*y[0] - x[1]*y[0] - x[0]*y[2] - x[2]*y[1])/2

        # calcula os parametros a1, b1, c1, a2, b2 ....
        self.a = [x[(i+1)%3]*y[(i+2)%3] - x[(i+2)%3]*y[(i+1)%3] for i in range(0, 3)]
        self.b = [y[(i+1)%3] - y[(i+2)%3] for i in range(0, 3)]
        self.c = [-x[(i+1)%3] +x[(i+2)%3] for i in range(0, 3)]

        # atribui as variaveis finais
        Elemento.__init2__(self)



    def Ne(self, xi, yi):
        N = [(self.a[i] + self.b[i]*xi + self.c[i]*yi)/(2*self.Ae) for i in range(0,3)]
        return np.array([[N[0], 0, N[1], 0, N[2], 0],
                         [0, N[0], 0, N[1], 0, N[2]]], dtype='float32')


    def Be(self, xi = 0, yi = 0):
        return (1/(2*self.Ae))*np.array(
            [[self.b[0], 0, self.b[1], 0, self.b[2], 0],
             [0, self.c[0], 0, self.c[1], 0, self.c[2]],
             [self.c[0], self.b[0], self.c[1], self.b[1], self.c[2], self.b[2]]]
            , dtype='float32')


    def Ke(self):
        return (self.mat.t if self.mat.EPT else 1)*self.Ae*(self.Be().transpose() @ self.D @ self.Be())


    def K_inc(self, n):
        # recalcula
        x = [no.coord[0] + no.du[n][0] for no in self.nos]
        y = [no.coord[1] + no.du[n][1] for no in self.nos]

        self.Ae = (x[1]*y[2] + x[0]*y[1] + x[2]*y[0] - x[1]*y[0] - x[0]*y[2] - x[2]*y[1])/2
        self.a = [x[(i+1)%3]*y[(i+2)%3] - x[(i+2)%3]*y[(i+1)%3] for i in range(0, 3)]
        self.b = [y[(i+1)%3] - y[(i+2)%3] for i in range(0, 3)]
        self.c = [-x[(i+1)%3] +x[(i+2)%3] for i in range(0, 3)]

        # recalcula as matrizes com os abc's atualizados
        Elemento.__init2__(self)

        return self.K


class ORTH4(Elemento):
    def __init__(self, i, mat, nos):
        """
        Define um elemento retangular ortogonal linear de 4 nós.

        Args:
            i (int): id do elemento.
            mat (Material): Material do elemento.
            nos (Nodes[4]): Lista com os nós do elemento.

        """
        # inicializa o elemento genérico
        Elemento.__init__(self, i, mat, nos)

        # representação no padrão VTK do retangulo ortogonal linear
        self.vtkcelltype = 9

        # projeções da diagonal do elemento 0-2
        #  3---2
        #  | / | b
        #  0---1
        #    a

        self.a, self.b = abs(self.xe[2] - self.xe[0])

        self.Ae = self.a*self.b

        Elemento.__init2__(self)


    def Ne(self, x, y):
        a = self.a
        b = self.b
        ab = self.Ae

        N1 = (x*b - x*y)/ab
        N2 = (x*y)/ab
        N3 = (y*a - x*y)/ab
        N4 = 1-(x*b + y*a - x*y)/ab
        return np.array([[N1, 0, N2, 0, N3, 0, N4, 0],
                         [0, N1, 0, N2, 0, N3, 0, N4]], dtype='float32')


    def Be(self, x, y):
        a = self.a
        b = self.b
        ab = self.Ae

        return (1/ab) * np.array(
			  [[b-y,  0 , y , 0 , -y ,  0 , y-b,  0 ],
               [ 0 , -x , 0 , x ,  0 , a-x,  0 , x-a],
               [ -x, b-y, x , y , a-x, -y , x-a, y-b]]
			  , dtype='float32')


    def Ke(self):
        """
        BtDB = lambda x, y: self.Be(x, y).transpose() @ self.D @ self.Be(x, y)

        intBDB = gauss2dn(BtDB, 2, 0, self.a, 0, self.b)
        """
        v = self.nu

        a = self.a
        b = self.b

        intBDB = (self.E/(1-v*v))*np.array(
            [[-(a*a*v-2*b*b-a*a)/(6*a*b),	-(v+1)/8,	(a*a*v+b*b-a*a)/(6*a*b),	(3*v-1)/8,	(a*a*v-2*b*b-a*a)/(12*a*b),	(v+1)/8,	-(a*a*v+4*b*b-a*a)/(12*a*b),	-(3*v-1)/8],
            [-(v+1)/8,	-(b*b*v-b*b-2*a*a)/(6*a*b),	-(3*v-1)/8,	-(b*b*v-b*b+4*a*a)/(12*a*b),	(v+1)/8,	(b*b*v-b*b-2*a*a)/(12*a*b),	(3*v-1)/8,	(b*b*v-b*b+a*a)/(6*a*b)],
            [(a*a*v+b*b-a*a)/(6*a*b),	-(3*v-1)/8,	-(a*a*v-2*b*b-a*a)/(6*a*b),	(v+1)/8,	-(a*a*v+4*b*b-a*a)/(12*a*b),	(3*v-1)/8,	(a*a*v-2*b*b-a*a)/(12*a*b),	-(v+1)/8],
            [(3*v-1)/8,	-(b*b*v-b*b+4*a*a)/(12*a*b),	(v+1)/8,	-(b*b*v-b*b-2*a*a)/(6*a*b),	-(3*v-1)/8,	(b*b*v-b*b+a*a)/(6*a*b),	-(v+1)/8,	(b*b*v-b*b-2*a*a)/(12*a*b)],
            [(a*a*v-2*b*b-a*a)/(12*a*b),	(v+1)/8,	-(a*a*v+4*b*b-a*a)/(12*a*b),	-(3*v-1)/8,	-(a*a*v-2*b*b-a*a)/(6*a*b),	-(v+1)/8,	(a*a*v+b*b-a*a)/(6*a*b),	(3*v-1)/8],
            [(v+1)/8,	(b*b*v-b*b-2*a*a)/(12*a*b),	(3*v-1)/8,	(b*b*v-b*b+a*a)/(6*a*b),	-(v+1)/8,	-(b*b*v-b*b-2*a*a)/(6*a*b),	-(3*v-1)/8,	-(b*b*v-b*b+4*a*a)/(12*a*b)],
            [-(a*a*v+4*b*b-a*a)/(12*a*b),	(3*v-1)/8,	(a*a*v-2*b*b-a*a)/(12*a*b),	-(v+1)/8,	(a*a*v+b*b-a*a)/(6*a*b),	-(3*v-1)/8,	-(a*a*v-2*b*b-a*a)/(6*a*b),	(v+1)/8],
            [-(3*v-1)/8,	(b*b*v-b*b+a*a)/(6*a*b),	-(v+1)/8,	(b*b*v-b*b-2*a*a)/(12*a*b),	(3*v-1)/8,	-(b*b*v-b*b+4*a*a)/(12*a*b),	(v+1)/8,	-(b*b*v-b*b-2*a*a)/(6*a*b)]]
            , dtype='float32')

        return (self.mat.t if self.mat.EPT else 1)*intBDB


class QUAD4(Elemento):
    def __init__(self, i, mat, nos):
        """
        Define um elemento quadrado linear de 4 nós.

        Args:
            i (int): id do elemento.
            mat (Material): Material do elemento.
            nos (Nodes[4]): Lista com os nós do elemento.

        """
        # inicializa o elemento genérico
        Elemento.__init__(self, i, mat, nos)

        # representação no padrão VTK do quadrado linear
        self.vtkcelltype = 9

        Elemento.__init2__(self)


    def Ne(self, xi, yi):
        N = [(1+xi)*(1-yi)/4, (1+xi)*(1+yi)/4, (1-xi)*(1+yi)/4, (1-xi)*(1-yi)/4]
        return np.array([[N[0], 0, N[1], 0, N[2], 0, N[3], 0],
                         [0, N[0], 0, N[1], 0, N[2], 0, N[3]]], dtype='float32')

    def Ji(self, xi, yi):
        return (1/4) * np.array([[ 1-yi, 1+yi, -1-yi, -1+yi],
                                 [-1-xi, 1+xi,  1-xi, -1+xi]], dtype='float32') @ self.xe

    def Bi(self, xi, yi):
        B = self.invJi(xi, yi) @ np.array([[ 1-yi, 1+yi, -1-yi, -1+yi],
                                           [-1-xi, 1+xi,  1-xi, -1+xi]], dtype='float32')

        return np.array(
			  [[ B[0,0],   0   , B[0,1],   0   , B[0,2],   0   , B[0,3],   0    ],
               [   0   , B[1,0],   0   , B[1,1],   0   , B[1,2],   0   , B[1,3] ],
               [ B[1,0], B[0,0], B[1,1], B[0,1], B[1,2], B[0,2], B[1,3], B[0,3] ]]
			  , dtype='float32')


    def Ke(self):
        BtDB = lambda xi, yi: self.Bi(xi, yi).transpose() @ self.D @ self.Bi(xi, yi) * self.detJi(xi, yi)

        intBDB = gauss2d2(BtDB)

        return (self.mat.t if self.mat.EPT else 1)*intBDB
