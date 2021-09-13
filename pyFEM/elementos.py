# -*- coding: utf-8 -*-
"""
Módulo de Elementos

@author: César Eduardo Petersen

@date: Mon Sep  6 17:31:10 2021

Inclui as definições de nós e elementos.

"""

import numpy as np
from .materiais import *

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
            
        self.x = self.coord[0]
        self.y = self.coord[1]
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
        
        # tipo de célula, para exportação no VTK
        self.vtkcelltype = 0
        
        # acumula as coordenadas nodais
        self.x = [no.x for no in nos]
        self.y = [no.y for no in nos]
        
        # definiçao do material
        self.mat = mat
        self.E = mat.E
        self.nu = mat.nu
        
        # tensoes
        self.sigma = np.array([0., 0., 0.])
        
        
    def __init2__(self):
        """
        Chamada após a inicialização do elemento, para armazenar os valores das
        matrizes calculadas nas classes especializadas.

        Returns:
            None.

        """
        self.D = self.De()
        self.B = self.Be()
        self.K = self.Ke()
        
        
    def ue(self):
        """
        Retonar os deslocamentos nodais do elemento

        Returns:
            float[]: vetor de deslocamentos nodais.

        """
        return np.array([no.u[i] for no in self.nos for i in range(0,2)])
                

    def De(self):
        """
        Matriz constitutiva do Elemento Bidimensional

        Returns:
            float[][]: Matriz constitutiva.

        """
        E = self.E
        nu = self.nu
        if(self.mat.EPT):
            return (E/(1-nu**2))*np.array(
                [[ 1,nu, 0],
                 [nu, 1, 0],
                 [ 0, 0, (1-nu)/2]]
                , dtype='float32')
        else:
            return (E/((1-nu)*(1-2*nu)))*np.array(
                [[1-nu,  nu, 0],
                 [ nu ,1-nu, 0],
                 [  0 ,  0 , (1-2*nu)/2]]
                , dtype='float32')
    
        
    def Ne(self, xi, yi):
        """
        Definição genérica de um vetor de funções interpoladores.

        Args:
            xi, yi (float): coordenadas locais ξ e η ∈ [-1,1].

        """
        pass
    
    def Be(self):
        """
        Definição genérica da matriz de derivadas das funções interpoladores
        """
        pass
    
    def Ke(self):
        """
        Definição genérica da matriz de rigidez do elemento
        """
        pass
    
    def K_inc(self, n):
        """
        Definição genéria da matriz de rigidez do elemento na formulação incremental

        Args:
            n (int): passo do incremento

        """
        pass
    
    
    
class CST(Elemento):
    def __init__(self, i, mat, no1, no2, no3):
        """
        Define um elemento triangular linear de deformação constante CST.

        Args:
            i (int): id do elemento.
            mat (Material): Material do elemento.
            no1 (Node): nó 1 do elemento.
            no2 (Node): nó 2 do elemento.
            no3 (Node): nó 3 do elemento.

        Returns:
            None.

        """
        # inicializa o elemento genérico
        Elemento.__init__(self, i, mat, [no1, no2, no3])

        # representação no padrão VTK do triangular linear
        self.vtkcelltype = 5
        
        # calcula a área do triangulo
        x = self.x
        y = self.y
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

        
    def Be(self):
        return (1/(2*self.Ae))*np.array(
            [[self.b[0], 0, self.b[1], 0, self.b[2], 0],
             [0, self.c[0], 0, self.c[1], 0, self.c[2]],
             [self.c[0], self.b[0], self.c[1], self.b[1], self.c[2], self.b[2]]]
            , dtype='float32')
    
    
    def Ke(self):
        return (self.mat.t if self.mat.EPT else 1)*self.Ae*(self.B.transpose() @ self.De() @ self.B)

        
    def K_inc(self, n):
        # recalcula 
        x = [no.x + no.du[n][0] for no in self.nos]
        y = [no.y + no.du[n][0] for no in self.nos]
        
        self.Ae = (x[1]*y[2] + x[0]*y[1] + x[2]*y[0] - x[1]*y[0] - x[0]*y[2] - x[2]*y[1])/2
        self.a = [x[(i+1)%3]*y[(i+2)%3] - x[(i+2)%3]*y[(i+1)%3] for i in range(0, 3)]
        self.b = [y[(i+1)%3] - y[(i+2)%3] for i in range(0, 3)]
        self.c = [-x[(i+1)%3] +x[(i+2)%3] for i in range(0, 3)]
        
        # recalcula as matrizes com os abc's atualizados
        Elemento.__init2__(self)
        
        return self.K
        
        