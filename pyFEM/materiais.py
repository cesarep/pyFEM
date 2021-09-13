# -*- coding: utf-8 -*-
"""
Módulo de Materiais

@author: César Eduardo Petersen

@date: Mon Sep  6 17:33:10 2021

Inclui Definição de Materiais e Leis constitutivas.

"""

import numpy as np

class Material:
    def __init__(self, E, nu, t):
        """
        Define um material elastico-linear homogeneo.

        Args:
            E (float): Módulo de Elasticidade.
            nu (float): Coeficiente de Poisson.
            t (float): Espessura, caso esteja no Estado Plano de Tensoes, ou
                       nulo para considerar Estado Plano de Deformações

        Returns:
            None.

        """
        self.E = E
        self.nu = nu
        self.t = t
        self.EPT = (t > 0)
        
    def De(self, **kwargs):
        """
        Define uma matriz constitutiva para elementos bidimensionais

        Args:
            **kwargs (dict): dicionario de parametros extras para o modelo, opcional.

        Returns:
            float[3][3]: Matriz constitutiva do elemento.

        """
        E = self.E
        nu = self.nu
        if(self.EPT):
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