# -*- coding: utf-8 -*-
"""
Módulo de Materiais

@author: César Eduardo Petersen

@date: Mon Sep  6 17:33:10 2021

Inclui Definição de Material Elastico Linear Homogeneo.

"""

class Material:
    def __init__(self, E, nu, t):
        """
        Define um material elastico-linear

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