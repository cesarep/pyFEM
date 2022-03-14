# -*- coding: utf-8 -*-
"""
Nome

@author: cesar

@date: Sat Mar 12 01:11:43 2022

Descrição

"""

import pyFEM

from pyFEM.materiais import Material
from pyFEM.elementos import Node, ORTH4
from pyFEM.solver import ManualSolver

a = .10
b = .15

# 2---1
# |   |
# 3---0

nos = [Node(0, [a, 0]),
       Node(1, [a, b]),
       Node(2, [0, b]),
       Node(3, [0, 0])]

mat = Material(2e+7, .3, .1)

elem = ORTH4(0, mat, nos)

nos[2].apoio = [True, True]
nos[3].apoio = [True, True]

nos[1].forcas = [0, -100]

solv = ManualSolver(nos, [elem])

solv.calcular()

solv.exportar("testeA")


# 3---2
# |   |
# 0---1

nos = [Node(0, [0, 0]),
       Node(1, [a, 0]),
       Node(2, [a, b]),
       Node(3, [0, b])]

mat = Material(2e+7, .3, .1)

elem = ORTH4(0, mat, nos)

nos[0].apoio = [True, True]
nos[3].apoio = [True, True]

nos[2].forcas = [0, -100]

solv = ManualSolver(nos, [elem])

solv.calcular()

solv.exportar("testeB")
