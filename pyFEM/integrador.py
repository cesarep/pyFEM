# -*- coding: utf-8 -*-
"""
Módulo do Integrados

@author: César Eduardo Petersen

@date: Mar 4 2022

Define a integração pelo método da Quadratura de Gauss

"""

from numpy.polynomial.legendre import leggauss

def gauss1dn(func, n, xi=-1, xf=1):
    """
	Integra uma função no domínio x ∈ [xi, xf] pelo método da
	Quadratura de Gauss, utilizando n pontos

	Args:
	    func (function): Função a ser integrada.
        n (int): Pontos da quadratura.
	    xi, xf (float): Domínio de integração em x, padrão (-1,1).

	Returns:
	    acc (float): Valor aproximado da integral.

    """
    dx = xf-xi

    acc = 0

    for pi, wi in zip(*leggauss(n)):
        px = (pi+1)*dx/2 + xi
        acc += wi*func(px)

    return acc*dx/2

def gauss1d2(func, xi=-1, xf=1):
    return gauss1dn(func, 2, xi, xf)
	

def gauss2dn(func, n, xi=-1, xf=1, yi=-1, yf=1):
	"""
	Integra uma função no domínio x ∈ [xi, xf], y ∈ [yi, yf] pelo método da
	Quadratura de Gauss, utilizando n pontos

	Args:
	    func (function): Função a ser integrada.
        n (int): Pontos da quadratura.
	    xi, xf (float): Domínio de integração em x, padrão (-1,1).
	    yi, yf (float): Domínio de integração em y, padrão (-1,1).

	Returns:
	    acc (float): Valor aproximado da integral.

	"""
	dx = xf-xi
	dy = yf-yi

	acc = 0

	for pi, wi in zip(*leggauss(n)):
		for pj, wj in zip(*leggauss(n)):
			px = (pi+1)*dx/2 + xi
			py = (pj+1)*dy/2 + yi
			acc += wi*wj*func(px, py)

	return acc*dx*dy/4

def gauss2d2(func, xi=-1, xf=1, yi=-1, yf=1):
    return gauss2dn(func, 2, xi, xf, yi, yf)

def gauss2d3(func, xi=-1, xf=1, yi=-1, yf=1):
    return gauss2dn(func, 3, xi, xf, yi, yf)
