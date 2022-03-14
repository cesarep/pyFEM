# -*- coding: utf-8 -*-
"""
Módulo de pré-processamento

@author: César Eduardo Petersen

@date: Wed Sep  1 11:28:45 2021

Inclui da definição da malha, importação do gmsh e condições de contorno.

"""

import numpy as np
import meshio
#import gmsh
from .elementos import *

##### MALHADOR

class Malha:
    def __init__(self, filename):
        """
        Define uma malha de elementos a partir de um arquivo de malha do gmsh

        Args:
            filename (string): caminho do arquivo de malha .msh.

        Returns:
            None.

        """
        self.points = []
        self.elementos = []

        ### teste utilizando API própria do GMSH
        #gmsh.initialize()
        #gmsh.open(filename)
        #self.model = gmsh.model
        #self.mesh = gmsh.model.mesh
        # nós brutos em forma de matriz
        #self.points = self.mesh.getNodes()[1].reshape((-1,3))

        # importação da malha pelo gmsh
        self.mesh = meshio.read(filename)

        # apaga os dados do gmsh, só atrapalham
        self.mesh.cell_data = {}
        self.mesh.point_data = {}

        # nós brutos em forma de matriz
        self.points = self.mesh.points
        # Objetos de Nó
        self.nodes = [Node(i, pto[0], pto[1]) for i, pto in enumerate(self.points)]

        # dicionarios de dados para campos de resultados nodais e de elementos
        self.pointdata = {}
        self.celldata = {}

        ### com API GMSH
        # pega grupos fisicos
        #self.grupos = {}
        # monta dicionário de dados com nome do grupo e lista de nós
        #for g in self.model.getPhysicalGroups():
        #    self.grupos[self.model.getPhysicalName(1,g[1])] = mesh.getNodesForPhysicalGroup(1, g[1])[0]


        # prepara dicionario para armazenar os grupos de elementos
        self.grupos = {}
        self.linhas = {}
        # itera pelos grupos
        for g, val in self.mesh.cell_sets_dict.items():
            # ignora os grupos internos do gmsh
            if not g.startswith("gmsh:"):
                # prepara um set vazio para id dos nos
                self.grupos[g] = set()
                if 'line' in val.keys():
                    self.linhas[g] = val['line']

                # itera pelos elementos do grupo
                for eltype, ids in val.items():
                    # agrupa todos os nos
                    nos = self.mesh.cells_dict[eltype][ids].reshape(-1)
                    # adiciona cada nó no set,já previne valores repetidos
                    for n in nos:
                        self.grupos[g].add(n)

        # armazena os elementos brutos do malha
        self.elems = self.mesh.cells_dict

    def defineElementos(self, mat: Material, elems = {'triangle': CST, 'quad': QUAD4}):
        """
        Cria as instancias dos elementos a partir dos dados brutos da malha e
        define seus materiais.

        Args:
            mat (Material): Material dos elementos.
            elems (dict): Dicionario relacionando os elementos com suas instancia
                          Padrão: {'triangle': CST, 'quad': QUAD4}

        Returns:
            None.

        """
        # itera pelos tipos de elementos
        for eltype, val in self.mesh.cells_dict.items():
            if eltype in elems.keys():
                self.elementos += [elems[eltype](i, mat, [self.nodes[no] for no in nos]) for i, nos in enumerate(val)]


    def Apoios(self, idnos, apoio = [True, True]):
        """
        Define condições de contorno de apoio para um grupo de nós.

        Args:
            idnos (int[]): Grupo de ids de nós para serem fixados.
            apoio (Bool[2], optional): Vetor do tipo de apoio, True => Fixo, False => Livre
                por padrão [True, True].

        Returns:
            None.

        """
        # itera pelos nós do grupo
        for i in idnos:
            # aplica o apoio
            self.nodes[i].apoio = np.array(apoio)


    def Forcas(self, idlins, q = [0., 0.]):
        """
        Define as forças de superficie para um grupo de elementos.

        Args:
            idlins (int[]): Grupo de ids de linhas para serem carregadas.
            q (float[2]): Carregamento [qx, qy] na direção global, a ser aplicado.

        Returns:
            None.

        """
        # itera pelas linhas do grupo
        for i in idlins:
            # obtem o par de nós da linha
            n1, n2 = self.mesh.cells_dict['line'][i]
            no1 = self.nodes[n1]
            no2 = self.nodes[n2]
            # calcula o comprimento [Lx, Ly]
            L = abs(no2.coord - no1.coord)

            # Calcula a força total [Qx, Qy]
            Q = q*np.flip(L) # => [qx*Ly, qy*Lx]

            # aplica metade em cada nó.
            no1.forcas += Q/2
            no2.forcas += Q/2
