# -*- coding: utf-8 -*-
"""
Módulo de Processamento

@author: César Eduardo Petersen

@date: Wed Sep  8 16:59:34 2021

Resolve o sistema e exporta os resultados no padrão VTK

"""
import numpy as np
from scipy import linalg
from .pre import *    
from functools import reduce

class Solver:
    def __init__(self, mesh: Malha):
        """
        Inicializa o Solver com uma malha

        Args:
            mesh (Malha): Malha de elementos.

        Returns:
            None.

        """
        self.mesh = mesh
        
    def calcular(self):
        """
        Resolve o sistema.

        Returns:
            None.

        """
        # 2 graus de liberdade por nó
        self.ngl = len(self.mesh.nodes)*2
        
        # prepara a matriz de rigidez global
        self.Kg = np.zeros((self.ngl, self.ngl), dtype='float32')
        self.F = np.zeros((self.ngl), dtype='float32')
        self.U = np.zeros((self.ngl), dtype='float32')
        
        # itera por cada elemento
        for elem in self.mesh.elementos:
            # soma sua parcela de rigidez
            self.Kg[np.ix_(elem.gls, elem.gls)] += elem.K
        
        # separa os graus de liberdade Fixos e Livres
        glF = []
        glL = []
        for no in self.mesh.nodes:
            glF += [2*no.id + i for i in range(0,2) if no.apoio[i]]
            glL += [2*no.id + i for i in range(0,2) if not no.apoio[i]]
            self.F[2*no.id:2*no.id+2] = no.forcas 
        
        self.glF = glF
        self.glL = glL
            
        # Separando as submatrizes
        KFF = self.Kg[np.ix_(glF, glF)]
        KFL = self.Kg[np.ix_(glF, glL)]
        KLL = self.Kg[np.ix_(glL, glL)]
        
        FL = self.F[np.ix_(glL)]
        
        # calcula os deslocamentos livres
        self.U[np.ix_(glL)] = linalg.solve(KLL, FL)
        
        # calcula as reações de apoio
        self.F[np.ix_(glF)] = KFL @ self.U[np.ix_(glL)] - self.F[np.ix_(glF)]
        
        # aplica os resultados nos dicionarios de dados para exportalçao
        self.mesh.pointdata['deslocamento'] = self.U.reshape(-1, 2)
        self.mesh.pointdata['forcas'] = self.F.reshape(-1, 2)
        
        # repassa os deslocamentos calculados para cada nó para calculo dos ue's
        for no in self.mesh.nodes:
            no.u = self.U[2*no.id:2*no.id+2]
            
        # prepara o campo de tensões
        self.mesh.celldata['stress'] = np.zeros((len(self.mesh.elementos), 3))
          
        # itera pelos elementos calculando as tensões
        for elem in self.mesh.elementos:
            elem.stress = elem.D @ elem.B @ elem.ue()
            self.mesh.celldata['stress'][elem.id] = elem.stress
    
    
    def exportar(self, filename):
        """
        Gera um arquivo no formato VTK para visualização dos dados.

        Args:
            filename (string): Nome do arquivo, sem a extensão.

        Returns:
            None.

        """
        # cria o arquivo
        f = open("%s.vtk"%filename, "w")
        
        # escreve o cabeçario
        f.write("# vtk DataFile Version 4.2\n")
        f.write("%s\n" % filename)
        f.write("ASCII\n")
        
        # declara a geometria
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # declara os pontos
        nnos = len(self.mesh.nodes)
        f.write("POINTS %i double\n" % nnos)
        np.savetxt(f, self.mesh.points)
        
        # declara as celulas        
        nelem = len(self.mesh.elementos)
        # numero de dados esperados numero de elementos + numero de nos de cada elemento
        ndados = reduce(lambda acc, el: acc + 1 + len(el.nos), self.mesh.elementos, 0)
        f.write("\nCELLS %i %i\n" % (nelem, ndados)) 
        for el in self.mesh.elementos:
            f.write(str(len(el.nos)) + reduce(lambda acc, no: acc + " " + str(no.id), el.nos, "")+"\n")
        
        # declara o tipo de cada celula
        f.write("\nCELL_TYPES %i\n" % nelem)
        for el in self.mesh.elementos:
            f.write(str(el.vtkcelltype)+"\n")
        
        # declara os dados nodais
        f.write("\n\nPOINT_DATA %i\n" % nnos)
        for key, val in self.mesh.pointdata.items():
            # campo escalar
            if val.ndim == 1:
                f.write("\nSCALARS %s float\nLOOKUP_TABLE default\n" % key)
                np.savetxt(f, val)
            # campo vetorial
            else:
                f.write("\nVECTORS %s double\n" % key)
                # preenche a ultima coluna (Z) com zero caso sejam dados apenas X, Y
                if val.shape[1] == 2:
                    np.savetxt(f, np.pad(val, ((0,0),(0,1))))
                else:
                    np.savetxt(f, val)
                   
        # declara os dados de elementos
        f.write("\n\nCELL_DATA %i\n" % nelem)
        for key, val in self.mesh.celldata.items():
            if val.ndim == 1:
                f.write("\nSCALARS %s double\nLOOKUP_TABLE default\n" % key)
                np.savetxt(f, val)
            else:
                f.write("\nVECTORS %s double\n" % key)
                if val.shape[1] == 2:
                    np.savetxt(f, np.pad(val, ((0,0),(0,1))))
                else:
                    np.savetxt(f, val)
                    
        f.close()
        
        
        
            
            
        
            
            
        
        