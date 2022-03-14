import time
import pyFEM

from pyFEM.elementos import ORTH4

# inicia contagem do tempo
start_time = time.time()

# carrega o arquivo de malha do gmsh
mesh = pyFEM.pre.Malha("malha_quad.msh")

# define e aplica o material
mat = pyFEM.materiais.Material(2e+7, 0.3, 0.1)
mesh.defineElementos(mat, {'quad': ORTH4})
#mesh.defineElementos(mat)

# aplica os apoios
mesh.Apoios(mesh.grupos['apoio'])

# aplica as forças
mesh.Forcas(mesh.linhas['topo'], [0, -100])

# inicializa o solver
solv = pyFEM.solver.Solver(mesh)

# resolve
solv.calcular()

# exporta os dados para VTK
solv.exportar("malha_quad")

# tempo de processamento
elapsed_time = time.time() - start_time

# deslocamento vertical do nó inferior da extremidade livre (Nó #1)
umx = solv.U[1*2+1]*1000

# deslocamento teórico
umt = -1000*100*8**4/(8*2e+7/15)

print("%5i elementos | %5i nós | %7.3f seg \n u = %.3f mm | ut = %.1f mm\n" % (len(mesh.nodes), len(mesh.elementos), elapsed_time, umx, umt))
