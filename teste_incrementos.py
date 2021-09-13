import time
import pyFEM

# inicia contagem do tempo
start_time = time.time()

# carrega o arquivo de malha do gmsh
mesh = pyFEM.pre.Malha("malha.msh")

# define e aplica o material
mat = pyFEM.materiais.Material(2e+7, 0.3, 0.1)
mesh.defineElementos(mat)

# aplica os apoios 
mesh.Apoios(mesh.grupos['apoio'])

# aplica as forças
mesh.Forcas(mesh.linhas['topo'], [0, -100])

# inicializa o solver
solv = pyFEM.solver.Solver(mesh)

# resolve e exporta
solv.calcularInc(10, "malhaA")

# tempo de processamento
elapsed_time = time.time() - start_time

print(elapsed_time)