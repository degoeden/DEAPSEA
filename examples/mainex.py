import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder)
import src.ga as ga
import time

def objective(ind):
    return [ind["a"]**2 + abs(ind["b"])]

BOUNDS = {
    'a' : (0., 10.),
    'b' : (-5., 5.)
}

BITS = {
    'a' : 10,
    'b' : 10
}


ga = ga.DeapSeaGa(objective, BOUNDS, BITS, NGEN=30, NPOP=50, NWORKERS=4, PATIENCE=40, TOL=1e-3, csv_path="examples/results.csv")
print(ga.run())