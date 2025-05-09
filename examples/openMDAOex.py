import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder)
os.environ["OPENMDAO_KEEP_TEMPFILES"] = "false"
import src.ga as ga
import time
from openmdao.api import Problem
import openmdao.api as om


class SimpleObjectiveComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('a', val=0.0)
        self.add_input('b', val=0.0)
        self.add_output('obj', val=0.0)

    def compute(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        outputs['obj'] = a**2 + abs(b)

def objective(ind):
    # Create the problem (or load a reusable template)
    prob = Problem(reports=None)

    prob.model.add_subsystem("simple_obj", SimpleObjectiveComp(), promotes=["*"])

    # Setup and assign design variables from `ind`
    prob.setup()
    prob.set_val("a", ind["a"])
    prob.set_val("b", ind["b"])

    # Run model
    prob.run_model()

    # Extract and return the objective
    result = prob.get_val("obj")
    return [result.item()]

BOUNDS = {
    'a' : (0., 10.),
    'b' : (-5., 5.)
}

BITS = {
    'a' : 8,
    'b' : 8
}

ga = ga.DeapSeaGa(objective, BOUNDS, BITS, NGEN=100, NPOP=10, NWORKERS=4, PATIENCE=30, TOL=1e-3, csv_path="examples/results.csv")
print(ga.run())