import pytest

gp = pytest.importorskip("gurobipy", reason="Gurobi is optional for core tests")
from gurobipy import GRB


def test_gurobi_mip_smoke():
    # Create a new model
    m = gp.Model("mip1")
    m.setParam("OutputFlag", 0)

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Set objective
    m.setObjective(x * y + 2 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")

    # Optimize model
    m.optimize()

    assert m.Status == GRB.OPTIMAL
    assert m.objVal == pytest.approx(2.0)
