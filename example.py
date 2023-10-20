import cvxpy as cp

x = cp.Variable(name='x')
y = cp.Variable(name='y')


objective = cp.Maximize(3*x + 4*y)

constraints = [
    x + 2*y <= 8,
    x + y <= 6,
    x >= 0,
    y >= 0
]

problem = cp.Problem(objective, constraints)


problem.solve(solver=cp.GUROBI)


if problem.status == cp.OPTIMAL:
    print("optimal:", problem.value)
    print("x =", x.value)
    print("y =", y.value)
else:
    print("No answer")
