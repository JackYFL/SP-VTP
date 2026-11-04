#%%
import sympy as sp

# CONSTANT MEASUREMENTS from FLYING UMI CAD with OAK
A = 85.073 # distance between gripper aruco tag center and the OAK camera center, along the horizontal direction
B = 42.547 # distance between gripper sufrace plane and the OAK camera center, along the gravity direction 
alpha = 75 # angle between OAK camera image plane surface and UMI gripper surface plane
alpha_radian = alpha * sp.pi / 180

# Compute x
x = sp.tan(alpha_radian) * B - A

# Define symbolic variable
z = sp.symbols('z', real=True)

# Equality One: A**2 + x**2 + 2Ax + B**2 = (z+y)**2
# Equality Two: A**2 + Ax + B**2 = z**2 + zy

z, y = sp.symbols('z y', real=True)

# Equations
eq1 = sp.Eq(A**2 + x**2 + 2*A*x + B**2, (z + y)**2)
eq2 = sp.Eq(x/(z+y), y/(A+x))

# Solve
solutions = sp.solve((eq1, eq2), (z, y), dict=True)

print(solutions)
# [{y: -71.2028027926171, z: -93.1861817318661}, {y: 71.2028027926171, z: 93.1861817318661}] choose z = 93.186181

#%%
# Double check the solution based on the similar triangle rules, equality two
import math
y = 71.2028027926171  # from the solution
z = 93.1861817318661  # from the solution
x = math.tan(alpha_radian) * B - A  # recompute x
# Double check
LHS = x/(z+y)
RHS = y/(A+x)
print(f'LHS: {LHS}, RHS: {RHS}')  # Should be equal
assert math.isclose(LHS, RHS, rel_tol=1e-5), "The solution does not satisfy the equality condition."

# %%
