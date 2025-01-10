import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

def calculate_alpha(D_plus, D_minus, X_plus):
    """
    Calculates alpha using the provided formula:
    alpha = (D^+ - D^-)/(D^- X^+)
    """
    alpha = (D_plus - D_minus) / (D_minus * X_plus)
    return alpha

def calc_X(beta,c,gamma,e, omega):
    return (beta*c + gamma*e)/omega

def calculate_Q(W, P, C, k, beta, gamma, E, alpha, X, r, delta, L):
    """
    Calculates Q based on the given formula:
    Q = [(ln(W(P-C)k * beta - 1) + beta*P + gamma*E) * (alpha*X + 1) * (r + delta)] 
        / [(1 - delta)^L * (1 + r)]
    """
    try:
        term1 = np.log(W * (P - C) * k * beta - 1)
        term2 = beta * P + gamma * E
        numerator = (term1 + term2) * (alpha * X + 1) * (r + delta)
        denominator = (1 - delta)**L * (1 + r)
        Q = numerator / denominator
        return Q
    except Exception as e:
        print(f"Error in calculate_Q: {e}")
        return np.nan
    
#------------------------------
# 1) Define parameters
#------------------------------

beta_i  = 1.0      # example: slope wrt price
C_m     = 1.0      # example: marginal cost
D_plus = 30000
D_minus = 1#100

#ICE
beta_plus = 1.0
beta_minus = 0.25404965019667486
gamma_minus = 1.323175261441015e-08
gamma_plus = 3.9605791486638497
c_minus = 0.05623261991186667
c_plus = 0.16853363453157436
e_minus = 0.26599820413049985#THEY ARE THE SAME
e_plus = 0.26599820413049985
omega_minus= 0.7#0.5 + 0.2*1
omega_plus= 1.3#0.5 + 0.8*1
P_minus =  10000
P_plus = 100000
C_minus = 1000
C_plus = 50000
E_minus = 3000#THEY ARE THE SAME
E_plus = 3000

beta_avg = 0.5
P_avg = 30000
P_second_hand_car_offer = 10000
P_z = P_avg - P_second_hand_car_offer
gamma_avg = 0.5
E_avg = 6000

L_change_car = 120

r = 0.00417
S = 16
J = 10
#nu = 1
cost = 20000

E = 6000
delta = 0.001
L = 0

price_relative = 20000#just try see what happens
X_plus = calc_X(beta_plus,c_plus,gamma_plus,e_plus, omega_minus)
alpha = calculate_alpha(D_plus, D_minus, X_plus)
#print("alpha", alpha)
#quit()

beta = (beta_plus + beta_minus)/2
c= (c_plus + c_minus)/2
gamma= (gamma_plus + gamma_minus)/2
e = (e_plus + e_minus)/2
omega= (omega_plus + omega_minus)/2
X = calc_X(beta,c,gamma,e, omega)

# You can change these to suit your model

#------------------------------
# 2) Define ranges for Q and W
#------------------------------
W_min, W_max   =  0, 10000    # range for W (must be >0)
n_points       = 1000

Q_vals = np.logspace(0,7, n_points)
k  = 10#5     # example: curvature for exponent
#alpha = 10e4
#other_stuff = (1-delta)**(L)*(1+r)/((alpha*X +1)*(r+delta))
#print("other_stuff", other_stuff)
#B = Q_vals*other_stuff
B = Q_vals*(1-delta)**(L)*(1+r)/((alpha*X +1)*(r+delta))
print("B", B)

#print("other crap",beta*price_relative + gamma*E )
#print("thing in the exp", B - beta*price_relative - gamma*E)
#U_k = np.exp(k*(B - beta*price_relative - gamma*E))
#print("U", U_k)

#W_vals = S*J*U_k
#print("W_vals",W_vals)

W_vals = np.linspace(W_min, W_max, n_points)

# Create a meshgrid so we can compute P^*(Q,W) over a 2D plane
Q_grid, W_grid = np.meshgrid(Q_vals, W_vals)

#------------------------------
# 3) Compute the optimal price
#    P^*(Q,W) = C_m + 1/(k beta_i) [ 1 + lambertw( Arg ) ]
#    Arg = [exp(k Q) - 1] / W
#------------------------------
#Arg = (np.exp(k * Q_grid) - 1.0) / W_grid
Arg = (np.exp(k*(B - beta*cost - gamma*E)- 1.0)) / W_grid
print("Arg",Arg)

# Call scipy.special.lambertw; we typically use the principal branch
# If Arg < -1/e, lambertw might give complex results. 
# In that scenario, you may want to mask or handle those points.
LW   = lambertw(Arg, 0)  # principal branch
print("LW", LW)
# P^*
P_star = C_m + (1.0 + LW) / (k * beta_i)

print("P_star", P_star)
# For plotting, let’s show the ratio P^*/C_m
P_ratio = P_star / C_m

#------------------------------
# 4) Mask or filter invalid values (optional)
#    If the argument to lambertw is out of its domain, we may get NaNs.
#------------------------------
P_ratio_real = np.real(P_ratio)  # just in case some are complex
P_ratio_real[np.imag(P_ratio) != 0] = np.nan  # mask complex results

#------------------------------
# 5) Plot as a contour/heatmap
#------------------------------
plt.figure(figsize=(7,5))

# We can do a filled contour
cont = plt.contourf(
    Q_grid, W_grid, P_ratio_real, 
    levels=50, cmap='viridis'
)
plt.colorbar(cont, label='P*(Q,W) / C_m')
plt.xscale('log')
plt.xlabel("Quality (Q)")
plt.ylabel("W")
plt.title("Optimal Price Ratio P^*/C_m over (Q, W)")

plt.tight_layout()
plt.show()




