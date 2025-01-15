import numpy as np
import matplotlib.pyplot as plt

def calculate_alpha(D_plus, D_minus, X_plus):
    """
    Calculates alpha using the provided formula:
    alpha = (D^+ - D^-)/(D^- X^+)
    """
    alpha = (D_plus - D_minus) / (D_minus * X_plus)
    return alpha


def calc_X(beta,c,gamma,e, omega):
    return (beta*c + gamma*e)/omega


def calc_B(u_star, r, delta):
    return u_star*(1+r)(r+delta)


def calculate_Q(nu, W, P, C, k, beta, gamma, E, alpha, X, r, delta):
    """
    Calculates Q based on the given formula:
    Q = [(ln(W(P-C)k * beta - 1) + beta*P + gamma*E) * (alpha*X + 1) * (r + delta)] 
        / [(1 - delta)^L * (1 + r)]
    """
    try:
        term1 = (1/k)*np.log(W * (P - C) * k * nu * beta - 1)
        term2 = beta * P + gamma * E
        numerator = (term1 + term2) * (alpha * X + 1) * (r + delta)
        denominator = (1 + r)
        Q = numerator / denominator
        return Q
    except Exception as e:
        print(f"Error in calculate_Q: {e}")
        return np.nan
    

def main():
    # Example values
    D_plus = 4000
    D_minus = 500#100
    print("D_plus", D_plus)

    #ICE
    beta_plus = 1.0
    beta_minus = 0.25404965019667486
    gamma_minus = 0#1.323175261441015e-08
    gamma_plus = 0#3.9605791486638497
    c_minus = 0.05623261991186667
    c_plus = 0.16853363453157436
    e_minus = 0.26599820413049985#THEY ARE THE SAME
    e_plus = 0.26599820413049985

    omega_minus= 0.7#0.5 + 0.2*1
    omega_plus= 1.3#0.5 + 0.8*1

    r = 0.00417

    #nu = 1
    cost = 17000

    E = 6000
    delta = 0.001

    W_min, W_max = 1, 5
    W_points =  1000
    
    n_points = 50
    
    nu = 0.00001

    P_values = [20000, 30000,50000]
    X_plus = calc_X(beta_plus,c_plus,gamma_plus,e_plus, omega_minus)
    print("X plus",X_plus)
    alpha = calculate_alpha(D_plus, D_minus, X_plus)
    print("alpha", alpha)
    beta = (beta_plus + beta_minus)/2
    print("beta", beta)
    c= (c_plus + c_minus)/2
    print("c", c)
    gamma= (gamma_plus + gamma_minus)/2
    print("gamma", gamma)
    e = (e_plus + e_minus)/2
    print("e", e)
    omega= (omega_plus + omega_minus)/2
    print("omega", omega)
    X = calc_X(beta,c,gamma,e, omega)
    print("X", X)


    #alpha_vec = np.logspace(-3, 4, 10000)
    av_distance = D_plus/(alpha*X + 1)
    print("av_distance ", av_distance )#SO YEHA THIS DOES NOT WORK
    #plt.plot(alpha_vec,av_distance)
    #plt.show()
    #quit()

    # Generate W and n values
    W_vals = np.logspace(W_min, W_max, W_points)
    k_vals = np.arange(29, 32)


    Q_grids = []

    for P in P_values:
        Q_grid = np.zeros((len(W_vals), len(k_vals)))
        for i, W in enumerate(W_vals):
            for j, k in enumerate(k_vals):
                Q_grid[i, j] = calculate_Q(nu,W, P, cost, k, beta, gamma, E, alpha, X, r, delta)
        Q_grids.append(Q_grid)
        #print(Q_grid)

    print("Min Q", np.min(Q_grids))
    print("Max Q", np.max(Q_grids))

    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = [f"P = {P}" for P in P_values]

    for i, ax in enumerate(axes):
        c = ax.contourf(k_vals, W_vals, Q_grids[i], levels=50, cmap='viridis')
        fig.colorbar(c, ax=ax, label='Q')
        ax.set_title(titles[i])
        ax.set_xlabel("k")
        ax.set_ylabel("W")
        #ax.set_yscale("log")

    plt.suptitle("Heatmaps of Q for Different P Values", fontsize=16)
    plt.show()
if __name__ == "__main__":
    main()
    
