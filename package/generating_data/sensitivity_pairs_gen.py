from package.generating_data.sensitivity_single_pair_gen import main as main_single
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

def main(
        BASE_PARAMS_LOAD="package/constants/base_params.json",
        VARY_LOAD_DICT = {
            "alpha": "package/constants/vary_single__sensitivity.json",
            "zeta":  "package/constants/vary_single__sensitivity.json",
            "r": "package/constants/vary_single_r_sensitivity.json",
            "delta": "package/constants/vary_single_delta_sensitivity.json",
            "K": "package/constants/vary_single_K_sensitivity.json",
            "lambda": "package/constants/vary_single_lambda_sensitivity.json",
            "kappa": "package/constants/vary_single_kappa_sensitivity.json",
            "mu": "package/constants/vary_single_mu_sensitivity.json",
            "num_beta": "package/constants/vary_single_num_beta_sensitivity.json",
            "num_gamma": "package/constants/vary_single_num_gamma_sensitivity.json",
            "a_chi": "package/constants/vary_single_a_chi_sensitivity.json",
            "b_chi": "package/constants/vary_single_b_chi_sensitivity.json",
            "num_users": "package/constants/vary_single_num_users_sensitivity.json",
            "num_firms": "package/constants/vary_single_num_firms_sensitivity.json",
        },
        pairs_list = [
            ("alpha", "zeta"),
            ("r", "delta"),
            ("K", "lambda"),
            ("kappa", "mu"),
            ("num_beta", "num_gamma"),
            ("a_chi", "b_chi"),
            ("num_users", "num_firms")
        ]
    ) -> str: 

    fileName_outer = produce_name_datetime("sens_2d_all")
    createFolder(fileName_outer)

    print("TOTAL RUNSish: ", len(pairs_list)*64*4*4)
    
    fileName_list = []
    for (var_1, var_2) in pairs_list:
        VARY_LOAD_1 = VARY_LOAD_DICT[var_1]
        VARY_LOAD_2 = VARY_LOAD_DICT[var_2]
        print(VARY_LOAD_1,VARY_LOAD_2)
        fileName = main_single(BASE_PARAMS_LOAD = BASE_PARAMS_LOAD, VARY_LOAD_1=VARY_LOAD_1, VARY_LOAD_2=VARY_LOAD_2 )
        fileName_list.append(fileName)

    print("fileName_list: ", fileName_list)
    save_object(fileName_list, fileName_outer + "/Data", "2D_sensitivity_fileName_list")

if __name__ == "__main__":
    results = main(
        BASE_PARAMS_LOAD="package/constants/sensitivity/base_params_sensitivity_2D.json",
        VARY_LOAD_DICT = {
            "alpha": "package/constants/sensitivity/vary_single_alpha_sensitivity.json",
            "zeta":  "package/constants/sensitivity/vary_single_zeta_sensitivity.json",
            "r": "package/constants/sensitivity/vary_single_r_sensitivity.json",
            "delta": "package/constants/sensitivity/vary_single_delta_sensitivity.json",
            "K": "package/constants/sensitivity/vary_single_K_sensitivity.json",
            "lambda": "package/constants/sensitivity/vary_single_lambda_sensitivity.json",
            "kappa": "package/constants/sensitivity/vary_single_kappa_sensitivity.json",
            "mu": "package/constants/sensitivity/vary_single_mu_sensitivity.json",
            "num_beta_segments": "package/constants/sensitivity/vary_single_beta_segments_sensitivity.json",
            "num_gamma_segments": "package/constants/sensitivity/vary_single_gamma_segments_sensitivity.json",
            "a_chi": "package/constants/sensitivity/vary_single_a_chi_sensitivity.json",
            "b_chi": "package/constants/sensitivity/vary_single_b_chi_sensitivity.json",
            "num_users": "package/constants/sensitivity/vary_single_num_users_sensitivity.json",
            "num_firms": "package/constants/sensitivity/vary_single_num_firms_sensitivity.json",
        },
        pairs_list = [
            #("alpha", "zeta"),
            #("r", "delta"),
            #("K", "lambda"),
            #("kappa", "mu"),
            #("num_beta_segments", "num_gamma_segments"),
            #("a_chi", "b_chi"),
            ("num_users", "num_firms")
        ]
    )