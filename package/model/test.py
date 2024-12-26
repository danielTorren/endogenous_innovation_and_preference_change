def generate_utilities(
    self,
    switcher_indices: np.ndarray
) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Compute the utilities & distances ONLY for the subset of users in `switcher_indices`.
    Returns
    -------
    utilities_matrix : np.ndarray
        Shape = (S, total_columns), where S = len(switcher_indices).
    car_options : list
        The list of all new + second-hand cars (columns in the returned arrays).
    d_matrix : np.ndarray
        Shape = (S, total_columns).
    """

    # Number of switchers
    S = len(switcher_indices)

    # ----------------------------------------------------------------
    # 0) Build sub-vectors for switchers
    #    (beta, gamma, d_i_min, and any other user-level data you need).
    # ----------------------------------------------------------------
    sub_beta_vec = self.beta_vec[switcher_indices]
    sub_gamma_vec = self.gamma_vec[switcher_indices]
    sub_d_i_min_vec = self.d_i_min_vec[switcher_indices]

    # If your model needs to differentiate who currently owns a car:
    sub_owns_car_mask = (self.users_current_vehicle_type_vec[switcher_indices] > 1)
    
    # Similarly, if you use second_hand_merchant_offer_price to offset a user’s 
    # cost when selling their old car:
    sub_price_owns_car_vec = np.where(
        sub_owns_car_mask,
        self.second_hand_merchant_offer_price[switcher_indices],
        0
    )

    # ----------------------------------------------------------------
    # 1) Build the dictionary of user-level data for switchers
    #    so we can pass to vectorized calculations.
    #    We can do it directly or store in a small named tuple.
    # ----------------------------------------------------------------
    switchers_data = {
        "beta_vec": sub_beta_vec,
        "gamma_vec": sub_gamma_vec,
        "d_i_min_vec": sub_d_i_min_vec,
        "price_owns_car_vec": sub_price_owns_car_vec
        # add anything else needed by your vectorized methods
    }

    # ----------------------------------------------------------------
    # 2) Generate the list of "car options" = new_cars + second_hand_cars
    # ----------------------------------------------------------------
    car_options = self.new_cars.copy()
    if self.second_hand_cars:
        car_options += self.second_hand_cars

    # Let's define total_cars = len(car_options)
    # We'll build the dictionary-of-vehicle-attributes for these cars 
    # (the same approach you have in gen_vehicle_dict_vecs).
    vehicle_dict_vecs = self.gen_vehicle_dict_vecs(car_options)


    # ----------------------------------------------------------------
    # 3) Actually compute the (S x total_cars) utility & distance 
    #    for these switchers with these cars.
    #    We'll call a new or adapted vectorized method that 
    #    accepts the "sub" user arrays.
    # ----------------------------------------------------------------
    utilities_matrix, d_matrix = self.vectorised_calculate_utility_sub(
        switchers_data,
        vehicle_dict_vecs
    )
    # shape(utilities_matrix) = (S, total_cars)
    # shape(d_matrix)         = (S, total_cars)

    # ----------------------------------------------------------------
    # 4) If you want to append the "current cars" columns after new/used cars,
    #    you can do so. For example, if your model puts user i's current car 
    #    as an extra column. But that might require a separate function, 
    #    as in your original code. That’s up to you.
    # ----------------------------------------------------------------
    # For demonstration, let's say we skip that. 
    # Or if you do want it, you can do something like:
    #   util_current, d_current = self._generate_utilities_current_sub(switcher_indices)
    #   utilities_matrix = np.concatenate([utilities_matrix, util_current], axis=1)
    #   d_matrix = np.concatenate([d_matrix, d_current], axis=1)
    #   car_options += self.current_vehicles  # or however you track them

    # ----------------------------------------------------------------
    # 5) Return the final subset-based result
    # ----------------------------------------------------------------
    return utilities_matrix, car_options, d_matrix
