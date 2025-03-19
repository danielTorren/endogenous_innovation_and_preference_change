import pandas as pd
from package.resources.utility import save_object

def load_in_output_data():

    #DEAL WITH EV POPULTATION, TOTAL VEHICLES AND PENERTRATION
    #POPULATION
    Vehicle_Population_df = pd.read_excel("package/calibration_data/Vehicle_Population.xlsx") 

    # Grouping and summing
    Vehicle_Population_Grouped_df = Vehicle_Population_df.groupby(['Data Year', 'Dashboard Fuel Type Group'], as_index=False)['Number of Vehicles'].sum()

    #EV_list = [ 'Battery Electric (BEV)', 'Plug-in Hybrid (PHEV)', 'Fuel Cell (FCEV)']
    #ICE_list = ['Diesel', 'Gasoline', 'Gasoline Hybrid', 'Other']

    EV_list = [ 'Battery Electric (BEV)']
    ICE_list = ['Diesel', 'Gasoline', 'Gasoline Hybrid', 'Other', 'Plug-in Hybrid (PHEV)', 'Fuel Cell (FCEV)']

    # Create a new column for categories
    Vehicle_Population_Grouped_df['Category'] = Vehicle_Population_Grouped_df['Dashboard Fuel Type Group'].apply(lambda x: 'EV' if x in EV_list else 'ICE' if x in ICE_list else 'Other')

    # Group by Year and Category and sum the number of vehicles
    Vehicle_Population_grouped_result_df = Vehicle_Population_Grouped_df.groupby(['Data Year', 'Category'], as_index=False)['Number of Vehicles'].sum()

    # Calculate total vehicles per year
    total_vehicles_per_year = Vehicle_Population_grouped_result_df.groupby('Data Year', as_index=False)['Number of Vehicles'].sum()
    total_vehicles_per_year.rename(columns={'Number of Vehicles': 'Total Vehicles'}, inplace=True)

    # Extract EV data
    ev_data = Vehicle_Population_grouped_result_df[Vehicle_Population_grouped_result_df['Category'] == 'EV'].rename(columns={'Number of Vehicles': 'EV Vehicles'})

    # Merge EV data with total vehicles
    merged_data = pd.merge(ev_data, total_vehicles_per_year, on='Data Year', how='left')

    # Calculate EV prop
    merged_data['EV Prop'] = (merged_data['EV Vehicles'] / merged_data['Total Vehicles'])

    #UP TO 2022 TO MATCH THE PRICING DATA!
    merged_data_filtered = merged_data[(merged_data['Data Year'] >= 2010) & (merged_data['Data Year'] <= 2023)]

    print("merged_data_filtered", merged_data_filtered)

    return merged_data_filtered['EV Prop'].to_numpy()

if __name__ == "__main__":
    EV_Prop = load_in_output_data()
    calibration_data_output = {}

    calibration_data_output["EV Prop"] = EV_Prop
    
    save_object(calibration_data_output, "package/calibration_data", "calibration_data_output")