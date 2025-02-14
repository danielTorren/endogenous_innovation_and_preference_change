import pandas as pd
from package.resources.utility import save_object

def load_in_output_data():

    #DEAL WITH EV POPULTATION, TOTAL VEHICLES AND PENERTRATION
    #POPULATION
    Vehicle_Population_df = pd.read_excel("package/calibration_data/Vehicle_Population.xlsx") 

    # Grouping and summing
    Vehicle_Population_Grouped_df = Vehicle_Population_df.groupby(['Data Year', 'Dashboard Fuel Type Group'], as_index=False)['Number of Vehicles'].sum()

    EV_list = [ 'Battery Electric (BEV)', 'Plug-in Hybrid (PHEV)', 'Fuel Cell (FCEV)']
    ICE_list = ['Diesel', 'Gasoline', 'Gasoline Hybrid', 'Other']

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
    merged_data_filtered = merged_data[(merged_data['Data Year'] >= 2010) & (merged_data['Data Year'] <= 2022)]

    print("merged_data_filtered", merged_data_filtered)
    #############################################################

    #NOW DO EV SALES DATA
    #EV SALES
    New_ZEV_sales_df = pd.read_excel("package/calibration_data/New_ZEV_Sales.xlsx")
    # Grouping and summing
    New_ZEV_sales_Grouped_df = New_ZEV_sales_df.groupby(['Data_Year'], as_index=False)['Number of Vehicles'].sum()


    #########################################

    #Find minimum and maximmum values for EV fuel efficiency 
    """
    Searching for US models in perdio 2006-2011 : https://www.fueleconomy.gov/feg/PowerSearch.do?action=Cars&vtype=Electric&srchtyp=evSelect&rowLimit=50&sortBy=Comb&year1=2006&year2=2011&mclass=Small%20Cars,Coupes,Sports/Sporty%20Cars,Hatchbacks,Family%20Sedans,Upscale%20Sedans,Luxury%20Sedans,Sport%20Utility%20Vehicles&range=&drive=
    Found Several models: 2011 smart fortwo electric drive cabriolet (39 kWh/100 mi), 2008 MINI MiniE (34 kWh/100 mi), 2011 Nissan Leaf (34 kWh/100 mi), 2011 BMW Active E (33 kWh/100 mi)
    
    Take the value of the Nissan Leaf of 34 kWh/100 mi or quoted 110.2 MPG
    """
    km_to_miles = 1.60934
    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE)
    #MIN VALUE FIRST
    
    nissan_leaf_kWh_per_100_miles = 34
    nissan_leaf_kWhr_per_km = nissan_leaf_kWh_per_100_miles/(100*km_to_miles)
    nissan_leaf_km_per_kWhr = 1/nissan_leaf_kWhr_per_km
    #4.73335294117647

    #Alternatively use the MPGe
    nissan_leaf_mpg = 110.2
    
    nissan_leaf_km_per_kWhr__using_mpg = nissan_leaf_mpg*(km_to_miles/gasoline_Kilowatt_Hour_per_gallon)
    #5.308269021251123


    #MAX VALUE NEXT
    """
    Using the most recent models of 2022-23 as this matches the rest of our dataset: https://www.fueleconomy.gov/feg/PowerSearch.do?action=Cars&vtype=Electric&srchtyp=evSelect&rowLimit=50&sortBy=Comb&year1=2022&year2=2023&mclass=Small%20Cars,Coupes,Sports/Sporty%20Cars,Hatchbacks,Family%20Sedans,Upscale%20Sedans,Luxury%20Sedans,Sport%20Utility%20Vehicles&range=&drive=
    
    Take the value of the 2022 Tesla Model 3 RWD Automatic (A1): Combined MPGe 132, 25 kWh/100 mi
    """
    tesla_model_3_kWh_per_100_miles = 25
    tesla_model_3_kWhr_per_km = tesla_model_3_kWh_per_100_miles/(100*km_to_miles)
    tesla_model_3_km_per_kWhr = 1/tesla_model_3_kWhr_per_km
    #print(tesla_model_3_km_per_kWhr)
    #6.43736

    #Alternatively use the MPGe
    tesla_model_3_mpg = 132
    tesla_model_3_km_per_kWhr__using_mpg = tesla_model_3_mpg*(km_to_miles/gasoline_Kilowatt_Hour_per_gallon)
    #print(tesla_model_3_km_per_kWhr__using_mpg)
    #6.358362167015864

    #USE THESE VALUES OF THE MIN AND THE MAX TO PARAMETERISE THE LANSCAPE
    #"min_max_Efficiency":[4,7], historial min and max for 2006-2022 period are (4.73335294117647,6.43736)
    
    ##############################################################################################################################
    #EMISSIONS INTENSITY OF FLEET
    emissions_intensity_cars_df = pd.read_excel("package/calibration_data/emissions_intensity_cars.xlsx") 
    emisisons_intensity_cars_data = emissions_intensity_cars_df["gCO2e per mile"].to_numpy()
    kg_CO2_per_km_vec = emisisons_intensity_cars_data/(km_to_miles*1000)

    ##################################################################################################
    MMTCO2e_df = pd.read_excel("package/calibration_data/emissions_passenger_vehicle_2000_21.xlsx") 
    MMTCO2e_data = MMTCO2e_df["MMTCO2e"].to_numpy()
    CO2_index = MMTCO2e_data/max(MMTCO2e_data)

    return merged_data_filtered['EV Prop'].to_numpy(), kg_CO2_per_km_vec, CO2_index

if __name__ == "__main__":
    EV_Prop, kg_CO2_per_km, CO2_index = load_in_output_data()
    calibration_data_output = {}

    calibration_data_output["EV Prop"] = EV_Prop
    calibration_data_output["kg_CO2_per_km"] = kg_CO2_per_km
    calibration_data_output["CO2_index"] = CO2_index
    save_object(calibration_data_output, "package/calibration_data", "calibration_data_output")