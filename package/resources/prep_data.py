import pandas as pd

def load_in_calibration_data():

    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE)
    gasoline_gco2_per_gallon = 8887 #grams CO2/ gallon

    #CPI
    CPI_california_df = pd.read_excel("package/calibration_data/CPI_california.xlsx") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    CPI_california_df["Date"] = pd.to_datetime(CPI_california_df["Date"])
    CPI_california_df["Weighted Average"] = CPI_california_df["Weighted Average"].interpolate(method='linear')
    # Handle the first NaN explicitly by assigning the second value
    if pd.isna(CPI_california_df.loc[0, "Weighted Average"]):
        CPI_california_df.loc[0, "Weighted Average"] = CPI_california_df.loc[1, "Weighted Average"]
    # Set "Date" as the index
    CPI_california_df.set_index('Date', inplace=True)
    # Get the value for "Weighted Average" on 2020-01-01
    reference_value = CPI_california_df.loc["2020-01-01", "Weighted Average"]
    # Normalize by the 2020 reference value
    CPI_california_df["2020 relative Weighted Average"] = CPI_california_df["Weighted Average"] / reference_value

    #Gasoline Price
    gas_price_california_df = pd.read_excel("package/calibration_data/gas_price_california.xlsx") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    gas_price_california_df["Date"] = pd.to_datetime(gas_price_california_df["Date"])
    gas_price_california_df.set_index('Date', inplace=True)
    gas_price_california_df["Real Dollars per Gallon"] = gas_price_california_df["Dollars per Gallon"]/CPI_california_df["2020 relative Weighted Average"]
    gas_price_california_df["Real Dollars per Kilowatt-Hour"] = gas_price_california_df["Real Dollars per Gallon"]/gasoline_Kilowatt_Hour_per_gallon

    #Electricity Price
    electricity_price_df = pd.read_excel("package/calibration_data/electricity_price.xlsx") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    electricity_price_df["Date"] = pd.to_datetime( electricity_price_df["Date"])
    electricity_price_df.set_index('Date', inplace=True)
    electricity_price_df["Dollars per Kilowatt-Hour (City Average)"] = electricity_price_df[["Dollars per Kilowatt-Hour (San Francisco)", "Dollars per Kilowatt-Hour (Los Angeles)"]].mean(axis=1)
    electricity_price_df["Real Dollars per Kilowatt-Hour (City Average)"] = electricity_price_df["Dollars per Kilowatt-Hour (City Average)"]/CPI_california_df["2020 relative Weighted Average"]

    #Electricty
    electricity_emissions_intensity_df = pd.read_csv("package/calibration_data/emissions_intensity_nc_2000_2001_emberChartData.csv") 
    # Ensure the "Date" column is in datetime format (optional, for time-based operations)
    #gco2_per_kwh
    electricity_emissions_intensity_df["Date"] = pd.to_datetime( electricity_emissions_intensity_df["Date"])
    electricity_emissions_intensity_df.set_index('Date', inplace=True)
    electricity_emissions_intensity_df["KgCO2 per Kilowatt-Hour"] = electricity_emissions_intensity_df["emissions_intensity_gco2_per_kwh"]/1000

    #NOW - NEED TO GET ALL THE PRICES INTO 2020 DOLLARS BY DIVIDING BY THE CPI 

    #Emissions Gasoline - WE 
    gasoline_Kgco2_per_Kilowatt_Hour =  (gasoline_gco2_per_gallon/gasoline_Kilowatt_Hour_per_gallon)/1000
    #0.26599820413049985
    #print(gasoline_Kgco2_per_Kilowatt_Hour )
    #quit()

    # Align Data
    aligned_data = CPI_california_df.join([
        gas_price_california_df["Real Dollars per Kilowatt-Hour"],
        electricity_price_df["Real Dollars per Kilowatt-Hour (City Average)"],
        electricity_emissions_intensity_df["KgCO2 per Kilowatt-Hour"]
    ], how='inner')  # Only keep rows with data in all columns

    return aligned_data, gasoline_Kgco2_per_Kilowatt_Hour

def load_in_output_data():
    #"""
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

    # Calculate EV percentage
    merged_data['EV Percentage'] = (merged_data['EV Vehicles'] / merged_data['Total Vehicles']) * 100

    #######################################

    #NOW DO EV SALES DATA
    #EV SALES
    New_ZEV_sales_df = pd.read_excel("package/calibration_data/New_ZEV_Sales.xlsx")
    # Grouping and summing
    New_ZEV_sales_Grouped_df = New_ZEV_sales_df.groupby(['Data_Year'], as_index=False)['Number of Vehicles'].sum()

    #"""
    ########################################

    #What are the historical ranges of Cars in terms of efficiency (km/kWhr) us this to parameterise the omega limits on the landscape
    efficiency_and_power_df = pd.read_excel("package/calibration_data/efficiency_and_power.xlsx")
    # Convert mpg to km/kWhr
    km_to_miles = 1.60934
    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE)
    efficiency_and_power_df["km_per_kWhr"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*(km_to_miles/gasoline_Kilowatt_Hour_per_gallon)
    #"min_max_Efficiency":[0.5,1.5], historial min and max for period are (0.953754,1.252405)

    #########################################

    #Find minimum and maximmum values for EV fuel efficiency 
    """
    Searching for US models in perdio 2006-2011 : https://www.fueleconomy.gov/feg/PowerSearch.do?action=Cars&vtype=Electric&srchtyp=evSelect&rowLimit=50&sortBy=Comb&year1=2006&year2=2011&mclass=Small%20Cars,Coupes,Sports/Sporty%20Cars,Hatchbacks,Family%20Sedans,Upscale%20Sedans,Luxury%20Sedans,Sport%20Utility%20Vehicles&range=&drive=
    Found Several models: 2011 smart fortwo electric drive cabriolet (39 kWh/100 mi), 2008 MINI MiniE (34 kWh/100 mi), 2011 Nissan Leaf (34 kWh/100 mi), 2011 BMW Active E (33 kWh/100 mi)
    
    Take the value of the Nissan Leaf of 34 kWh/100 mi or quoted 110.2 MPG
    """
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

    ###############################################################################

    #What is the distance travelled by your typical car?
    average_gas_tank_size = 16#Gallons https://mechanicbase.com/cars/average-gas-tank-size/  https://millsequipment.com/blogs/blogs/understanding-average-fuel-tank-size-what-you-need-to-know
    efficiency_and_power_df["Average Distance km"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*average_gas_tank_size*km_to_miles
    #print(efficiency_and_power_df["Average Distance km"])
    #MIN in 2000 is 509.838912km, MAX in 2022 its 669.485440km
    
    #Now compare to the evolving range in EVs
    EV_range_df = pd.read_excel("package/calibration_data/EVrange.xlsx")# https://www.iea.org/data-and-statistics/charts/evolution-of-average-range-of-electric-vehicles-by-powertrain-2010-2021
    #print(EV_range_df)
    #Min in 2010 is 127km, MAX in 2021 is 349km
    #Quality limits should be calibrated correspondingly to the ratios ICE: [450,700], EV: [100,400]

    #quit()
