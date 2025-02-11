import pandas as pd
from package.resources.utility import save_object

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
    # Extend data to include January to April 2000
    # Create a date range from January to April 2000
    new_dates = pd.date_range(start="2000-01-01", end="2000-04-01", freq='MS')
    # Get the data from May 2000
    may_2000_data = gas_price_california_df.loc["2000-05-01"]
    # Create a new DataFrame for the missing months
    new_rows = pd.DataFrame([may_2000_data.values] * len(new_dates), 
                            index=new_dates, 
                            columns=gas_price_california_df.columns)
    # Append the new rows to the original DataFrame
    gas_price_california_df = pd.concat([new_rows, gas_price_california_df]).sort_index()

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

    ###############################################################################

    #What are the historical ranges of Cars in terms of efficiency (km/kWhr) us this to parameterise the omega limits on the landscape
    efficiency_and_power_df = pd.read_excel("package/calibration_data/efficiency_and_power.xlsx")
    efficiency_and_power_df["Date"] = pd.to_datetime( efficiency_and_power_df["Date"])
    efficiency_and_power_df.set_index('Date', inplace=True)
    # Convert mpg to km/kWhr
    km_to_miles = 1.60934
    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE)
    efficiency_and_power_df["km_per_kWhr"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*(km_to_miles/gasoline_Kilowatt_Hour_per_gallon)
    #"min_max_Efficiency":[0.5,1.5], historial min and max for period are (0.953754,1.252405)

    #What is the distance travelled by your typical car?
    average_gas_tank_size = 16#Gallons https://mechanicbase.com/cars/average-gas-tank-size/  https://millsequipment.com/blogs/blogs/understanding-average-fuel-tank-size-what-you-need-to-know
    efficiency_and_power_df["Average Distance km"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*average_gas_tank_size*km_to_miles
    #print(efficiency_and_power_df)
    #MIN in 2000 is 509.838912km, MAX in 2022 its 669.485440km
    
    #Now compare to the evolving range in EVs
    # Load EV range data
    EV_range_df = pd.read_excel("package/calibration_data/EVrange.xlsx")
    EV_range_df["Date"] = pd.to_datetime(EV_range_df["Date"])
    EV_range_df.set_index("Date", inplace=True)

    # Align the indices to yearly for calculation
    yearly_data = EV_range_df.join(efficiency_and_power_df["Average Distance km"], how="inner")

    # Calculate the year-wise range ratio
    yearly_data["Range Ratio (ICE to EV)"] = yearly_data["Average Distance km"] / yearly_data["EV Range (km)"]

    # Extend the yearly data to monthly frequency
    monthly_index = pd.date_range(
        start=yearly_data.index.min(),
        end="2022-12-01",  # Extend up to December 2022
        freq="MS"  # Month Start frequency
    )

    # Reindex to monthly and forward-fill yearly data to monthly
    EV_range_monthly_data = yearly_data.reindex(monthly_index, method="ffill")

    # If needed, set the index name back to "Date"
    EV_range_monthly_data.index.name = "Date"
    ##############################################################################################    ##############################################################################################

    #NOW - NEED TO GET ALL THE PRICES INTO 2020 DOLLARS BY DIVIDING BY THE CPI 

    #Emissions Gasoline - WE 
    gasoline_Kgco2_per_Kilowatt_Hour =  (gasoline_gco2_per_gallon/gasoline_Kilowatt_Hour_per_gallon)/1000
    #0.26599820413049985
    
    #print("gasoline_Kgco2_per_Kilowatt_Hour", gasoline_Kgco2_per_Kilowatt_Hour)
    #ALTERNATIVE DATA FROM USA: US energy information adminitstration: https://www.eia.gov/environment/emissions/co2_vol_mass.php
    #1 Btu = 0.000293071 KWh
    #millionBTU = 293.071 KWh
    #Mototr gasoline  = 76.1104903685078 kgCO2/millionBTU
    #Motot gasoline  = 76.1104903685078/293.071 kgCO2/KWh = 0.260 kgCO2/KWh WHICH IS BASICALLY IDENTICAL, IF not lower than what i had earlier
    #

    #AND AGAIN, this time EPA: https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator-calculations-and-references#:~:text=filled%20with%20gasoline-,The%20amount%20of%20carbon%20dioxide%20emitted%20per%20gallon%20of%20motor,A%20barrel%20equals%2042%20gallons.
    #gallon of motor gasoline burned is 8.89 × 10-3 metric tons
    

    ##############################################################################################    ##############################################################################################
    
    # Align Data
    # Perform the join
    aligned_data = CPI_california_df.join(
        [
            gas_price_california_df["Real Dollars per Kilowatt-Hour"],
            electricity_price_df["Real Dollars per Kilowatt-Hour (City Average)"],
            electricity_emissions_intensity_df["KgCO2 per Kilowatt-Hour"]
        ],
        how='inner'  # Only keep rows with data in all columns
    )

    # Ensure the index is a datetime type if it's not already
    aligned_data.index = pd.to_datetime(aligned_data.index)

    # Filter the data up to 01/12/2022
    aligned_data = aligned_data[aligned_data.index <= "2022-12-01"]

    #########################################################################
    #Get end averages

    Gas_price_2022 = gas_price_california_df.loc[gas_price_california_df.index.year == 2022, "Real Dollars per Kilowatt-Hour"].mean()
    electricity_price_2022 = electricity_price_df.loc[electricity_price_df.index.year == 2022, "Real Dollars per Kilowatt-Hour (City Average)"].mean()
    electricity_emissions_intensity_2022 = electricity_emissions_intensity_df.loc[electricity_emissions_intensity_df.index.year == 2022, "KgCO2 per Kilowatt-Hour"].mean()

    ################################################################################################################################################
    #LOAD IN BETA distribution data - using the 2020 (inflation adjusted) distribution of percentiles, from census data - https://data.census.gov/table/ACSST5Y2020.S1901?q=california%20income
    #DATA FROM: https://www.bls.gov/cex/tables/geographic/mean/cu-state-ca-income-quintiles-before-taxes-2-year-average-2020.htm
    #income_df = pd.read_excel("package/calibration_data/income.xlsx")
    income_df = pd.read_excel("package/calibration_data/income_quintiles_2019_20.xlsx")
    
    ################################################################################################################################################
    return aligned_data, gasoline_Kgco2_per_Kilowatt_Hour, EV_range_monthly_data["Range Ratio (ICE to EV)"], Gas_price_2022 , electricity_price_2022, electricity_emissions_intensity_2022, income_df

def future_calibration_data():
    # Load data
    electricity_emissions_intensity_future_df = pd.read_excel(
        "package/calibration_data/GUBERT_2020_future_utilities_emissions.xlsx"
    )
    electricity_emissions_intensity_future_df["Year"] = pd.to_datetime(electricity_emissions_intensity_future_df["Year"])
    electricity_emissions_intensity_future_df.set_index("Year", inplace=True)

    # Calculate means
    electricity_emissions_intensity_future_df["Mean KgCO2 per MWh"] = electricity_emissions_intensity_future_df[
        ["SCE", "PG&E", "LADWP", "SDG&E", "SMUD", "Other utility"]
    ].mean(axis=1)
    electricity_emissions_intensity_future_df["Mean KgCO2 per kWh"] = electricity_emissions_intensity_future_df[
        "Mean KgCO2 per MWh"
    ] / 1000

    # Create a new date range with monthly frequency
    monthly_index = pd.date_range(
        start=electricity_emissions_intensity_future_df.index.min(),
        end=electricity_emissions_intensity_future_df.index.max(),
        freq="MS"  # Month Start frequency
    )

    # Reindex to monthly, filling forward the yearly data for each month
    electricity_emissions_intensity_future_df = electricity_emissions_intensity_future_df.reindex(
        monthly_index, method="ffill"
    )

    # Return the relevant column
    return electricity_emissions_intensity_future_df["Mean KgCO2 per kWh"]

if __name__ == "__main__":

    calibration_data_input = {}

    calibration_data_output, gasoline_Kgco2_per_Kilowatt_Hour, EV_range_ratio, Gas_price_2022 , electricity_price_2022, electricity_emissions_intensity_2022, income_df = load_in_calibration_data()
    
    calibration_data_input["gas_price_california_vec"] = calibration_data_output["Real Dollars per Kilowatt-Hour"].to_numpy()
    calibration_data_input["electricity_price_vec"] = calibration_data_output["Real Dollars per Kilowatt-Hour (City Average)"].to_numpy()
    calibration_data_input["electricity_emissions_intensity_vec"] = calibration_data_output["KgCO2 per Kilowatt-Hour"].to_numpy()
    calibration_data_input["tank_ratio_vec"] = EV_range_ratio.to_numpy()
    calibration_data_input["Gas_price_2022"] = Gas_price_2022
    calibration_data_input["Electricity_price_2022"] = electricity_price_2022
    calibration_data_input["Electricity_emissions_intensity_2022"] = electricity_emissions_intensity_2022
    calibration_data_input["gasoline_Kgco2_per_Kilowatt_Hour"] = gasoline_Kgco2_per_Kilowatt_Hour

    calibration_data_input["income"] = income_df["Income"].to_numpy()
    

    save_object( calibration_data_input, "package/calibration_data", "calibration_data_input")