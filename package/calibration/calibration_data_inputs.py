import pandas as pd
from package.resources.utility import save_object

def load_in_calibration_data():

    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE) https://afdc.energy.gov/fuels/properties
    km_to_miles = 1.60934
    gasoline_g_co2_per_MJ = 92#FROM https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter10.pdf table 10.8 p1145
    gasoline_Kgco2_per_MJ = gasoline_g_co2_per_MJ/1000
    kWr_per_MJ = 0.2777777778
    gasoline_Kgco2_per_Kilowatt_Hour = gasoline_Kgco2_per_MJ/kWr_per_MJ
    
    #CPI
    CPI_california_df = pd.read_excel("package/calibration_data/CPI_california.xlsx") 
    CPI_california_df["Date"] = pd.to_datetime(CPI_california_df["Date"])
    CPI_california_df["Weighted Average"] = CPI_california_df["Weighted Average"].interpolate(method='linear')
    CPI_california_df.set_index('Date', inplace=True)
    reference_value = CPI_california_df.loc["2020-01-01", "Weighted Average"]
    CPI_california_df["CPI 2020 Real"] = CPI_california_df["Weighted Average"] / reference_value
     # Filter data to start from 2001
    CPI_california_df = CPI_california_df[CPI_california_df.index >= "2001-01-01"]

    #dec_2017_relative_value = CPI_california_df.loc["2017-12-01", "CPI 2020 Real"]
    #jan_2015_relative_value = CPI_california_df.loc["2015-01-01", "CPI 2020 Real"]
    #sept_2009_relative_value = CPI_california_df.loc["2009-09-01", "CPI 2020 Real"]
    #jan_2018_relative_value = CPI_california_df.loc["2018-01-01", "CPI 2020 Real"]

    #Gasoline Price
    gas_price_california_df = pd.read_excel("package/calibration_data/gas_price_california.xlsx") 
    gas_price_california_df["Date"] = pd.to_datetime(gas_price_california_df["Date"])
    gas_price_california_df.set_index('Date', inplace=True)
    gas_price_california_df["Real Dollars per Gallon"] = gas_price_california_df["Dollars per Gallon"]/CPI_california_df["CPI 2020 Real"]
    gas_price_california_df["Gas Real Dollars per Kilowatt-Hour"] = gas_price_california_df["Real Dollars per Gallon"]/gasoline_Kilowatt_Hour_per_gallon

    # Filter data to start from 2001
    gas_price_california_df = gas_price_california_df[gas_price_california_df.index >= "2001-01-01"]

    #ELECTRICITY
    electricity_price_df = pd.read_excel("package/calibration_data/residential_electricity_price.xlsx") 
    electricity_price_df["Date"] = pd.to_datetime(electricity_price_df["Date"])
    electricity_price_df.set_index('Date', inplace=True)

    # Filter data to start from 2001
    electricity_price_df = electricity_price_df[electricity_price_df.index >= "2001-01-01"]
    electricity_price_df["Electricty Real Dollars per Kilowatt-Hour"] = electricity_price_df["Dollars"]/CPI_california_df["CPI 2020 Real"]

    #Electricty
    electricity_emissions_intensity_df = pd.read_csv("package/calibration_data/emissions_intensity_nc_2000_2001_emberChartData.csv") 
    electricity_emissions_intensity_df["Date"] = pd.to_datetime(electricity_emissions_intensity_df["Date"])
    electricity_emissions_intensity_df.set_index('Date', inplace=True)
    electricity_emissions_intensity_df["Electricty KgCO2 per Kilowatt-Hour"] = electricity_emissions_intensity_df["emissions_intensity_gco2_per_kwh"]/1000

    # Filter data to start from 2001
    electricity_emissions_intensity_df = electricity_emissions_intensity_df[electricity_emissions_intensity_df.index >= "2001-01-01"]

    ###############################################################################
    #WE CALIBRATE EFFICIENCY IN TWO WAY BOTH FROM THE INPUT AND OUTPUT SIDE TO CROSS CHECK 

    efficiency_and_power_df = pd.read_excel("package/calibration_data/efficiency_and_power.xlsx")
    efficiency_and_power_df["Date"] = pd.to_datetime(efficiency_and_power_df["Date"])
    efficiency_and_power_df.set_index('Date', inplace=True)
    efficiency_and_power_df["km_per_kWhr"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*(km_to_miles/gasoline_Kilowatt_Hour_per_gallon)

    ##############################################################################################
    # Align Data
    aligned_data = pd.concat([
        CPI_california_df["CPI 2020 Real"],
        gas_price_california_df["Gas Real Dollars per Kilowatt-Hour"],
        electricity_price_df["Electricty Real Dollars per Kilowatt-Hour"],
        electricity_emissions_intensity_df["Electricty KgCO2 per Kilowatt-Hour"]
    ], axis=1)

    aligned_data.index = pd.to_datetime(aligned_data.index)
    aligned_data = aligned_data[aligned_data.index <= "2022-12-01"]

    #########################################################################
    #Get end averages
    Gas_price_2022 = gas_price_california_df.loc[gas_price_california_df.index.year == 2022, "Gas Real Dollars per Kilowatt-Hour"].mean()
    electricity_price_2022 = electricity_price_df.loc[electricity_price_df.index.year == 2022, "Electricty Real Dollars per Kilowatt-Hour"].mean()
    electricity_emissions_intensity_2022 = electricity_emissions_intensity_df.loc[electricity_emissions_intensity_df.index.year == 2022, "Electricty KgCO2 per Kilowatt-Hour"].mean()

    ################################################################################################################################################
    #LOAD IN BETA distribution data
    income_df = pd.read_excel("package/calibration_data/income_quintiles_2019_20.xlsx")
    
    ################################################################################################################################################
    #Save output as Excel
    aligned_data.to_excel("package/calibration_data/aligned_data.xlsx")  
    
    return aligned_data, gasoline_Kgco2_per_Kilowatt_Hour, Gas_price_2022 , electricity_price_2022, electricity_emissions_intensity_2022, income_df

def future_calibration_data():
    electricity_emissions_intensity_future_df = pd.read_excel(
        "package/calibration_data/GUBERT_2020_future_utilities_emissions.xlsx"
    )
    electricity_emissions_intensity_future_df["Year"] = pd.to_datetime(electricity_emissions_intensity_future_df["Year"])
    electricity_emissions_intensity_future_df.set_index("Year", inplace=True)

    electricity_emissions_intensity_future_df["Mean KgCO2 per MWh"] = electricity_emissions_intensity_future_df[
        ["SCE", "PG&E", "LADWP", "SDG&E", "SMUD", "Other utility"]
    ].mean(axis=1)
    electricity_emissions_intensity_future_df["Mean KgCO2 per kWh"] = electricity_emissions_intensity_future_df[
        "Mean KgCO2 per MWh"
    ] / 1000

    monthly_index = pd.date_range(
        start=electricity_emissions_intensity_future_df.index.min(),
        end=electricity_emissions_intensity_future_df.index.max(),
        freq="MS"
    )

    electricity_emissions_intensity_future_df = electricity_emissions_intensity_future_df.reindex(
        monthly_index, method="ffill"
    )

    return electricity_emissions_intensity_future_df["Mean KgCO2 per kWh"]

def data_range():
    efficiency_and_power_df = pd.read_excel("package/calibration_data/efficiency_and_power.xlsx")
    efficiency_and_power_df["Date"] = pd.to_datetime(efficiency_and_power_df["Date"])
    efficiency_and_power_df.set_index('Date', inplace=True)
    km_to_miles = 1.60934
    gasoline_Kilowatt_Hour_per_gallon = 33.41
    efficiency_and_power_df["km_per_kWhr"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*(km_to_miles/gasoline_Kilowatt_Hour_per_gallon)

    average_gas_tank_size = 16
    efficiency_and_power_df["Average Distance km"] = efficiency_and_power_df["Avg Fuel Economy mpg"]*average_gas_tank_size*km_to_miles
    #print("ICE VEHICLE RANGE",efficiency_and_power_df["Average Distance km"])

    EV_range_df = pd.read_excel("package/calibration_data/EVrange.xlsx")
    EV_range_df["Date"] = pd.to_datetime(EV_range_df["Date"])
    EV_range_df.set_index("Date", inplace=True)

    yearly_data = EV_range_df.join(efficiency_and_power_df["Average Distance km"], how="inner")

    #print("ICE VEHICLE RANGE", yearly_data["EV Range (km)"])

if __name__ == "__main__":
    calibration_data_input = {}

    calibration_data_output, gasoline_Kgco2_per_Kilowatt_Hour, Gas_price_2022 , electricity_price_2022, electricity_emissions_intensity_2022, income_df = load_in_calibration_data()

    scale_dollars = 0.00001
    scale_co2 = 0.00001

    calibration_data_input["gas_price_california_vec"] = calibration_data_output["Gas Real Dollars per Kilowatt-Hour"].to_numpy()*scale_dollars
    calibration_data_input["electricity_price_vec"] = calibration_data_output["Electricty Real Dollars per Kilowatt-Hour"].to_numpy()*scale_dollars
    calibration_data_input["electricity_emissions_intensity_vec"] = calibration_data_output["Electricty KgCO2 per Kilowatt-Hour"].to_numpy()*scale_co2
    calibration_data_input["Gas_price_2022"] = Gas_price_2022*scale_dollars
    calibration_data_input["Electricity_price_2022"] = electricity_price_2022*scale_dollars
    calibration_data_input["Electricity_emissions_intensity_2022"] = electricity_emissions_intensity_2022*scale_co2
    calibration_data_input["gasoline_Kgco2_per_Kilowatt_Hour"] = gasoline_Kgco2_per_Kilowatt_Hour*scale_co2
    calibration_data_input["income"] = income_df["Income"].to_numpy()
    
    calibration_data_input["scale_co2"] = scale_co2
    calibration_data_input["scale_dollars"] = scale_dollars
    #print("gas price 2022",calibration_data_input["Gas_price_2022"] )

    save_object( calibration_data_input, "package/calibration_data", "calibration_data_input")