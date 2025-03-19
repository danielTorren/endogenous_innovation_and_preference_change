import pandas as pd
from package.resources.utility import save_object

def load_in_calibration_data():

    gasoline_Kilowatt_Hour_per_gallon = 33.41 #Gasoline gallon equivalent (GGE) https://afdc.energy.gov/fuels/properties
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
    electricity_emissions_intensity_df = pd.read_excel("package/calibration_data/emissions_intensity.xlsx") 
    electricity_emissions_intensity_df["Date"] = pd.to_datetime(electricity_emissions_intensity_df["Date"])
    electricity_emissions_intensity_df.set_index('Date', inplace=True)
    electricity_emissions_intensity_df["Electricty KgCO2 per Kilowatt-Hour"] = electricity_emissions_intensity_df["emissions_intensity_gco2_per_kwh"]/1000

    # Filter data to start from 2001
    electricity_emissions_intensity_df = electricity_emissions_intensity_df[electricity_emissions_intensity_df.index >= "2001-01-01"]

    ##############################################################################################
    # Align Data
    aligned_data = pd.concat([
        CPI_california_df["CPI 2020 Real"],
        gas_price_california_df["Gas Real Dollars per Kilowatt-Hour"],
        electricity_price_df["Electricty Real Dollars per Kilowatt-Hour"],
        electricity_emissions_intensity_df["Electricty KgCO2 per Kilowatt-Hour"]
    ], axis=1)

    aligned_data.index = pd.to_datetime(aligned_data.index)
    aligned_data = aligned_data[aligned_data.index <= "2023-12-01"]

    #########################################################################
    #Get end averages
    Gas_price_2023 = gas_price_california_df.loc[gas_price_california_df.index.year == 2023, "Gas Real Dollars per Kilowatt-Hour"].mean()
    electricity_price_2023 = electricity_price_df.loc[electricity_price_df.index.year == 2023, "Electricty Real Dollars per Kilowatt-Hour"].mean()
    electricity_emissions_intensity_2023 = electricity_emissions_intensity_df.loc[electricity_emissions_intensity_df.index.year == 2023, "Electricty KgCO2 per Kilowatt-Hour"].mean()

    ################################################################################################################################################
    #Save output as Excel
    aligned_data.to_excel("package/calibration_data/aligned_data.xlsx")  

    print(aligned_data)
    
    return aligned_data, gasoline_Kgco2_per_Kilowatt_Hour, Gas_price_2023 , electricity_price_2023, electricity_emissions_intensity_2023

if __name__ == "__main__":
    calibration_data_input = {}

    calibration_data_output, gasoline_Kgco2_per_Kilowatt_Hour, Gas_price_2023 , electricity_price_2023, electricity_emissions_intensity_2023= load_in_calibration_data()

    calibration_data_input["gas_price_california_vec"] = calibration_data_output["Gas Real Dollars per Kilowatt-Hour"].to_numpy()
    calibration_data_input["electricity_price_vec"] = calibration_data_output["Electricty Real Dollars per Kilowatt-Hour"].to_numpy()
    calibration_data_input["electricity_emissions_intensity_vec"] = calibration_data_output["Electricty KgCO2 per Kilowatt-Hour"].to_numpy()
    calibration_data_input["Gas_price_2023"] = Gas_price_2023
    calibration_data_input["Electricity_price_2023"] = electricity_price_2023
    calibration_data_input["Electricity_emissions_intensity_2023"] = electricity_emissions_intensity_2023
    calibration_data_input["gasoline_Kgco2_per_Kilowatt_Hour"] = gasoline_Kgco2_per_Kilowatt_Hour

    save_object( calibration_data_input, "package/calibration_data", "calibration_data_input")