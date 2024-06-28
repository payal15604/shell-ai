import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load data
demand = pd.read_csv('Demand.csv')
vehicles = pd.read_csv('Vehicles.csv')
vehicles_fuels = pd.read_csv('Vehicles_fuels.csv')
fuels = pd.read_csv('Fuels.csv')
carbon_emissions = pd.read_csv('Carbon_emissions.csv')

# Decision variables initialization
years = range(2023, 2039)
vehicle_ids = vehicles['ID'].unique()
num_vehicles = {year: {vid: 0 for vid in vehicle_ids} for year in years}
distance_traveled = {year: {vid: 0 for vid in vehicle_ids} for year in years}

# Define objective function and constraints
def objective_function(num_vehicles, distance_traveled):
    total_cost = 0
    for year in years:
        for vid in vehicle_ids:
            # Add purchase cost
            total_cost += vehicles.loc[vehicles['ID'] == vid, 'Cost'].values[0] * num_vehicles[year][vid]
            # Add insurance cost
            total_cost += 0.05 * vehicles.loc[vehicles['ID'] == vid, 'Cost'].values[0] * num_vehicles[year][vid]
            # Add maintenance cost
            total_cost += 0.01 * vehicles.loc[vehicles['ID'] == vid, 'Cost'].values[0] * num_vehicles[year][vid]
            # Add fuel cost
            fuel_type = vehicles_fuels.loc[vehicles_fuels['ID'] == vid, 'Fuel'].values[0]
            fuel_cost_per_unit = fuels.loc[fuels['Fuel'] == fuel_type, 'Cost'].values[0]
            fuel_consumption_per_km = vehicles_fuels.loc[vehicles_fuels['ID'] == vid, 'Fuel Consumption (unit_fuel/km)'].values[0]
            total_cost += fuel_cost_per_unit * fuel_consumption_per_km * distance_traveled[year][vid]
            # Subtract resale value
            resale_value = 0.9 * vehicles.loc[vehicles['ID'] == vid, 'Cost'].values[0] * num_vehicles[year][vid]
            total_cost -= resale_value
    return total_cost

def constraints(num_vehicles, distance_traveled):
    constraints = []
    for year in years:
        # Demand constraints
        for size in ['S1', 'S2', 'S3', 'S4']:
            for distance in ['D1', 'D2', 'D3', 'D4']:
                demand_value = demand.loc[(demand['Year'] == year) & (demand['Size'] == size) & (demand['Distance'] == distance), 'Demand'].values[0]
                total_distance_covered = sum(distance_traveled[year][vid] for vid in vehicle_ids if vehicles.loc[vehicles['ID'] == vid, 'Size'].values[0] == size)
                constraints.append(total_distance_covered >= demand_value)
        # Carbon emission constraints
        total_emissions = 0
        for vid in vehicle_ids:
            fuel_type = vehicles_fuels.loc[vehicles_fuels['ID'] == vid, 'Fuel'].values[0]
            emission_per_unit = fuels.loc[fuels['Fuel'] == fuel_type, 'Emissions (CO2/unit_fuel)'].values[0]
            fuel_consumption_per_km = vehicles_fuels.loc[vehicles_fuels['ID'] == vid, 'Fuel Consumption (unit_fuel/km)'].values[0]
            total_emissions += emission_per_unit * fuel_consumption_per_km * distance_traveled[year][vid]
        constraints.append(total_emissions <= carbon_emissions.loc[carbon_emissions['Year'] == year, 'Total Carbon emission limit'].values[0])
    return constraints

# Optimization
result = linprog(c=objective_function(num_vehicles, distance_traveled), A_ub=constraints(num_vehicles, distance_traveled))

# Extract solution
solution = pd.DataFrame(columns=['Year', 'ID', 'Num_Vehicles', 'Type', 'Fuel', 'Distance_bucket', 'Distance_per_vehicle(km)'])
for year in years:
    for vid in vehicle_ids:
        if num_vehicles[year][vid] > 0:
            solution = solution.append({
                'Year': year,
                'ID': vid,
                'Num_Vehicles': num_vehicles[year][vid],
                'Type': 'Use' if distance_traveled[year][vid] > 0 else 'Buy',
                'Fuel': vehicles_fuels.loc[vehicles_fuels['ID'] == vid, 'Fuel'].values[0],
                'Distance_bucket': vehicles.loc[vehicles['ID'] == vid, 'Distance'].values[0],
                'Distance_per_vehicle(km)': distance_traveled[year][vid]
            }, ignore_index=True)

# Save solution
solution.to_csv('solution.csv', index=False)
