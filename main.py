
from tools import *
from emission_tools import *
from proxy_tools import *
from mass_balance_tools import calculate_mass_balance
from osgeo import gdal
gdal.UseExceptions()

def main():
    ## Read Configuration file
    data_parameters, job_parameters = config_parameters()

    ## Define sectors
    sectors = [['A_PublicPower', '1A1a'],
            ['B_Industry', '1A1b','1A1c','1A2a','1A2b','1A2c','1A2d','1A2e','1A2f','1A2gviii','2A1','2A2','2A3','2A5a','2A5b','2A5c','2A6','2B1','2B2','2B3','2B5','2B6','2B7','2B10a','2B10b','2C1','2C2','2C3','2C4','2C5','2C6','2C7a','2C7b','2C7c','2C7d','2D3b','2D3c','2H1','2H2','2H3','2I','2J','2KV','2L'],
            ['C_OtherStationaryComb','1A4ai','1A4bi','1A4ci','1A5a'],
            ['D_Fugitives', '1B1a','1B1b','1B1c','1B2ai','1B2aiv','1B2av','1B2b','1B2c','1B2d'],
            ['E_Solvents', '2D3a','2D3d','2D3e','2D3f','2D3g','2D3h','2D3i','2G'],
            ['F_RoadTransport', '1A3bi','1A3bii','1A3biii','1A3biv','1A3bv','1A3bvi','1A3bvii'],
            ['G_Shipping', '1A3di_ii','1A3dii'],
            ['H_Aviation', '1A3ai_i','1A3aii_i'],
            ['I_OffRoad', '1A2gvii','1A3c','1A3ei','1A3eii','1A4aii','1A4bii','1A4cii','1A4ciii','1A5b'],
            ['J_Waste', '5B1','5B2','5C1a','5C1bi','5C1bii','5C1biii','5C1biv','5C1bv','5C1bvi','5C2','5D1','5D2','5D3','5E'],
            ['K_AgriLivestock', '3B1a','3B1b','3B2','3B3','3B4a','3B4d','3B4e','3B4f','3B4gi','3B4gii','3B4giii','3B4giv','3B4h'],
            ['L_AgriOther','3Da1','3Da2a','3Da2b','3Da2c','3Da3','3Da4','3Db','3Dc','3Dd','3De','3Df','3F','3I'],
            ['SumAllSectors', 'SUM']]

    if job_parameters['inventory'] == 'GRETA':
        gdf_grid, bbox_grid, bbox_epsg, hourly_trf, weekly_trf, weekend_types, daytype_mapping = prep_greta_data(data_parameters, job_parameters, sectors)
    elif job_parameters['inventory'] == 'TNO':
        print('Prep TNO Emissions')
    
    ## Create proxies and downscale
    downscaling_proxies(data_parameters, job_parameters, bbox_grid, bbox_epsg)
    downscale_emissions(job_parameters, sectors, gdf_grid, bbox_grid, bbox_epsg, hourly_trf, weekly_trf, weekend_types, daytype_mapping)

    ## Apply mass balance correction
    calculate_mass_balance(job_parameters, data_parameters, sectors, job_parameters['species'])

if __name__ == "__main__":
    main()