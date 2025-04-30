import os
import numpy as np
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from patchify import patchify, unpatchify
from geocube.api.core import make_geocube
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
import fiona
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import gc

gdal.UseExceptions()

def bbox_transform(in_crs, out_crs, cell_minx, cell_miny, cell_maxx, cell_maxy, order_xy=True):
    """Enhanced bbox transformation with error handling"""
    try:
        transformer = Transformer.from_crs(out_crs, in_crs, always_xy=order_xy)
        xmin, ymin = transformer.transform(cell_minx, cell_miny)
        xmax, ymax = transformer.transform(cell_maxx, cell_maxy)
        return [xmin, ymin, xmax, ymax]
    except Exception as e:
        raise ValueError(f"CRS transformation failed: {str(e)}")

def nfr_to_gnfr(job_parameters, gdf_data, sectors):
    """Convert NFR to GNFR sectors with improved column handling"""
    coln = list(gdf_data.columns)
    src_type = [item[0] for item in sectors]
    col_sum = []
    
    for spec in job_parameters['species']:
        for sec in src_type:
            sidx = src_type.index(sec)
            subsec = sectors[sidx][1:]
            gfd = [f'E_{n}_{spec.upper()}' for n in subsec if f'E_{n}_{spec.upper()}' in coln]
            
            if gfd:
                gdf_data[gfd] = gdf_data[gfd].fillna(0)
                gdf_data[sec + '_' + spec] = gdf_data[gfd].sum(axis=1)
                col_sum.append(sec + '_' + spec)
            else:
                gdf_data[sec + '_' + spec] = 0.0
                col_sum.append(sec + '_' + spec)
    
    if 'pm10' in job_parameters['species']:
        for sec in src_type:
            if sec + '_pm10' in gdf_data.columns and sec + '_pm2_5' in gdf_data.columns:
                gdf_data[sec + '_pm10'] += gdf_data[sec + '_pm2_5']
    
    if 'plant_id_left' in gdf_data.columns:
        col_out = ['plant_id_left', 'plant_name_left', 'prtr_code_left', 'prtr_sector_left', 
                   'maingroup_left', 'emission_height_left', 'wgs84_x_left', 'wgs84_y_left', 
                   'geometry'] + col_sum
    else:
        col_out = ['ID_RASTER_left', 'geometry'] + col_sum
    
    return gdf_data[col_out].drop_duplicates()

def load_edgar_profiles(data_parameters):
    """Load EDGAR profiles with robust encoding handling"""
    try:
        hourly_df = pd.read_csv(data_parameters['edgar_hourly'], encoding='utf-8')
    except UnicodeDecodeError:
        hourly_df = pd.read_csv(data_parameters['edgar_hourly'], encoding='iso-8859-1')
    
    try:
        weekly_df = pd.read_csv(data_parameters['edgar_weekly'], encoding='utf-8')
    except UnicodeDecodeError:
        weekly_df = pd.read_csv(data_parameters['edgar_weekly'], encoding='iso-8859-1')
    
    try:
        weekend_types = pd.read_csv(data_parameters['weekend_types'], sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        weekend_types = pd.read_csv(data_parameters['weekend_types'], sep=';', encoding='iso-8859-1')
    
    try:
        daytype_mapping = pd.read_csv(data_parameters['daytype_mapping'], sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        daytype_mapping = pd.read_csv(data_parameters['daytype_mapping'], sep=';', encoding='iso-8859-1')

    hourly_trf = hourly_df[(hourly_df['Country_code_A3'] == 'DEU') & 
                          (hourly_df['activity_code'] == 'TRF')].copy()
    weekly_trf = weekly_df[(weekly_df['Country_code_A3'] == 'DEU') & 
                          (weekly_df['activity_code'] == 'TRF')].copy()
    
    return hourly_trf, weekly_trf, weekend_types, daytype_mapping

def prep_greta_data(data_parameters, job_parameters, sectors):
    """Prepare GRETA data with enhanced error handling and proper unit conversion"""
    bbox_grid = [
        job_parameters['min_lon'],
        job_parameters['min_lat'],
        job_parameters['max_lon'],
        job_parameters['max_lat']
    ]
    bbox_epsg = str(job_parameters['epsg_code'])
    
    try:
        bbox_greta = tuple(bbox_transform(25832, job_parameters['epsg_code'],
                                        job_parameters['min_lon'], job_parameters['min_lat'],
                                        job_parameters['max_lon'], job_parameters['max_lat']))
    except Exception as e:
        raise ValueError(f"Bounding box transformation failed: {str(e)}")

    try:
        prtr1_greta = gpd.read_file(data_parameters['emiss_dir'], layer=2, bbox=bbox_greta)
        prtr2_greta = gpd.read_file(data_parameters['emiss_dir'], layer=3, bbox=bbox_greta)
        prtr_greta = gpd.sjoin(prtr1_greta, prtr2_greta, how="left", predicate="intersects")
    except Exception as e:
        raise RuntimeError(f"Failed to load PRTR data: {str(e)}")

    try:
        rast1_greta = gpd.read_file(data_parameters['emiss_dir'], layer=0, bbox=bbox_greta)
        rast2_greta = gpd.read_file(data_parameters['emiss_dir'], layer=1, bbox=bbox_greta)
        rast_greta = gpd.sjoin(rast1_greta, rast2_greta, how="left", predicate="intersects")
    except Exception as e:
        raise RuntimeError(f"Failed to load raster data: {str(e)}")

    gdf_prtr = nfr_to_gnfr(job_parameters, prtr_greta, sectors)
    gdf_grid = nfr_to_gnfr(job_parameters, rast_greta, sectors)
    
    try:
        gdf_prtr.drop('geometry', axis=1).to_csv(
            os.path.join(job_parameters['job_path'], 'point_sources.csv'),
            index=False
        )
    except Exception as e:
        print(f"Warning: Could not save point sources: {str(e)}")

    # Calculate area in km² (original geometry is in m²)
    gdf_grid['area_km2'] = gdf_grid.geometry.area / 1e6
    
    # Conversion factors
    g_to_kg = 1e6  # 1 Gg = 1,000,000 kg
    km2_to_m2 = 1e6  # 1 km² = 1,000,000 m²
    conversion_factor = g_to_kg / km2_to_m2  # This equals 1 (1 Gg/km² = 1 kg/m²)

    for spec in job_parameters['species']:
        for sec in [s[0] for s in sectors]:
            col_name = f"{sec}_{spec}"
            
            # Convert point sources (PRTR) from Gg to kg
            if col_name in gdf_prtr.columns:
                gdf_prtr[col_name] *= g_to_kg  # Gg → kg
            
            # Convert grid data from Gg/km² to kg/m²
            if col_name in gdf_grid.columns:
                gdf_grid[col_name] = gdf_grid[col_name] * conversion_factor  # Gg/km² → kg/m²

    hourly_trf, weekly_trf, weekend_types, daytype_mapping = load_edgar_profiles(data_parameters)
    
    return gdf_grid, bbox_grid, bbox_epsg, hourly_trf, weekly_trf, weekend_types, daytype_mapping

def apply_temporal_profiles(annual_emiss, sector, hourly_trf, weekly_trf, weekend_types, daytype_mapping, daytype=1):
    """Apply temporal profiles with enhanced road transport handling"""
    hourly_factors = np.ones(24) / 24
    weekly_factor = 1 / 7

    weekday_mask = (daytype_mapping['Daytype_id'] == daytype) & \
                  (daytype_mapping['Weekend_type_id'] == 4)
    weekday_id = daytype_mapping.loc[weekday_mask, 'Weekday_id'].values
    weekday_id = weekday_id[0] if len(weekday_id) > 0 else 1

    if sector == 'F_RoadTransport':
        if not weekly_trf.empty:
            weekly_factor = weekly_trf.loc[
                weekly_trf['Weekday_id'] == weekday_id, 'daily_factor'
            ].values[0]
        
        if not hourly_trf.empty:
            hourly_row = hourly_trf[hourly_trf['Daytype_id'] == daytype]
            if not hourly_row.empty:
                hourly_cols = [f'h{h}' for h in range(1, 25)]
                hourly_factors = hourly_row[hourly_cols].values[0]
                
                if np.sum(hourly_factors) > 0:
                    hourly_factors = hourly_factors / np.sum(hourly_factors)

    daily_emiss = annual_emiss * weekly_factor / 365
    hourly_emiss = np.stack([daily_emiss * factor for factor in hourly_factors], axis=0)

    return hourly_emiss

def downscale_emissions(job_parameters, sectors, gdf_grid, bbox, epsg, hourly_trf, weekly_trf, weekend_types, daytype_mapping):
    """Enhanced downscaling with proper scaling of downscaled values"""
    print('\n=== Starting Emission Downscaling ===')
    print(f"Downscaling BBOX: {bbox}")
    print(f"Resolution: {job_parameters['resol']} m")
    print(f"Species: {job_parameters['species']}")

    x_min, y_min, x_max, y_max = bbox
    resol = job_parameters['resol']
    cols = int(np.ceil((x_max - x_min) / resol))
    rows = int(np.ceil((y_max - y_min) / resol))
    print(f"Output grid: {rows}x{cols} cells")

    # Conversion factors
    GG_TO_KG = 1e6  # 1 Gg = 1,000,000 kg
    KM2_TO_M2 = 1e6  # 1 km² = 1,000,000 m²
    CONV_FACTOR = GG_TO_KG / KM2_TO_M2  # Direct Gg/km² → kg/m²
    coarse_res = 1000  # 1km coarse resolution
    patch_size = int(coarse_res / resol)  # Size of coarse cell in fine cells

    def load_proxy(proxy_path):
        try:
            ds = gdal.Open(proxy_path)
            if ds is None:
                raise ValueError(f"Could not open {proxy_path}")
            arr = ds.ReadAsArray()
            ds = None
            return np.nan_to_num(arr, nan=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load proxy {proxy_path}: {str(e)}")

    try:
        print("Loading proxy data...")
        clc_arr = load_proxy('clc_proxy.tif')[:rows, :cols]
        osm_arr = load_proxy('osm_proxy.tif')[:rows, :cols]
        pop_arr = load_proxy('pop_proxy.tif')[:rows, :cols]
        
        clc_ds = gdal.Open('clc_proxy.tif')
        clc_trans = (x_min, resol, 0, y_max, 0, -resol)
        clc_wkt = clc_ds.GetProjection()
        clc_ds = None
    except Exception as e:
        raise RuntimeError(f"Proxy loading failed: {str(e)}")

    main_sectors = [item[0] for item in sectors]
    max_input_values = {}
    max_output_values = {}

    for spec in tqdm(job_parameters['species'], desc="Processing species"):
        yearly_arr = np.zeros((rows, cols, len(sectors)), dtype=np.float32)
        hourly_arr = np.zeros((24, rows, cols, len(sectors)), dtype=np.float32)
        max_input_values[spec] = {}
        max_output_values[spec] = {}

        for sec_idx, sec in enumerate(tqdm(main_sectors, desc=f"Downscaling {spec}", leave=False)):
            col_name = f"{sec}_{spec}"

            if col_name not in gdf_grid.columns or gdf_grid[col_name].sum() == 0:
                print(f"\nSkipping {col_name}: No emissions found")
                continue

            # Store max input value
            max_input = gdf_grid[col_name].max()
            max_input_values[spec][sec] = max_input
            print(f"\n{col_name} max input: {max_input:.6f} Gg/km²")

            try:
                # Create coarse resolution grid in memory (keep original Gg/km² units)
                out_grid = make_geocube(
                    vector_data=gdf_grid,
                    measurements=[col_name],
                    resolution=(-coarse_res, coarse_res),
                    output_crs=f"EPSG:{epsg}"
                )
                
                # Get the array directly without saving to file
                emis_arr = out_grid[col_name].values
                emis_arr = np.nan_to_num(emis_arr, nan=0)
                
                # Get sector-specific proxy
                if sec == 'A_PublicPower':
                    proxy = np.isin(clc_arr, [121]).astype(float)
                elif sec == 'B_Industry':
                    proxy = np.isin(clc_arr, [121, 131, 133]).astype(float)
                elif sec == 'C_OtherStationaryComb':
                    proxy = pop_arr.copy()
                elif sec == 'D_Fugitives':
                    proxy = np.isin(clc_arr, [121, 131]).astype(float)
                elif sec == 'E_Solvents':
                    proxy = pop_arr.copy()
                elif sec == 'F_RoadTransport':
                    proxy = osm_arr.copy()
                    if np.sum(proxy) <= 0:
                        proxy = np.ones_like(proxy)
                elif sec == 'G_Shipping':
                    proxy = np.isin(clc_arr, [123]).astype(float)
                elif sec == 'H_Aviation':
                    proxy = np.isin(clc_arr, [124]).astype(float)
                elif sec == 'I_OffRoad':
                    proxy = np.isin(clc_arr, [122, 244]).astype(float)
                elif sec == 'J_Waste':
                    proxy = np.isin(clc_arr, [132]).astype(float)
                elif sec == 'K_AgriLivestock':
                    proxy = np.isin(clc_arr, [231]).astype(float)
                elif sec == 'L_AgriOther':
                    proxy = np.isin(clc_arr, [211, 212, 213, 221, 222, 223, 241]).astype(float)
                elif sec == 'SumAllSectors':
                    continue

                # Process in patches corresponding to coarse cells
                sector_arr = np.zeros((rows, cols), dtype=np.float32)
                
                for i in range(0, rows, patch_size):
                    for j in range(0, cols, patch_size):
                        # Get current patch bounds
                        i_end = min(i + patch_size, rows)
                        j_end = min(j + patch_size, cols)
                        
                        # Get coarse cell emissions (Gg/km²)
                        coarse_i, coarse_j = i//patch_size, j//patch_size
                        if coarse_i >= emis_arr.shape[0] or coarse_j >= emis_arr.shape[1]:
                            continue
                            
                        coarse_emis = emis_arr[coarse_i, coarse_j]
                        
                        if coarse_emis > 0:
                            # Get proxy patch
                            proxy_patch = proxy[i:i_end, j:j_end]
                            proxy_sum = np.sum(proxy_patch)
                            
                            if proxy_sum > 0:
                                # Scale proxy weights to sum to patch_size^2 to maintain mass
                                proxy_norm = (proxy_patch / proxy_sum) * (patch_size**2)
                            else:
                                proxy_norm = np.ones_like(proxy_patch)
                            
                            # Convert directly from Gg/km² to kg/m² and distribute
                            sector_arr[i:i_end, j:j_end] = (coarse_emis * CONV_FACTOR) * proxy_norm

                # Store max output value
                max_output = np.max(sector_arr)
                max_output_values[spec][sec] = max_output
                print(f"{col_name} max output: {max_output:.6f} kg/m² (Input was {max_input:.6f} Gg/km²)")
                print(f"Conversion ratio: {max_output/(max_input*CONV_FACTOR):.2f}x")

                yearly_arr[:,:,sec_idx] = sector_arr
                hourly_emiss = apply_temporal_profiles(
                    sector_arr, sec, hourly_trf, weekly_trf, weekend_types, daytype_mapping, daytype=1
                )
                hourly_arr[:,:,:,sec_idx] = hourly_emiss

            except Exception as e:
                print(f"\nError processing {sec}: {str(e)}")
                continue

        # Sum all sectors if needed
        if 'SumAllSectors' in main_sectors:
            sum_idx = main_sectors.index('SumAllSectors')
            yearly_arr[:,:,sum_idx] = np.sum(yearly_arr[:,:,:-1], axis=2)
            hourly_arr[:,:,:,sum_idx] = np.sum(hourly_arr[:,:,:,:-1], axis=3)
            
            # Calculate and print sum comparison
            max_input_sum = gdf_grid[f'SumAllSectors_{spec}'].max()
            max_output_sum = np.max(yearly_arr[:,:,sum_idx])
            print(f"\nSumAllSectors_{spec}:")
            print(f"Max input: {max_input_sum:.6f} Gg/km²")
            print(f"Max output: {max_output_sum:.6f} kg/m²")
            print(f"Conversion ratio: {max_output_sum/(max_input_sum*CONV_FACTOR):.2f}x")

        # Create output file
        output_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_combined.tif')
        try:
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                output_fn,
                cols,
                rows,
                len(main_sectors) * 25,
                gdal.GDT_Float32
            )
            dst_ds.SetGeoTransform(clc_trans)
            dst_ds.SetProjection(clc_wkt)

            band_idx = 1
            for sec_idx, sec in enumerate(main_sectors):
                dst_ds.GetRasterBand(band_idx).WriteArray(yearly_arr[:,:,sec_idx])
                dst_ds.GetRasterBand(band_idx).SetDescription(f"{sec}_yearly")
                band_idx += 1

            for hour in range(24):
                for sec_idx, sec in enumerate(main_sectors):
                    dst_ds.GetRasterBand(band_idx).WriteArray(hourly_arr[hour,:,:,sec_idx])
                    dst_ds.GetRasterBand(band_idx).SetDescription(f"{sec}_h{hour+1}")
                    band_idx += 1

            metadata = {
                'units': 'kg/m²',
                'conversion': 'Direct Gg/km²→kg/m²',
                'input_max_values': str(max_input_values),
                'output_max_values': str(max_output_values)
            }
            dst_ds.SetMetadata(metadata)
            dst_ds = None

            print(f"\nSuccessfully created: {output_fn}")

        except Exception as e:
            print(f"\nError creating output file: {str(e)}")
            continue

    
    print("\n=== Max Value Comparison ===")
    for spec in job_parameters['species']:
        print(f"\nSpecies: {spec}")
        for sec in main_sectors:
            if sec in max_input_values[spec] and sec in max_output_values[spec]:
                print(f"{sec}:")
                print(f"  Input max:  {max_input_values[spec][sec]:.6f} Gg/km²")
                print(f"  Output max: {max_output_values[spec][sec]:.6f} kg/m²")
                print(f"  Ratio: {max_output_values[spec][sec]/(max_input_values[spec][sec]*CONV_FACTOR):.2f}x")
    print("\n=== Downscaling Completed Successfully ===")