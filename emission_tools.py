
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

    gdf_grid['area_km2'] = gdf_grid.geometry.area / 1e6
    
    for spec in job_parameters['species']:
        for sec in [s[0] for s in sectors]:
            col_name = f"{sec}_{spec}"
            if col_name in gdf_prtr.columns:
                gdf_prtr[col_name] *= 1e6
            
            if col_name in gdf_grid.columns:
                gdf_grid[col_name] = gdf_grid[col_name] * 1e6

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
    """Proxy-based emission downscaling with proper resource handling"""
    print('\n=== Starting Emission Downscaling ===')
    print(f"Downscaling BBOX: {bbox}")
    print(f"Resolution: {job_parameters['resol']} m")
    print(f"Species: {job_parameters['species']}")

    x_min, y_min, x_max, y_max = bbox
    resol = job_parameters['resol']
    cols = int(np.ceil((x_max - x_min) / resol))
    rows = int(np.ceil((y_max - y_min) / resol))
    print(f"Output grid: {rows}x{cols} cells")

    def load_proxy(proxy_path):
        try:
            ds = gdal.Open(proxy_path)
            if ds is None:
                raise ValueError(f"Could not open {proxy_path}")
            arr = ds.ReadAsArray()
            ds = None  # Explicitly close dataset
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
        clc_ds = None  # Explicitly close dataset
    except Exception as e:
        raise RuntimeError(f"Proxy loading failed: {str(e)}")

    km2_to_m2 = 1e6
    g_to_kg = 1e3
    main_sectors = [item[0] for item in sectors]

    for spec in tqdm(job_parameters['species'], desc="Processing species"):
        yearly_arr = np.zeros((rows, cols, len(sectors)), dtype=np.float32)
        hourly_arr = np.zeros((24, rows, cols, len(sectors)), dtype=np.float32)

        original_max = gdf_grid[f'SumAllSectors_{spec}'].max() * (g_to_kg / km2_to_m2)
        print(f"\nOriginal SumAllSectors_{spec} max emission: {original_max:.6f} kg/m²")

        for sec_idx, sec in enumerate(tqdm(main_sectors, desc=f"Downscaling {spec}", leave=False)):
            col_name = f"{sec}_{spec}"

            if col_name not in gdf_grid.columns or gdf_grid[col_name].sum() == 0:
                print(f"\nSkipping {col_name}: No emissions found")
                continue

            tmp_raster = f'tmp_emission_{spec}_{sec_idx}.tif'
            out_raster = f'emission_{spec}_{sec_idx}.tif'

            try:
                out_grid = make_geocube(
                    vector_data=gdf_grid,
                    measurements=[col_name],
                    resolution=(-resol, resol),
                    output_crs=f"EPSG:{epsg}"
                )
                out_grid[col_name] = out_grid[col_name].fillna(0)
                out_grid.rio.to_raster(tmp_raster)

                ds = gdal.Open(tmp_raster)
                gdal.Warp(out_raster, ds,
                    xRes=resol, yRes=resol,
                    resampleAlg='average',
                    format='GTiff',
                    dstSRS=f'EPSG:{epsg}',
                    outputBounds=bbox,
                    outputBoundsSRS=f'EPSG:{epsg}',
                    targetAlignedPixels=True)
                ds = None  # Close dataset

                emis_ds = gdal.Open(out_raster)
                emis_arr = emis_ds.ReadAsArray()
                emis_ds = None
                
                emis_arr = emis_arr * (g_to_kg / km2_to_m2)
                print(f"{col_name} max emission: {np.nanmax(emis_arr):.4f} kg/m²")

                sector_arr = np.zeros((rows, cols), dtype=np.float32)

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

                proxy_sum = np.sum(proxy)
                if proxy_sum > 0:
                    proxy_norm = proxy / proxy_sum
                else:
                    proxy_norm = np.ones_like(proxy) / proxy.size

                for i in range(rows):
                    for j in range(cols):
                        sector_arr[i,j] = emis_arr[i,j] * proxy_norm[i,j]

                yearly_arr[:,:,sec_idx] = sector_arr
                hourly_emiss = apply_temporal_profiles(
                    sector_arr, 
                    sec, 
                    hourly_trf, 
                    weekly_trf, 
                    weekend_types, 
                    daytype_mapping, 
                    daytype=1
                )
                hourly_arr[:,:,:,sec_idx] = hourly_emiss

                if os.path.exists(tmp_raster):
                    os.remove(tmp_raster)
                if os.path.exists(out_raster):
                    os.remove(out_raster)

            except Exception as e:
                print(f"\nError processing {sec}: {str(e)}")
                continue

        if 'SumAllSectors' in main_sectors:
            sum_idx = main_sectors.index('SumAllSectors')
            yearly_arr[:,:,sum_idx] = np.sum(yearly_arr[:,:,:-1], axis=2)
            hourly_arr[:,:,:,sum_idx] = np.sum(hourly_arr[:,:,:,:-1], axis=3)
            
            downscaled_max = np.nanmax(yearly_arr[:,:,sum_idx])
            if downscaled_max > 0:
                scale_factor = original_max / downscaled_max
                yearly_arr[:,:,sum_idx] *= scale_factor
                hourly_arr[:,:,:,sum_idx] *= scale_factor
            
            print(f"Downscaled SumAllSectors_{spec} max emission: {np.nanmax(yearly_arr[:,:,sum_idx]):.6f} kg/m²")

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
                'units': 'kg per m²',
                'temporal_resolution': 'hourly + yearly',
                'sectors': ', '.join(main_sectors),
                'pollutant': spec,
                'original_max': str(original_max),
                'final_max': str(np.nanmax(yearly_arr[:,:,sum_idx]))
            }
            dst_ds.SetMetadata(metadata)
            dst_ds = None  # Properly close the dataset

            print(f"\nSuccessfully created: {output_fn}")

        except Exception as e:
            print(f"\nError creating output file: {str(e)}")
            continue

    print("\n=== Downscaling Completed Successfully ===")
#######################
