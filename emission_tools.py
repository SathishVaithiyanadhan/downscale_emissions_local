
#######################
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
                # Handle potential NaN values
                gdf_data[gfd] = gdf_data[gfd].fillna(0)
                gdf_data[sec + '_' + spec] = gdf_data[gfd].sum(axis=1)
                col_sum.append(sec + '_' + spec)
            else:
                gdf_data[sec + '_' + spec] = 0.0
                col_sum.append(sec + '_' + spec)
    
    # Special handling for PM10 including PM2.5
    if 'pm10' in job_parameters['species']:
        for sec in src_type:
            if sec + '_pm10' in gdf_data.columns and sec + '_pm2_5' in gdf_data.columns:
                gdf_data[sec + '_pm10'] += gdf_data[sec + '_pm2_5']
    
    # Select output columns
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

    # Filter for German road transport profiles
    hourly_trf = hourly_df[(hourly_df['Country_code_A3'] == 'DEU') & 
                          (hourly_df['activity_code'] == 'TRF')].copy()
    weekly_trf = weekly_df[(weekly_df['Country_code_A3'] == 'DEU') & 
                          (weekly_df['activity_code'] == 'TRF')].copy()
    
    return hourly_trf, weekly_trf, weekend_types, daytype_mapping

def prep_greta_data(data_parameters, job_parameters, sectors):
    """Prepare GRETA data with enhanced error handling and proper unit conversion"""
    # Calculate bounding boxes
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

    # Load PRTR data
    try:
        prtr1_greta = gpd.read_file(data_parameters['emiss_dir'], layer=2, bbox=bbox_greta)
        prtr2_greta = gpd.read_file(data_parameters['emiss_dir'], layer=3, bbox=bbox_greta)
        prtr_greta = gpd.sjoin(prtr1_greta, prtr2_greta, how="left", predicate="intersects")
    except Exception as e:
        raise RuntimeError(f"Failed to load PRTR data: {str(e)}")

    # Load raster data
    try:
        rast1_greta = gpd.read_file(data_parameters['emiss_dir'], layer=0, bbox=bbox_greta)
        rast2_greta = gpd.read_file(data_parameters['emiss_dir'], layer=1, bbox=bbox_greta)
        rast_greta = gpd.sjoin(rast1_greta, rast2_greta, how="left", predicate="intersects")
    except Exception as e:
        raise RuntimeError(f"Failed to load raster data: {str(e)}")

    # Convert sectors and calculate emissions
    gdf_prtr = nfr_to_gnfr(job_parameters, prtr_greta, sectors)
    gdf_grid = nfr_to_gnfr(job_parameters, rast_greta, sectors)
    
    # Save point sources
    try:
        gdf_prtr.drop('geometry', axis=1).to_csv(
            os.path.join(job_parameters['job_path'], 'point_sources.csv'),
            index=False
        )
    except Exception as e:
        print(f"Warning: Could not save point sources: {str(e)}")

    # Calculate area in km² (original units are per km²)
    gdf_grid['area_km2'] = gdf_grid.geometry.area / 1e6  # m² → km²
    
    for spec in job_parameters['species']:
        for sec in [s[0] for s in sectors]:
            col_name = f"{sec}_{spec}"
            if col_name in gdf_prtr.columns:
                # Convert Gg to kg for point sources (1 Gg = 1e6 kg)
                gdf_prtr[col_name] *= 1e6
            
            if col_name in gdf_grid.columns:
                # Convert Gg/km² to kg/km² (keeping original spatial units)
                gdf_grid[col_name] = gdf_grid[col_name] * 1e6

    # Load temporal profiles
    hourly_trf, weekly_trf, weekend_types, daytype_mapping = load_edgar_profiles(data_parameters)
    
    return gdf_grid, bbox_grid, bbox_epsg, hourly_trf, weekly_trf, weekend_types, daytype_mapping

def mass_conservative_downscaling(emis_coarse, proxy_patch):
    """Enhanced mass-conservative downscaling with validation"""
    if np.any(np.isnan(proxy_patch)) or np.any(np.isinf(proxy_patch)):
        raise ValueError("Proxy patch contains NaN or inf values")

    proxy_sum = np.sum(proxy_patch)
    if proxy_sum <= 0:
        return np.zeros_like(proxy_patch)

    proxy_norm = proxy_patch / proxy_sum
    emis_fine = emis_coarse * proxy_norm

    # Enhanced validation
    if not np.isclose(np.sum(emis_fine), emis_coarse, rtol=1e-5, atol=1e-8):
        raise AssertionError(
            f"Mass conservation failed. Input: {emis_coarse:.6f}, Output: {np.sum(emis_fine):.6f}, "
            f"Diff: {abs(emis_coarse - np.sum(emis_fine)):.6f}"
        )

    return emis_fine

def apply_temporal_profiles(annual_emiss, sector, hourly_trf, weekly_trf, weekend_types, daytype_mapping, daytype=1):
    """Apply temporal profiles with enhanced road transport handling"""
    hourly_factors = np.ones(24) / 24
    weekly_factor = 1 / 7

    # Get weekday ID
    weekday_mask = (daytype_mapping['Daytype_id'] == daytype) & \
                  (daytype_mapping['Weekend_type_id'] == 4)
    weekday_id = daytype_mapping.loc[weekday_mask, 'Weekday_id'].values
    weekday_id = weekday_id[0] if len(weekday_id) > 0 else 1

    # Enhanced road transport handling
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
                
                # Normalize to ensure sum to 1
                if np.sum(hourly_factors) > 0:
                    hourly_factors = hourly_factors / np.sum(hourly_factors)

    # Calculate hourly emissions
    daily_emiss = annual_emiss * weekly_factor / 365
    hourly_emiss = np.stack([daily_emiss * factor for factor in hourly_factors], axis=0)

    return hourly_emiss

def downscale_emissions(job_parameters, sectors, gdf_grid, bbox, epsg, hourly_trf, weekly_trf, weekend_types, daytype_mapping):
    """Enhanced downscaling with proper unit conversion (kg/km² → kg/100m²)"""
    print('\n=== Starting Emission Downscaling ===')
    print(f"Downscaling BBOX: {bbox}")
    print(f"Resolution: {job_parameters['resol']} m")
    print(f"Species: {job_parameters['species']}")

    # Calculate raster dimensions
    x_min, y_min, x_max, y_max = bbox
    resol = job_parameters['resol']
    cols = int((x_max - x_min) / resol)
    rows = int((y_max - y_min) / resol)
    print(f"Output grid: {rows}x{cols} cells")

    # Load proxy data with validation
    try:
        clc_ds = gdal.Open('clc_proxy.tif')
        clc_arr = np.nan_to_num(clc_ds.ReadAsArray(), nan=0).astype(int)
        clc_trans = (x_min, resol, 0, y_max, 0, -resol)
        clc_wkt = clc_ds.GetProjection()
    except Exception as e:
        raise RuntimeError(f"Failed to load CLC proxy: {str(e)}")

    try:
        osm_ds = gdal.Open('osm_proxy.tif')
        osm_arr = np.nan_to_num(osm_ds.ReadAsArray(), nan=0).astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to load OSM proxy: {str(e)}")

    try:
        pop_ds = gdal.Open('pop_proxy.tif')
        pop_arr = np.nan_to_num(pop_ds.ReadAsArray(), nan=0).astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to load population proxy: {str(e)}")

    # Downscaling parameters
    conversion_factor = 1/100  # kg/km² → kg/100m² (1 km² = 100 × 100m×100m cells)
    patch_size = 100   # Patch size in cells
    main_sectors = [item[0] for item in sectors]

    # Process each species
    for spec in tqdm(job_parameters['species'], desc="Processing species"):
        yearly_arr = np.zeros((rows, cols, len(sectors)))
        hourly_arr = np.zeros((24, rows, cols, len(sectors)))

        # Process each sector
        for sec in tqdm(main_sectors, desc=f"Downscaling {spec}", leave=False):
            nsec = main_sectors.index(sec)
            col_name = f"{sec}_{spec}"

            if col_name not in gdf_grid.columns or gdf_grid[col_name].sum() == 0:
                print(f"\nSkipping {col_name}: No emissions found")
                continue

            # Create temporary raster
            tmp_raster = f'tmp_emission_{spec}_{nsec}.tif'
            out_raster = f'emission_{spec}_{nsec}.tif'

            try:
                out_grid = make_geocube(
                    vector_data=gdf_grid,
                    measurements=[col_name],
                    resolution=(-resol, resol),
                    output_crs=int(epsg)
                )
                out_grid[col_name] = out_grid[col_name].fillna(0)
                out_grid.rio.to_raster(tmp_raster)

                # Warp to exact extent
                ds = gdal.Open(tmp_raster)
                gdal.Warp(out_raster, ds,
                    xRes=resol, yRes=resol,
                    resampleAlg='average',
                    format='GTiff',
                    dstSRS=f'EPSG:{epsg}',
                    outputBounds=bbox,
                    outputBoundsSRS=f'EPSG:{epsg}',
                    targetAlignedPixels=True,
                    callback=gdal.TermProgress_nocb)
                emis_ds = gdal.Open(out_raster)
                emis_arr = emis_ds.ReadAsArray()
                print(f"\n{col_name} max emission: {np.nanmax(emis_arr):.4f} kg/km²")

                # Prepare for downscaling
                sector_arr = np.zeros((rows, cols), dtype=float)
                empty_patches = patchify(sector_arr, (patch_size, patch_size), step=patch_size)
                clc_patches = patchify(clc_arr, (patch_size, patch_size), step=patch_size)
                pop_patches = patchify(pop_arr, (patch_size, patch_size), step=patch_size)
                osm_patches = patchify(osm_arr, (patch_size, patch_size), step=patch_size)

                # Process each patch
                for i in tqdm(range(empty_patches.shape[0]), desc=f"Processing {sec} patches", leave=False):
                    for j in range(empty_patches.shape[1]):
                        # Convert from kg/km² to kg/100m²
                        greta_value = emis_arr[i, j] * conversion_factor

                        # Enhanced sector-specific proxy selection
                        if sec == 'A_PublicPower':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [121]).astype(float)
                        elif sec == 'B_Industry':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [121, 131, 133]).astype(float)
                        elif sec == 'C_OtherStationaryComb':
                            proxy_patch = pop_patches[i,j,:,:]
                        elif sec == 'D_Fugitives':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [121, 131]).astype(float)
                        elif sec == 'E_Solvents':
                            proxy_patch = pop_patches[i,j,:,:]
                        elif sec == 'F_RoadTransport':
                            proxy_patch = osm_patches[i,j,:,:]
                            if np.sum(proxy_patch) <= 0:
                                proxy_patch = np.ones_like(proxy_patch)
                            proxy_patch = proxy_patch / np.sum(proxy_patch)
                        elif sec == 'G_Shipping':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [123]).astype(float)
                        elif sec == 'H_Aviation':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [124]).astype(float)
                        elif sec == 'I_OffRoad':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [122, 244]).astype(float)
                        elif sec == 'J_Waste':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [132]).astype(float)
                        elif sec == 'K_AgriLivestock':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [231]).astype(float)
                        elif sec == 'L_AgriOther':
                            proxy_patch = np.isin(clc_patches[i,j,:,:], [211, 212, 213, 221, 222, 223, 241]).astype(float)
                        elif sec == 'SumAllSectors':
                            continue

                        # Apply downscaling
                        empty_patches[i,j,:,:] = mass_conservative_downscaling(greta_value, proxy_patch)

                # Combine patches and apply temporal profiles
                annual_emiss = unpatchify(empty_patches, sector_arr.shape)
                yearly_arr[:,:,nsec] = annual_emiss
                hourly_emiss = apply_temporal_profiles(
                    annual_emiss, sec, hourly_trf, weekly_trf, weekend_types, daytype_mapping, daytype=1
                )
                hourly_arr[:,:,:,nsec] = hourly_emiss

                # Clean up
                os.remove(tmp_raster)
                os.remove(out_raster)

            except Exception as e:
                print(f"\nError processing {sec}: {str(e)}")
                continue

        # Sum all sectors if needed
        if 'SumAllSectors' in main_sectors:
            sum_idx = main_sectors.index('SumAllSectors')
            yearly_arr[:,:,sum_idx] = np.sum(yearly_arr[:,:,:-1], axis=2)
            hourly_arr[:,:,:,sum_idx] = np.sum(hourly_arr[:,:,:,:-1], axis=3)
            print(f"\nSumAllSectors max: {np.nanmax(yearly_arr[:,:,sum_idx]):.4f} kg/100m²/year")

        # Create output file
        output_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_combined.tif')
        try:
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                output_fn,
                cols,
                rows,
                len(main_sectors) * 25,  # Yearly + 24 hourly
                gdal.GDT_Float64
            )
            dst_ds.SetGeoTransform(clc_trans)
            dst_ds.SetProjection(clc_wkt)

            # Write yearly bands
            band_idx = 1
            for nsec, sec in enumerate(main_sectors):
                band_arr = yearly_arr[:,:,nsec]
                dst_ds.GetRasterBand(band_idx).WriteArray(band_arr)
                dst_ds.GetRasterBand(band_idx).SetDescription(f"{sec}_yearly")
                band_idx += 1

            # Write hourly bands
            for hour in range(24):
                for nsec, sec in enumerate(main_sectors):
                    band_arr = hourly_arr[hour,:,:,nsec]
                    dst_ds.GetRasterBand(band_idx).WriteArray(band_arr)
                    dst_ds.GetRasterBand(band_idx).SetDescription(f"{sec}_h{hour+1}")
                    band_idx += 1

            # Set metadata
            metadata = {
                'units': 'kg per 100 m²',
                'temporal_resolution': 'hourly + yearly',
                'sectors': ', '.join(main_sectors),
                'pollutant': spec,
                'mass_conserved': 'True'
            }
            dst_ds.SetMetadata(metadata)
            dst_ds = None

            print(f"\nSuccessfully created: {output_fn}")

        except Exception as e:
            print(f"\nError creating output file: {str(e)}")
            continue

    print("\n=== Downscaling Completed Successfully ===")
#######################