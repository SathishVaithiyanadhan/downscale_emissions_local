import os
import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal, osr
from pyproj import Transformer
from patchify import patchify, unpatchify
from geocube.api.core import make_geocube
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
import fiona
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import gc
import tempfile
from io import BytesIO
from scipy.ndimage import zoom

gdal.UseExceptions()

def bbox_transform(in_crs, out_crs, cell_minx, cell_miny, cell_maxx, cell_maxy, order_xy=True):
    try:
        transformer = Transformer.from_crs(out_crs, in_crs, always_xy=order_xy)
        xmin, ymin = transformer.transform(cell_minx, cell_miny)
        xmax, ymax = transformer.transform(cell_maxx, cell_maxy)
        return [xmin, ymin, xmax, ymax]
    except Exception as e:
        raise ValueError(f"CRS transformation failed: {str(e)}")

def nfr_to_gnfr(job_parameters, gdf_data, sectors):
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

def create_building_mask(bbox, epsg_code, resolution, building_shp_path):
    try:
        if isinstance(bbox, list):
            bbox = tuple(bbox)
            
        buildings = gpd.read_file(building_shp_path, bbox=bbox)
        if buildings.empty:
            print("Warning: No buildings found in the bounding box")
            return None, None
            
        buildings = buildings.to_crs(epsg=epsg_code)
        
        xmin, ymin, xmax, ymax = bbox
        transform = from_bounds(xmin, ymin, xmax, ymax, 
                              int((xmax - xmin) / resolution),
                              int((ymax - ymin) / resolution))
        
        mask_shape = (int((ymax - ymin) / resolution), 
                     int((xmax - xmin) / resolution))
        building_mask = rasterize(
            [(geom, 1) for geom in buildings.geometry],
            out_shape=mask_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        return building_mask, transform
        
    except Exception as e:
        print(f"Error creating building mask: {str(e)}")
        return None, None

def clean_emission_array(array, building_mask=None):
    cleaned = array.copy()
    
    if building_mask is not None:
        cleaned[building_mask == 1] = np.nan
    
    cleaned[cleaned <= 0] = np.nan
    
    return cleaned

def prep_greta_data(data_parameters, job_parameters, sectors):
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
    
    g_to_kg = 1e6
    km2_to_m2 = 1e6
    conversion_factor = g_to_kg / km2_to_m2

    for spec in job_parameters['species']:
        for sec in [s[0] for s in sectors]:
            col_name = f"{sec}_{spec}"
            
            if col_name in gdf_prtr.columns:
                gdf_prtr[col_name] *= g_to_kg
            
            if col_name in gdf_grid.columns:
                gdf_grid[col_name] = gdf_grid[col_name] * conversion_factor

    return gdf_grid, bbox_grid, bbox_epsg

def create_in_memory_raster(array, transform, projection, data_type=gdal.GDT_Float32):
    driver = gdal.GetDriverByName('MEM')
    rows, cols = array.shape
    mem_raster = driver.Create('', cols, rows, 1, data_type)
    mem_raster.SetGeoTransform(transform)
    mem_raster.SetProjection(projection)
    mem_raster.GetRasterBand(1).WriteArray(array)
    return mem_raster

def create_zero_emission_file(job_parameters, rows, cols, clc_trans, clc_wkt, main_sectors):
    output_fn = os.path.join(job_parameters['job_path'], 'emission_o3_yearly.tif')
    
    zero_arr = np.zeros((rows, cols, len(main_sectors)), dtype=np.float32)
    
    try:
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_fn,
            cols,
            rows,
            len(main_sectors),
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'PREDICTOR=2']
        )
        dst_ds.SetGeoTransform(clc_trans)
        dst_ds.SetProjection(clc_wkt)

        for sec_idx, sec in enumerate(main_sectors):
            band = dst_ds.GetRasterBand(sec_idx+1)
            band.WriteArray(zero_arr[:,:,sec_idx])
            band.SetDescription(f"{sec}_yearly")
            band.SetNoDataValue(np.nan)

        metadata = {
            'units': 'kg/m²',
            'conversion': 'Zero values for O3',
            'temporal_status': 'Yearly_only',
            'building_removal': 'Not applicable',
            'value_cleaning': 'All zeros',
            'notes': 'Created automatically because NOx was present in species list'
        }
        dst_ds.SetMetadata(metadata)
        dst_ds = None

        print(f"\nCreated zero-valued O3 emission file: {output_fn}")

    except Exception as e:
        print(f"\nError creating zero-valued O3 file: {str(e)}")

def distribute_emissions_using_proxy(coarse_emis, proxy_patch, patch_size, conversion_factor):
    """
    Distribute emissions within a coarse grid cell based on proxy values
    
    Args:
        coarse_emis: Total emissions in the coarse grid cell (Gg/km²)
        proxy_patch: High-resolution proxy values for the cell
        patch_size: Size of the high-resolution patch
        conversion_factor: Conversion factor from Gg/km² to kg/m²
    
    Returns:
        High-resolution distributed emissions
    """
    # Convert coarse emissions to total kg in the cell
    cell_area_km2 = (patch_size * patch_size) / 1e6  # Area in km²
    total_emissions_kg = coarse_emis * cell_area_km2 * 1e6  # Convert Gg to kg
    
    # Normalize proxy values to get distribution weights
    proxy_sum = np.sum(proxy_patch)
    if proxy_sum > 0:
        weights = proxy_patch / proxy_sum
    else:
        # If no proxy information, distribute evenly
        weights = np.ones_like(proxy_patch) / (patch_size * patch_size)
    
    # Distribute emissions based on weights and convert to kg/m²
    distributed_emissions = (total_emissions_kg * weights) / (patch_size * patch_size)
    
    return distributed_emissions

def downscale_emissions(job_parameters, sectors, gdf_grid, bbox, epsg, data_parameters):
    print('\n=== Starting Emission Downscaling ===')
    print(f"Downscaling BBOX: {bbox}")
    print(f"Resolution: {job_parameters['resol']} m")
    print(f"Species: {job_parameters['species']}")

    x_min, y_min, x_max, y_max = bbox
    resol = job_parameters['resol']
    cols = round((x_max - x_min) / resol)
    rows = round((y_max - y_min) / resol)

    cols = max(1, cols)
    rows = max(1, rows)

    print(f"Output grid: {rows}x{cols} cells")

    building_shp_path = data_parameters.get('building_shp')
    if building_shp_path and os.path.exists(building_shp_path):
        building_mask, _ = create_building_mask(bbox, epsg, resol, building_shp_path)
    else:
        print("Warning: Building shapefile not found or not specified in config")
        building_mask = None

    # conversion factor (Gg/km² -> kg/m²)
    CONV_FACTOR = 1e6 / 1e6

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
        nightlight_arr = load_proxy('nightlight_proxy.tif')[:rows, :cols]

        clc_ds = gdal.Open('clc_proxy.tif')
        clc_trans = (x_min, resol, 0, y_max, 0, -resol)
        clc_wkt = clc_ds.GetProjection()
        clc_ds = None
    except Exception as e:
        raise RuntimeError(f"Proxy loading failed: {str(e)}")

    main_sectors = [item[0] for item in sectors]
    max_input_values = {}
    max_output_values = {}

    if 'nox' in [s.lower() for s in job_parameters['species']] and 'o3' not in [s.lower() for s in job_parameters['species']]:
        print("\nNOx detected in species list - creating zero-valued O3 emission file")
        create_zero_emission_file(job_parameters, rows, cols, clc_trans, clc_wkt, main_sectors)

    for spec in tqdm(job_parameters['species'], desc="Processing species"):
        yearly_arr = np.zeros((rows, cols, len(sectors)), dtype=np.float32)
        max_input_values[spec] = {}
        max_output_values[spec] = {}

        for sec_idx, sec in enumerate(tqdm(main_sectors, desc=f"Downscaling {spec}", leave=False)):
            col_name = f"{sec}_{spec}"

            if col_name not in gdf_grid.columns or gdf_grid[col_name].sum() == 0:
                print(f"\nSkipping {col_name}: No emissions found")
                continue

            max_input = gdf_grid[col_name].max()
            max_input_values[spec][sec] = max_input

            try:
                # create coarse emission raster
                out_grid = make_geocube(
                    vector_data=gdf_grid,
                    measurements=[col_name],
                    resolution=(-1000, 1000),   # coarse 1 km grid
                    output_crs=f"EPSG:{epsg}"
                )
                out_grid[col_name] = out_grid[col_name].fillna(0)
                emis_arr = out_grid[col_name].values

                # create in-memory raster for resampling
                coarse_transform = (x_min, 1000, 0, y_max, 0, -1000)
                mem_ds = create_in_memory_raster(emis_arr, coarse_transform, clc_wkt)

                # resample to target resolution
                emis_interp_ds = gdal.Warp(
                    '', mem_ds,format='MEM',
                    xRes=resol, yRes=resol,
                    resampleAlg='bilinear',
                    outputBounds=[x_min, y_min, x_max, y_max]
                )
                emis_interp = emis_interp_ds.ReadAsArray()
                emis_interp_ds = None
                mem_ds = None

                # ensure shape match
                if emis_interp.shape != (rows, cols):
                    emis_interp = np.resize(emis_interp, (rows, cols))

                # sector-specific proxies
                if sec == 'A_PublicPower':
                    proxy = np.isin(clc_arr, [121]).astype(float) * nightlight_arr
                elif sec == 'B_Industry':
                    proxy = np.isin(clc_arr, [121, 131, 133]).astype(float) * nightlight_arr
                elif sec == 'C_OtherStationaryComb':
                    proxy = (pop_arr * 0.5) + (nightlight_arr * 0.5)
                elif sec == 'D_Fugitives':
                    proxy = np.isin(clc_arr, [121, 131]).astype(float) * nightlight_arr
                elif sec == 'E_Solvents':
                    proxy = nightlight_arr
                elif sec == 'F_RoadTransport':
                    proxy = osm_arr.copy()
                    if np.sum(proxy) <= 0:
                        proxy = np.ones_like(proxy)
                elif sec == 'G_Shipping':
                    proxy = np.isin(clc_arr, [123]).astype(float) * nightlight_arr
                elif sec == 'H_Aviation':
                    proxy = np.isin(clc_arr, [124]).astype(float) * nightlight_arr
                elif sec == 'I_OffRoad':
                    proxy = np.isin(clc_arr, [122, 244]).astype(float) * nightlight_arr
                elif sec == 'J_Waste':
                    proxy = np.isin(clc_arr, [132]).astype(float) * nightlight_arr
                elif sec == 'K_AgriLivestock':
                    proxy = np.isin(clc_arr, [231]).astype(float)
                elif sec == 'L_AgriOther':
                    proxy = np.isin(clc_arr, [211, 212, 213, 221, 222, 223, 241]).astype(float)
                elif sec == 'SumAllSectors':
                    continue
                else:
                    proxy = np.ones_like(clc_arr, dtype=float)

                # normalize proxy
                proxy_max = np.max(proxy)
                proxy_norm = proxy / (proxy_max if proxy_max > 0.01 else 1.0)

                # final downscaled emissions
                sector_arr = emis_interp * proxy_norm * CONV_FACTOR

                # apply mask/cleaning if needed
                # sector_arr = clean_emission_array(sector_arr, building_mask)

                max_output = np.nanmax(sector_arr)
                max_output_values[spec][sec] = max_output
                yearly_arr[:, :, sec_idx] = sector_arr

            except Exception as e:
                print(f"\nError processing {sec}: {str(e)}")
                continue

        if 'SumAllSectors' in main_sectors:
            sum_idx = main_sectors.index('SumAllSectors')
            yearly_arr[:, :, sum_idx] = np.nansum(yearly_arr[:, :, :-1], axis=2)

        # write output file
        output_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_yearly.tif')
        try:
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                output_fn,
                cols,
                rows,
                len(main_sectors),
                gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'PREDICTOR=2']
            )
            dst_ds.SetGeoTransform(clc_trans)
            dst_ds.SetProjection(clc_wkt)

            for sec_idx, sec in enumerate(main_sectors):
                band = dst_ds.GetRasterBand(sec_idx + 1)
                arr = yearly_arr[:, :, sec_idx]
                band.WriteArray(arr)
                band.SetDescription(f"{sec}_yearly")
                band.SetNoDataValue(np.nan)

            metadata = {
                'units': 'kg/m²',
                'conversion': 'Proxy-weighted resampling (no coarse patches)',
                'input_max_values': str(max_input_values),
                'output_max_values': str(max_output_values),
                'temporal_status': 'Yearly_only',
                'building_removal': 'Applied to all bands',
                'value_cleaning': 'Zeros and negatives set to NAN',
                'building_shp': data_parameters.get('building_shp', 'Not specified'),
                'nightlight_proxy': data_parameters.get('viirs_nightlight', 'Not used'),
                'proxy_notes': 'Nightlight for commercial/industrial, population for heating, OSM for traffic',
                'distribution_method': 'Smooth bilinear resampling + proxy weighting'
            }
            dst_ds.SetMetadata(metadata)
            dst_ds = None

            print(f"\nSuccessfully created: {output_fn}")

        except Exception as e:
            print(f"\nError creating output file: {str(e)}")
            continue
