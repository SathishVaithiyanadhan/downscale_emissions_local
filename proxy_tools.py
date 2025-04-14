"""
Enhanced Proxy Tools for Emission Downscaling
- Improved OSM road handling with traffic-based weighting
- Better error handling and logging
"""
import os
import osmnx as ox
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from geocube.api.core import make_geocube
from shapely.geometry import Polygon, box
import numpy as np
import warnings
from tqdm import tqdm

# Configure OSMnx
ox.settings.log_console = True
ox.settings.timeout = 600
ox.settings.memory = 1024 * 1024 * 500  # 500MB

def bbox_transform(in_crs, out_crs, cell_minx, cell_miny, cell_maxx, cell_maxy, order_xy=True):
    """Transform bounding box coordinates between CRS with error handling"""
    try:
        transformer = Transformer.from_crs(out_crs, in_crs, always_xy=order_xy)
        xmin, ymin = transformer.transform(cell_minx, cell_miny)
        xmax, ymax = transformer.transform(cell_maxx, cell_maxy)
        return [xmin, ymin, xmax, ymax]
    except Exception as e:
        raise ValueError(f"CRS transformation failed: {str(e)}")

def rasterize_clip_shp(job_parameters, out_fn, gdf, lyr, bbox, epsg):
    """Enhanced rasterization with progress tracking"""
    print(f"Rasterizing {lyr} to {out_fn}...")
    tmp_fn = 'tmp_ras.tif'
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_grid = make_geocube(
                vector_data=gdf,
                measurements=[lyr],
                resolution=(-job_parameters['resol'], job_parameters['resol']),
                output_crs=int(epsg)
            )
            out_grid.rio.to_raster(tmp_fn)
            
            ds = gdal.Open(tmp_fn)
            gdal.Warp(out_fn, ds,
                    xRes=job_parameters['resol'], 
                    yRes=job_parameters['resol'],
                    resampleAlg='bilinear', 
                    format='GTiff',
                    dstSRS='EPSG:' + epsg,
                    outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    outputBoundsSRS='EPSG:' + epsg,
                    targetAlignedPixels=False,
                    callback=gdal.TermProgress_nocb)
            ds = None
            
    except Exception as e:
        raise RuntimeError(f"Rasterization failed for {lyr}: {str(e)}")
    finally:
        if os.path.exists(tmp_fn):
            os.remove(tmp_fn)

def prepare_osm_roads(bbox, epsg):
    """
    Retrieve and prepare OSM road data with traffic-based weighting
    Returns GeoDataFrame with geometry and weight columns
    """
    print("Preparing OSM road network with traffic weighting...")
    
    # Transform bbox to WGS84 for OSM query
    bbox_4326 = bbox_transform(4326, int(epsg), bbox[0], bbox[1], bbox[2], bbox[3])
    bbox_polygon = box(*bbox_4326)
    
    # Traffic-based weights (can be adjusted based on local knowledge)
    road_weights = {
        'motorway': 10.0, 'motorway_link': 8.0,
        'trunk': 8.0, 'trunk_link': 6.0,
        'primary': 6.0, 'primary_link': 5.0,
        'secondary': 4.0, 'secondary_link': 3.0,
        'tertiary': 2.0, 'tertiary_link': 1.5,
        'residential': 1.0, 'living_street': 0.8,
        'service': 0.5, 'track': 0.3,
        'unclassified': 0.2
    }
    
    try:
        # Try using graph_from_polygon for older OSMnx versions
        print("Using graph_from_polygon() as fallback...")
        graph = ox.graph_from_polygon(
            bbox_polygon,
            network_type='all',  # all road types
            simplify=True,
            retain_all=True
        )
        gdf_roads = ox.graph_to_gdfs(graph, nodes=False)
        
        # Calculate traffic weights
        def calculate_weight(row):
            highway_type = row['highway']
            if isinstance(highway_type, list):
                highway_type = highway_type[0]  # Take first if multiple types
            
            base_weight = road_weights.get(highway_type, 0.1)
            return base_weight
        
        print("Calculating road weights...")
        gdf_roads['weight'] = gdf_roads.apply(calculate_weight, axis=1)
        
        # Simplify geometries for performance
        gdf_roads['geometry'] = gdf_roads['geometry'].simplify(10)
        
        return gdf_roads[['geometry', 'weight']].to_crs(epsg=int(epsg))
        
    except Exception as e:
        print(f"Error preparing OSM roads: {str(e)}")
        return gpd.GeoDataFrame(columns=['geometry', 'weight'], crs=f"EPSG:{epsg}")

def downscaling_proxies(data_parameters, job_parameters, bbox, epsg):
    """Enhanced proxy preparation with better road handling"""
    print('\n=== Preparing Enhanced Downscaling Proxies ===')
    gdal.UseExceptions()
    
    # 1. OSM Road Network (Enhanced)
    print("\n1. Processing OSM road network...")
    gdf_roads = prepare_osm_roads(bbox, epsg)
    
    if not gdf_roads.empty:
        # First create a template raster with exact dimensions
        x_min, y_min, x_max, y_max = bbox
        resol = job_parameters['resol']
        cols = int((x_max - x_min) / resol)
        rows = int((y_max - y_min) / resol)
        
        # Create template raster as xarray Dataset
        import xarray as xr
        import rioxarray
        
        # Create coordinates
        x_coords = np.linspace(x_min + resol/2, x_max - resol/2, cols)
        y_coords = np.linspace(y_max - resol/2, y_min + resol/2, rows)
        
        # Create template dataset
        template = xr.Dataset(
            coords={
                'x': x_coords,
                'y': y_coords
            }
        )
        template.rio.write_crs(f"EPSG:{epsg}", inplace=True)
        
        # Create road presence raster
        rasterize_clip_shp(job_parameters, 'osm_presence.tif', gdf_roads, 'weight', bbox, epsg)
        
        # Create road weight raster using template
        out_grid = make_geocube(
            vector_data=gdf_roads,
            measurements=['weight'],
            like=template  # Use template dataset for output dimensions and CRS
        )
        out_grid.rio.to_raster('osm_weights.tif')
        
        # Read both rasters
        presence_ds = gdal.Open('osm_presence.tif')
        weights_ds = gdal.Open('osm_weights.tif')
        
        # Read arrays (should be same dimensions now)
        presence = presence_ds.GetRasterBand(1).ReadAsArray()
        weights = weights_ds.GetRasterBand(1).ReadAsArray()
        
        # Combine into final proxy
        final_proxy = np.where(presence > 0, weights, 0)
        
        # Normalize and save final proxy
        if np.sum(final_proxy) > 0:
            final_proxy = final_proxy / np.sum(final_proxy)
        
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create('osm_proxy.tif', cols, rows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((x_min, resol, 0, y_max, 0, -resol))
        ds.SetProjection(f'EPSG:{epsg}')
        ds.GetRasterBand(1).WriteArray(final_proxy)
        ds = None
        
        # Clean up
        for f in ['osm_presence.tif', 'osm_weights.tif']:
            if os.path.exists(f):
                os.remove(f)
    else:
        raise RuntimeError("Failed to create OSM road proxy - no road data available")
    
    # 2. URBAN ATLAS Land Cover
    print("\n2. Processing URBAN ATLAS land cover data...")
    try:
        bbox_3035 = bbox_transform(3035, int(epsg), bbox[0], bbox[1], bbox[2], bbox[3])
        gdf_clc = gpd.read_file(
            data_parameters['urbanAtlas_dir'],
            bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
        )
        gdf_clc['code_2018'] = gdf_clc['code_2018'].astype('int')
        gdf_clc = gdf_clc.to_crs(epsg=int(epsg))
        rasterize_clip_shp(job_parameters, 'clc_proxy.tif', gdf_clc, 'code_2018', bbox, epsg)
    except Exception as e:
        raise RuntimeError(f"URBAN ATLAS processing failed: {str(e)}")
    
    # 3. Population Density
    print("\n3. Processing population density data...")
    try:
        gdal.Warp('pop_proxy.tif', data_parameters['popul_dir'],
                xRes=resol, 
                yRes=resol,
                resampleAlg='bilinear', 
                format='GTiff',
                dstSRS=f'EPSG:{epsg}',
                outputBounds=(x_min, y_min, x_max, y_max),
                outputBoundsSRS=f'EPSG:{epsg}',
                targetAlignedPixels=False,
                callback=gdal.TermProgress_nocb)
    except Exception as e:
        raise RuntimeError(f"Population processing failed: {str(e)}")
    
    print("\n=== Proxy preparation completed successfully ===")

#####

