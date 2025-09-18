
"""
Enhanced Proxy Tools with Nighttime Lights and Building Height Support
"""
import os
import osmnx as ox
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr
from pyproj import Transformer
from geocube.api.core import make_geocube
from shapely.geometry import Polygon, box, LineString
import numpy as np
import warnings
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# Configure OSMnx
ox.settings.log_console = True
ox.settings.timeout = 600
ox.settings.memory = 1024 * 1024 * 500
ox.settings.use_cache = True
ox.settings.cache_folder = "./osmnx_cache"
ox.settings.simplify_algorithm = 'douglas-peucker'

def bbox_transform(in_crs, out_crs, cell_minx, cell_miny, cell_maxx, cell_maxy, order_xy=True):
    try:
        transformer = Transformer.from_crs(out_crs, in_crs, always_xy=order_xy)
        xmin, ymin = transformer.transform(cell_minx, cell_miny)
        xmax, ymax = transformer.transform(cell_maxx, cell_maxy)
        return [xmin, ymin, xmax, ymax]
    except Exception as e:
        raise ValueError(f"CRS transformation failed: {str(e)}")

def rasterize_clip_shp(job_parameters, out_fn, gdf, lyr, bbox, epsg):
    print(f"Rasterizing {lyr} to {out_fn}...")
    tmp_fn = 'tmp_ras.tif'
    
    try:
        if 'weight' not in gdf.columns:
            raise ValueError("Weight column missing in land use data")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_grid = make_geocube(
                vector_data=gdf,
                measurements=['weight'],
                resolution=(-job_parameters['resol'], job_parameters['resol']),
                output_crs=f"EPSG:{epsg}"
            )
            out_grid.rio.to_raster(tmp_fn)
            ds = gdal.Open(tmp_fn)
            gdal.Warp(out_fn, ds,
                    xRes=job_parameters['resol'], 
                    yRes=job_parameters['resol'],
                    resampleAlg='near',
                    format='GTiff',
                    dstSRS=f'EPSG:{epsg}',
                    outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    outputBoundsSRS=f'EPSG:{epsg}',
                    targetAlignedPixels=True,
                    callback=gdal.TermProgress_nocb)
            ds = None
            
    except Exception as e:
        raise RuntimeError(f"Rasterization failed for {lyr}: {str(e)}")
    finally:
        if os.path.exists(tmp_fn):
            os.remove(tmp_fn)

def rasterize_line_shp(out_fn, gdf, lyr, bbox, epsg, resolution):
    print(f"Rasterizing {lyr} to {out_fn} with smooth lines...")
    
    x_min, y_min, x_max, y_max = bbox
    cols = int((x_max - x_min) / resolution)
    rows = int((y_max - y_min) / resolution)
    
    driver = ogr.GetDriverByName('Memory')
    ds = driver.CreateDataSource('temp')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer('roads', srs, ogr.wkbLineString)
    
    field_defn = ogr.FieldDefn('weight', ogr.OFTReal)
    layer.CreateField(field_defn)
    
    for _, row in gdf.iterrows():
        feat = ogr.Feature(layer.GetLayerDefn())
        geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
        feat.SetGeometry(geom)
        feat.SetField('weight', row[lyr])
        layer.CreateFeature(feat)
        feat = None
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_fn, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform((x_min, resolution, 0, y_max, 0, -resolution))
    out_ds.SetProjection(srs.ExportToWkt())
    
    gdal.RasterizeLayer(out_ds, [1], layer, 
                       options=["ATTRIBUTE=weight", 
                                "BURN_VALUE_FROM=Z",
                                "ALL_TOUCHED=TRUE"])
    
    ds = None
    out_ds = None
    
    ds = gdal.Open(out_fn, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    mask = arr > 0
    dilated = binary_dilation(mask, structure=np.ones((3,3)), iterations=1)
    arr[dilated & ~mask] = np.nanmean(arr[mask])
    band.WriteArray(arr)
    ds.FlushCache()
    ds = None

def prepare_osm_roads(bbox, epsg):
    print("Preparing OSM road network...")
    try:
        bbox_4326 = bbox_transform(4326, int(epsg), bbox[0], bbox[1], bbox[2], bbox[3])
        bbox_polygon = box(*bbox_4326)
        
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
        
        graph = ox.graph_from_polygon(
            bbox_polygon,
            network_type='all',
            simplify=True,
            retain_all=True,
            truncate_by_edge=True
        )
        
        gdf_roads = ox.graph_to_gdfs(graph, nodes=False)
        gdf_roads = gdf_roads.explode(index_parts=True)
        gdf_roads = gdf_roads[~gdf_roads.is_empty]
        
        def calculate_weight(row):
            highway_type = row['highway']
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            return road_weights.get(highway_type, 0.1)
        
        gdf_roads['weight'] = gdf_roads.apply(calculate_weight, axis=1)
        gdf_roads['geometry'] = gdf_roads['geometry'].simplify(0.0001, preserve_topology=True)
        gdf_roads = gdf_roads.dissolve(by='weight').reset_index()
        gdf_roads = gdf_roads.to_crs(epsg=int(epsg))
        
        if gdf_roads.empty:
            minx, miny, maxx, maxy = bbox
            dummy_road = gpd.GeoDataFrame(
                {'weight': [1.0]},
                geometry=[LineString([(minx, miny), (maxx, maxy)])],
                crs=f"EPSG:{epsg}"
            )
            return dummy_road
            
        return gdf_roads[['geometry', 'weight']]
        
    except Exception as e:
        print(f"Error preparing OSM roads: {str(e)}")
        minx, miny, maxx, maxy = bbox
        return gpd.GeoDataFrame(
            {'weight': [1.0]},
            geometry=[LineString([(minx, miny), (maxx, maxy)])],
            crs=f"EPSG:{epsg}"
        )

def process_urban_atlas(data_parameters, bbox, epsg):
    print("Processing Urban Atlas land use data...")
    
    try:
        bbox_3035 = bbox_transform(3035, epsg, bbox[0], bbox[1], bbox[2], bbox[3])
        gdf_clc = gpd.read_file(
            data_parameters['urbanAtlas_dir'],
            bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
        )
        
        if 'code_2018' not in gdf_clc.columns:
            raise ValueError("Urban Atlas data missing 'code_2018' column")
            
        gdf_clc['code_2018'] = pd.to_numeric(gdf_clc['code_2018'], errors='coerce').fillna(-1).astype(int)
        gdf_clc['weight'] = gdf_clc['code_2018']
        gdf_clc = gdf_clc.to_crs(epsg=epsg)
        gdf_clc = gdf_clc[gdf_clc.is_valid]
        
        return gdf_clc
        
    except Exception as e:
        raise RuntimeError(f"Urban Atlas processing failed: {str(e)}")

def process_nightlight(data_parameters, bbox, epsg, resolution):
    print("Processing VIIRS nighttime light data...")
    try:
        if os.path.exists('nightlight_proxy.tif'):
            os.remove('nightlight_proxy.tif')
            
        gdal.Warp('nightlight_proxy.tif', data_parameters['viirs_nightlight'],
                xRes=resolution, yRes=resolution,
                resampleAlg='bilinear', 
                format='GTiff',
                dstSRS=f'EPSG:{epsg}',
                outputBounds=bbox,
                outputBoundsSRS=f'EPSG:{epsg}',
                targetAlignedPixels=True,
                callback=gdal.TermProgress_nocb)
        
        ds = gdal.Open('nightlight_proxy.tif', gdal.GA_Update)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        arr[arr < 0] = 0
        if np.max(arr) > 0:
            arr = arr / np.max(arr)
        band.WriteArray(arr)
        ds.FlushCache()
        ds = None
        return True
        
    except Exception as e:
        print(f"Error processing nighttime lights: {str(e)}")
        return False

def downscaling_proxies(data_parameters, job_parameters, bbox, epsg):
    print('\n=== Preparing Proxies ===')
    gdal.UseExceptions()
    
    x_min, y_min, x_max, y_max = bbox
    resol = job_parameters['resol']
    cols = int((x_max - x_min) / resol)
    rows = int((y_max - y_min) / resol)
    
    try:
        epsg = int(epsg)
    except ValueError:
        raise ValueError(f"Invalid EPSG code: {epsg}")

    # 1. Process OSM roads
    print("\n1. Processing OSM roads...")
    gdf_roads = prepare_osm_roads(bbox, epsg)
    if not gdf_roads.empty:
        rasterize_line_shp('osm_proxy.tif', gdf_roads, 'weight', bbox, epsg, resol)
    else:
        neutral = np.ones((rows, cols), dtype=np.float32)
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create('osm_proxy.tif', cols, rows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((x_min, resol, 0, y_max, 0, -resol))
        ds.SetProjection(f'EPSG:{epsg}')
        band = ds.GetRasterBand(1)
        band.WriteArray(neutral)
        band.FlushCache()
        ds = None

    # 2. Process Urban Atlas
    print("\n2. Processing Urban Atlas...")
    try:
        gdf_clc = process_urban_atlas(data_parameters, bbox, epsg)
        rasterize_clip_shp(job_parameters, 'clc_proxy.tif', gdf_clc, 'weight', bbox, epsg)
    except Exception as e:
        raise RuntimeError(f"Urban Atlas processing failed: {str(e)}")

    # 3. Process population density
    print("\n3. Processing population density...")
    try:
        if os.path.exists('pop_proxy.tif'):
            os.remove('pop_proxy.tif')
        gdal.Warp('pop_proxy.tif', data_parameters['popul_dir'],
                xRes=resol, yRes=resol,
                resampleAlg='bilinear', 
                format='GTiff',
                dstSRS=f'EPSG:{epsg}',
                outputBounds=(x_min, y_min, x_max, y_max),
                outputBoundsSRS=f'EPSG:{epsg}',
                targetAlignedPixels=True,
                callback=gdal.TermProgress_nocb)
    except Exception as e:
        raise RuntimeError(f"Population processing failed: {str(e)}")

    # 4. Process nighttime lights
    print("\n4. Processing nighttime lights...")
    if not process_nightlight(data_parameters, bbox, epsg, resol):
        print("Using neutral nighttime light proxy")
        neutral = np.ones((rows, cols), dtype=np.float32) * 0.5
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create('nightlight_proxy.tif', cols, rows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((x_min, resol, 0, y_max, 0, -resol))
        ds.SetProjection(f'EPSG:{epsg}')
        band = ds.GetRasterBand(1)
        band.WriteArray(neutral)
        band.FlushCache()
        ds = None

    print("\n=== Proxy preparation completed ===")