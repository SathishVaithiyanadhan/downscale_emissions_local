
"""
Enhanced Proxy Tools with Nighttime Lights and Building Height Support
"""
#old urbanatlas only
'''import os
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

    print("\n=== Proxy preparation completed ===")'''

    ####

"""
Enhanced Proxy Tools with Nighttime Lights and Building Height Support
Added: Hybrid land use strategy (Urban Atlas + CORINE)
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

def enforce_shape(arr, rows, cols):
    """Enforce that array has exact shape (rows, cols)"""
    if arr.shape == (rows, cols):
        return arr
    
    print(f"    Reshaping array from {arr.shape} to ({rows}, {cols})")
    temp = np.full((rows, cols), np.nan, dtype=np.float32)
    r = min(arr.shape[0], rows)
    c = min(arr.shape[1], cols)
    temp[:r, :c] = arr[:r, :c]
    return temp

def rasterize_clip_shp(job_parameters, out_fn, gdf, lyr, bbox_exact, epsg, rows, cols):
    """Rasterize shapefile to exact grid dimensions"""
    print(f"Rasterizing {lyr} to {out_fn}...")
    tmp_fn = 'tmp_ras.tif'
    resol = job_parameters['resol']
    x_min, y_min, x_max, y_max = bbox_exact
    
    try:
        if 'weight' not in gdf.columns:
            raise ValueError("Weight column missing in land use data")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_grid = make_geocube(
                vector_data=gdf,
                measurements=['weight'],
                resolution=(-resol, resol),
                output_crs=f"EPSG:{epsg}"
            )
            out_grid.rio.to_raster(tmp_fn)
            ds = gdal.Open(tmp_fn)
            
            # Use exact dimensions and bounds
            gdal.Warp(out_fn, ds,
                    xRes=resol, yRes=resol,
                    resampleAlg='near',
                    format='GTiff',
                    dstSRS=f'EPSG:{epsg}',
                    outputBounds=bbox_exact,
                    outputBoundsSRS=f'EPSG:{epsg}',
                    targetAlignedPixels=True,
                    width=cols, height=rows,
                    callback=gdal.TermProgress_nocb)
            ds = None
            
    except Exception as e:
        raise RuntimeError(f"Rasterization failed for {lyr}: {str(e)}")
    finally:
        if os.path.exists(tmp_fn):
            os.remove(tmp_fn)

def rasterize_line_shp(out_fn, gdf, lyr, bbox_exact, epsg, resol, rows, cols):
    """Rasterize line shapefile to exact grid dimensions"""
    print(f"Rasterizing {lyr} to {out_fn} with smooth lines...")
    
    x_min, y_min, x_max, y_max = bbox_exact
    
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
    out_ds.SetGeoTransform((x_min, resol, 0, y_max, 0, -resol))
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
    if np.any(mask):
        arr[dilated & ~mask] = np.nanmean(arr[mask])
    band.WriteArray(arr)
    ds.FlushCache()
    ds = None

def prepare_osm_roads(bbox, epsg, bbox_exact):
    """Prepare OSM road network"""
    print("Preparing OSM road network...")
    try:
        bbox_4326 = bbox_transform(4326, int(epsg), bbox_exact[0], bbox_exact[1], bbox_exact[2], bbox_exact[3])
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
            minx, miny, maxx, maxy = bbox_exact
            dummy_road = gpd.GeoDataFrame(
                {'weight': [1.0]},
                geometry=[LineString([(minx, miny), (maxx, maxy)])],
                crs=f"EPSG:{epsg}"
            )
            return dummy_road
            
        return gdf_roads[['geometry', 'weight']]
        
    except Exception as e:
        print(f"Error preparing OSM roads: {str(e)}")
        minx, miny, maxx, maxy = bbox_exact
        return gpd.GeoDataFrame(
            {'weight': [1.0]},
            geometry=[LineString([(minx, miny), (maxx, maxy)])],
            crs=f"EPSG:{epsg}"
        )

def process_urban_atlas(data_parameters, bbox_exact, epsg, rows, cols, resol):
    """Process Urban Atlas data - high quality urban land use"""
    print("Processing Urban Atlas land use data...")
    
    try:
        # Transform bbox to Urban Atlas CRS (EPSG:3035)
        bbox_3035 = bbox_transform(3035, epsg, bbox_exact[0], bbox_exact[1], bbox_exact[2], bbox_exact[3])
        
        # Try common Urban Atlas layer names
        layer_names = ['UrbanAtlas', 'Urban_Atlas', 'UA2018', 'UA2018_LC', 
                       'LandUse', 'landuse', 'UrbanAtlas_Merged', 'merged_layer']
        
        gdf_ua = None
        for layer_name in layer_names:
            try:
                gdf_ua = gpd.read_file(
                    data_parameters['urbanAtlas_dir'],
                    layer=layer_name,
                    bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
                )
                if not gdf_ua.empty:
                    print(f"  Successfully loaded layer: {layer_name}")
                    break
            except:
                continue
        
        if gdf_ua is None or gdf_ua.empty:
            gdf_ua = gpd.read_file(
                data_parameters['urbanAtlas_dir'],
                bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
            )
            print(f"  Loaded first/default layer")
        
        if gdf_ua.empty:
            print("  No Urban Atlas data found in bounding box")
            return None
            
        # Find the correct column name for land use codes
        code_col = None
        for col in ['code_2018', 'Code_2018', 'code', 'CODE', 'class_code', 
                    'ClassCode', 'LC_CLASS', 'lc_class']:
            if col in gdf_ua.columns:
                code_col = col
                break
        
        if code_col is None:
            print(f"  Available columns: {list(gdf_ua.columns)}")
            raise ValueError(f"Urban Atlas data missing land use code column")
            
        gdf_ua[code_col] = pd.to_numeric(gdf_ua[code_col], errors='coerce').fillna(-1).astype(int)
        gdf_ua['weight'] = gdf_ua[code_col]
        gdf_ua = gdf_ua.to_crs(epsg=epsg)
        gdf_ua = gdf_ua[gdf_ua.is_valid]
        
        print(f"  Urban Atlas: {len(gdf_ua)} features loaded")
        print(f"  Unique land use codes: {sorted(gdf_ua['weight'].unique())}")
        return gdf_ua
        
    except Exception as e:
        print(f"  Urban Atlas processing failed: {str(e)}")
        return None

def process_corine_landcover(data_parameters, bbox_exact, epsg, rows, cols, resol):
    """Process CORINE Land Cover data - secondary source for gap filling"""
    print("Processing CORINE Land Cover data...")
    
    try:
        # CORINE uses EPSG:3035
        bbox_3035 = bbox_transform(3035, epsg, bbox_exact[0], bbox_exact[1], bbox_exact[2], bbox_exact[3])
        
        corine_path = data_parameters.get('corine_dir', '/home/vaithisa/Input_data/U2018_CLC2018_V2020_20u1.gdb')
        
        # Try common CORINE layer names
        layer_names = ['clc_2018', 'CLC_2018', 'clc18', 'CLC18', 'clc', 'CLC']
        
        gdf_clc = None
        for layer_name in layer_names:
            try:
                gdf_clc = gpd.read_file(
                    corine_path,
                    layer=layer_name,
                    bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
                )
                if not gdf_clc.empty:
                    print(f"  Successfully loaded layer: {layer_name}")
                    break
            except:
                continue
        
        if gdf_clc is None or gdf_clc.empty:
            try:
                gdf_clc = gpd.read_file(
                    corine_path,
                    bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
                )
                print("  Loaded first/default layer")
            except:
                pass
        
        if gdf_clc is None or gdf_clc.empty:
            print("  No CORINE data found in bounding box")
            return None
        
        # Find the correct column name
        code_col = None
        for col in ['code_18', 'Code_18', 'CLC_CODE', 'grid_code', 'code', 'CODE']:
            if col in gdf_clc.columns:
                code_col = col
                break
        
        if code_col is None:
            print(f"  Available columns: {list(gdf_clc.columns)}")
            raise ValueError("CORINE data missing land cover code column")
            
        gdf_clc[code_col] = pd.to_numeric(gdf_clc[code_col], errors='coerce').fillna(-1).astype(int)
        gdf_clc['weight'] = gdf_clc[code_col]
        gdf_clc = gdf_clc.to_crs(epsg=epsg)
        gdf_clc = gdf_clc[gdf_clc.is_valid]
        
        print(f"  CORINE: {len(gdf_clc)} features loaded")
        print(f"  Unique land cover codes: {sorted(gdf_clc['weight'].unique())}")
        return gdf_clc
        
    except Exception as e:
        print(f"  CORINE processing failed: {str(e)}")
        return None

def convert_corine_to_5digit(code):
    """Convert CORINE 3-digit code to 5-digit Urban Atlas-like code"""
    mapping = {
        111: 11100, 112: 11200, 121: 12100, 122: 12200, 124: 12400,
        131: 13100, 132: 13200, 133: 13300, 141: 14100, 142: 14200,
        211: 21100, 212: 21200, 213: 21300, 221: 22100, 222: 22200,
        223: 22300, 231: 23100, 241: 24100, 242: 24200, 243: 24300,
        311: 31100, 312: 31200, 313: 31300, 321: 32100, 322: 32200,
        323: 32300, 324: 32400, 331: 33100, 332: 33200, 333: 33300,
        334: 33400, 335: 33500, 411: 41100, 412: 41200, 421: 42100,
        422: 42200, 423: 42300, 511: 51100, 512: 51200, 521: 52100,
        522: 52200, 523: 52300
    }
    return mapping.get(int(code), int(code) * 100)

def create_hybrid_landuse_raster(job_parameters, bbox_exact, epsg, data_parameters, rows, cols):
    """Create hybrid land use raster with exact dimensions"""
    print("\n=== Creating Hybrid Land Use Raster (Urban Atlas + CORINE) ===")
    
    x_min, y_min, x_max, y_max = bbox_exact
    resol = job_parameters['resol']
    
    print(f"  Target grid: {rows} rows x {cols} cols")
    
    # Process both datasets
    gdf_ua = process_urban_atlas(data_parameters, bbox_exact, epsg, rows, cols, resol)
    gdf_clc = process_corine_landcover(data_parameters, bbox_exact, epsg, rows, cols, resol)
    
    if gdf_ua is None and gdf_clc is None:
        raise RuntimeError("Neither Urban Atlas nor CORINE data available!")
    
    ua_raster = 'urban_atlas_temp.tif'
    clc_raster = 'corine_temp.tif'
    output_path = 'clc_proxy.tif'
    
    # Rasterize Urban Atlas
    ua_valid = False
    ua_arr = None
    if gdf_ua is not None and not gdf_ua.empty:
        print("Rasterizing Urban Atlas...")
        try:
            rasterize_clip_shp(job_parameters, ua_raster, gdf_ua, 'weight', bbox_exact, epsg, rows, cols)
            ua_ds = gdal.Open(ua_raster)
            if ua_ds:
                ua_arr = ua_ds.ReadAsArray().astype(np.float32)
                ua_arr = enforce_shape(ua_arr, rows, cols)
                ua_ds = None
                ua_valid = True
                print(f"  Urban Atlas rasterized successfully")
        except Exception as e:
            print(f"  Urban Atlas rasterization failed: {e}")
    
    # Rasterize CORINE
    clc_valid = False
    clc_arr = None
    if gdf_clc is not None and not gdf_clc.empty:
        print("Rasterizing CORINE...")
        try:
            rasterize_clip_shp(job_parameters, clc_raster, gdf_clc, 'weight', bbox_exact, epsg, rows, cols)
            clc_ds = gdal.Open(clc_raster)
            if clc_ds:
                clc_arr = clc_ds.ReadAsArray().astype(np.float32)
                clc_arr = enforce_shape(clc_arr, rows, cols)
                clc_ds = None
                clc_valid = True
                print(f"  CORINE rasterized successfully")
        except Exception as e:
            print(f"  CORINE rasterization failed: {e}")
    
    # Create hybrid
    if ua_valid and clc_valid:
        print("Creating hybrid raster (Urban Atlas prioritized)...")
        
        # Get masks
        ua_valid_mask = (ua_arr > 0) & ~np.isnan(ua_arr)
        clc_valid_mask = (clc_arr > 0) & ~np.isnan(clc_arr)
        
        # Convert CORINE codes to 5-digit
        clc_arr_converted = np.zeros_like(clc_arr)
        for code in np.unique(clc_arr[clc_valid_mask]):
            converted = convert_corine_to_5digit(code)
            mask = (clc_arr == code) & clc_valid_mask
            clc_arr_converted[mask] = converted
        
        # Combine
        hybrid_arr = np.where(ua_valid_mask, ua_arr, 
                              np.where(clc_valid_mask, clc_arr_converted, np.nan))
        
        # Statistics
        ua_pixels = np.sum(ua_valid_mask)
        clc_pixels = np.sum(~ua_valid_mask & clc_valid_mask)
        total_pixels = rows * cols
        
        print(f"  Urban Atlas covers: {ua_pixels} pixels ({100*ua_pixels/total_pixels:.1f}%)")
        print(f"  CORINE fills: {clc_pixels} pixels ({100*clc_pixels/total_pixels:.1f}%)")
        
    elif ua_valid:
        print("Only Urban Atlas available - using as is")
        hybrid_arr = ua_arr
    elif clc_valid:
        print("Only CORINE available - using as is")
        hybrid_arr = clc_arr_converted if 'clc_arr_converted' in dir() else clc_arr
    else:
        raise RuntimeError("No valid data to create hybrid raster")
    
    # Write output
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW'])
    out_ds.SetGeoTransform((x_min, resol, 0, y_max, 0, -resol))
    out_ds.SetProjection(f'EPSG:{epsg}')
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(hybrid_arr)
    out_band.SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None
    
    # Cleanup
    for temp_file in [ua_raster, clc_raster]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"  Hybrid land use raster saved: {output_path}")
    return output_path

def process_nightlight(data_parameters, bbox_exact, epsg, resol, rows, cols):
    print("Processing VIIRS nighttime light data...")
    try:
        nl_output = 'nightlight_proxy.tif'
        if os.path.exists(nl_output):
            os.remove(nl_output)
        
        x_min, y_min, x_max, y_max = bbox_exact
        
        # Use the same pattern as population density with callback
        gdal.Warp(nl_output, data_parameters['viirs_nightlight'],
                xRes=resol, yRes=resol,
                resampleAlg='bilinear',
                format='GTiff',
                dstSRS=f'EPSG:{epsg}',
                outputBounds=(x_min, y_min, x_max, y_max),
                outputBoundsSRS=f'EPSG:{epsg}',
                targetAlignedPixels=True,
                width=cols, height=rows,
                callback=gdal.TermProgress_nocb)  # <-- THIS IS WHAT YOU NEED
        
        # Normalize
        ds = gdal.Open(nl_output, gdal.GA_Update)
        if ds:
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            arr[arr < 0] = 0
            if np.max(arr) > 0:
                arr = arr / np.max(arr)
            band.WriteArray(arr)
            ds.FlushCache()
            ds = None
        
        print("  Nighttime lights raster saved: nightlight_proxy.tif")
        return True
        
    except Exception as e:
        print(f"Error processing nighttime lights: {str(e)}")
        return False

def process_proxy_raster(data_parameters, proxy_name, file_path, bbox_exact, epsg, resol, rows, cols, is_categorical=False):
    """Generic function to process proxy rasters with exact dimensions"""
    print(f"Processing {proxy_name}...")
    output_path = f'{proxy_name}_proxy.tif'
    
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
        
        resample_alg = 'near' if is_categorical else 'bilinear'
        
        gdal.Warp(output_path, file_path,
                 xRes=resol, yRes=resol,
                 resampleAlg=resample_alg,
                 format='GTiff',
                 dstSRS=f'EPSG:{epsg}',
                 outputBounds=bbox_exact,
                 outputBoundsSRS=f'EPSG:{epsg}',
                 targetAlignedPixels=True,
                 width=cols, height=rows,
                 options=['COMPRESS=LZW'],
                 callback=gdal.TermProgress_nocb)
        
        # Verify and enforce shape
        ds = gdal.Open(output_path, gdal.GA_Update)
        if ds:
            arr = ds.ReadAsArray()
            if arr.shape != (rows, cols):
                print(f"  Shape mismatch, reshaping from {arr.shape} to ({rows}, {cols})")
                arr_corrected = enforce_shape(arr, rows, cols)
                ds.GetRasterBand(1).WriteArray(arr_corrected)
            ds = None
        
        print(f"  {proxy_name} raster saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  Error processing {proxy_name}: {str(e)}")
        return None

def downscaling_proxies(data_parameters, job_parameters, bbox, epsg):
    """Prepare all downscaling proxies with consistent grid geometry"""
    print('\n=== Preparing Proxies ===')
    gdal.UseExceptions()
    
    x_min, y_min, x_max, y_max = bbox
    resol = job_parameters['resol']
    
    # CRITICAL: Compute master grid ONCE and use everywhere
    cols = int(round((x_max - x_min) / resol))
    rows = int(round((y_max - y_min) / resol))
    
    cols = max(1, cols)
    rows = max(1, rows)
    
    x_max_exact = x_min + (cols * resol)
    y_max_exact = y_min + (rows * resol)
    
    bbox_exact = [x_min, y_min, x_max_exact, y_max_exact]
    
    print(f"\n  Master grid configuration:")
    print(f"    Original BBOX: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    print(f"    Exact BBOX:    x=[{x_min:.1f}, {x_max_exact:.1f}], y=[{y_min:.1f}, {y_max_exact:.1f}]")
    print(f"    Resolution: {resol} m")
    print(f"    Output grid: {rows} rows x {cols} cols ({rows*cols:,} pixels)")
    
    try:
        epsg = int(epsg)
    except ValueError:
        raise ValueError(f"Invalid EPSG code: {epsg}")
    

    print("\n1. Processing land use data (hybrid: Urban Atlas + CORINE)...")
    create_hybrid_landuse_raster(job_parameters, bbox_exact, epsg, data_parameters, rows, cols)

    # 2. Process population density
    print("\n2. Processing population density...")
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
        print("  Population raster saved: pop_proxy.tif")
    except Exception as e:
        raise RuntimeError(f"Population processing failed: {str(e)}")

    # 3. Process nighttime lights
    print("\n3. Processing nighttime lights...")
    try:
        nl_output = 'nightlight_proxy.tif'
        if os.path.exists(nl_output):
            os.remove(nl_output)
        
        x_min, y_min, x_max, y_max = bbox_exact
        
        # Use direct gdal.Warp with callback (same as population density)
        gdal.Warp(nl_output, data_parameters['viirs_nightlight'],
                xRes=resol, yRes=resol,
                resampleAlg='bilinear',
                format='GTiff',
                dstSRS=f'EPSG:{epsg}',
                outputBounds=(x_min, y_min, x_max, y_max),
                outputBoundsSRS=f'EPSG:{epsg}',
                targetAlignedPixels=True,
                width=cols, height=rows,
                callback=gdal.TermProgress_nocb)  
        
        # Post-process: normalize
        ds = gdal.Open(nl_output, gdal.GA_Update)
        if ds:
            arr = ds.ReadAsArray()
            arr[arr < 0] = 0
            if np.max(arr) > 0:
                arr = arr / np.max(arr)
            ds.GetRasterBand(1).WriteArray(arr)
            ds.FlushCache()
            ds = None
        
        print("  Nighttime lights raster saved: nightlight_proxy.tif")
        
    except Exception as e:
        print(f"  Error processing nighttime lights: {str(e)}")
        print("  Using neutral nighttime light proxy")
        neutral = np.ones((rows, cols), dtype=np.float32) * 0.5
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create('nightlight_proxy.tif', cols, rows, 1, gdal.GDT_Float32,
                        options=['COMPRESS=LZW', 'TILED=YES'])
        ds.SetGeoTransform((x_min, resol, 0, y_max_exact, 0, -resol))
        ds.SetProjection(f'EPSG:{epsg}')
        band = ds.GetRasterBand(1)
        band.WriteArray(neutral)
        band.SetNoDataValue(np.nan)
        band.FlushCache()
        ds = None

    # 4. Process OSM roads

    print("\n4. Processing OSM roads...")
    gdf_roads = prepare_osm_roads(bbox, epsg, bbox_exact)
    if not gdf_roads.empty:
        rasterize_line_shp('osm_proxy.tif', gdf_roads, 'weight', bbox_exact, epsg, resol, rows, cols)
        print("  OSM roads rasterized successfully")
    else:
        print("  No OSM roads found, using neutral proxy")
        neutral = np.ones((rows, cols), dtype=np.float32)
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create('osm_proxy.tif', cols, rows, 1, gdal.GDT_Float32,
                          options=['COMPRESS=LZW'])
        ds.SetGeoTransform((x_min, resol, 0, y_max_exact, 0, -resol))
        ds.SetProjection(f'EPSG:{epsg}')
        band = ds.GetRasterBand(1)
        band.WriteArray(neutral)
        band.SetNoDataValue(np.nan)
        band.FlushCache()
        ds = None
    # ===============================================================
    # Final verification
    # ===============================================================
    print("\n" + "="*60)
    print("PROXY PREPARATION COMPLETED")
    print("="*60)
    print(f"  All proxies aligned to grid: {rows} rows x {cols} cols")
    print(f"  Resolution: {resol} m")
    print(f"  CRS: EPSG:{epsg}")
    
    proxy_files = ['osm_proxy.tif', 'clc_proxy.tif', 'pop_proxy.tif', 'nightlight_proxy.tif']
    for pf in proxy_files:
        if os.path.exists(pf):
            ds = gdal.Open(pf)
            if ds:
                print(f"    - {pf}: {ds.RasterYSize} rows x {ds.RasterXSize} cols")
                ds = None
            else:
                print(f"    - {pf}: EXISTS BUT CANNOT OPEN")
        else:
            print(f"    - {pf}: MISSING")
    
    print("="*60 + "\n")
