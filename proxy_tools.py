"""
https://www.zensus2011.de/DE/Home/Aktuelles/DemografischeGrunddaten.html?nn=559100#Gitter
https://gdz.bkg.bund.de/index.php/default/inspire/sonstige-inspire-themen/geographische-gitter-fur-deutschland-in-lambert-projektion-geogitter-inspire.html
https://ghsl.jrc.ec.europa.eu/download.php?ds=pop
"""
import os
import osmnx as ox
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from geocube.api.core import make_geocube

def bbox_transform(in_crs, out_crs, cell_minx, cell_miny, cell_maxx, cell_maxy, order_xy = True):
    transformer = Transformer.from_crs(out_crs, in_crs, always_xy = order_xy)
    xmin, ymin = transformer.transform(cell_minx, cell_miny)
    xmax, ymax = transformer.transform(cell_maxx, cell_maxy)
    bbox_out = [xmin, ymin, xmax, ymax]

    return bbox_out

def rasterize_clip_shp(job_parameters, out_fn, gdf, lyr, bbox, epsg):
    tmp_fn = 'tmp_ras.tif'
    out_grid = make_geocube(vector_data=gdf,
            measurements=[lyr],
            resolution=(-1*job_parameters['resol'], job_parameters['resol']),
            output_crs=int(epsg))
    rast_mem = out_grid.rio.to_raster(tmp_fn)

    ds = gdal.Open(tmp_fn)
    gdal.Warp(out_fn, ds,
            xRes=job_parameters['resol'], yRes=job_parameters['resol'],
            resampleAlg='mode', format='GTiff',
            dstSRS ='EPSG:'+ epsg,
            outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]),
            outputBoundsSRS='EPSG:'+ epsg)
    ds = None
    os.remove(tmp_fn)

def road_type_code(row, ttable):
    if type(row['highway']) == list:
        osm_code = osm_code = row['highway'][0]
    else:
        osm_code = row['highway']
    
    if osm_code in ttable:
        return ttable[osm_code]
    elif osm_code not in ttable:
        return 9

def downscaling_proxies(data_parameters, job_parameters, bbox, epsg):
    print('Preparing Proxies...')
    # Transform bbox
    bbox_3035 = bbox_transform(3035, int(epsg), bbox[0], bbox[1], bbox[2], bbox[3])
    bbox_4326 = bbox_transform(4326, int(epsg), bbox[0], bbox[1], bbox[2], bbox[3])

    ## CORINE data - rasterize
    gdf_clc = gpd.read_file(data_parameters['corine_dir'], bbox=(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3]))
    gdf_clc['Code_18'] = gdf_clc['Code_18'].astype('int')
    gdf_clc = gdf_clc.to_crs(int(epsg))
    rasterize_clip_shp(job_parameters, 'clc_proxy.tif', gdf_clc, 'Code_18', bbox, epsg)

    ## Population data - resample & clip
    gdal.Warp('pop_proxy.tif', data_parameters['popul_dir'], xRes=job_parameters['resol'], yRes=job_parameters['resol'],
            resampleAlg='bilinear', format='GTiff',
            dstSRS='EPSG:'+ epsg,
            outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]),
            outputBoundsSRS='EPSG:'+ epsg)

    ## OSM data - retrieve & rasterize
    tags = [ 'name', 'highway', 'service', 'width', 'est_width', 'junction', 'surface']
    ox.settings.useful_tags_way = tags
    cust_fil = '["highway"~"motorway|motorway_link|primary|primary_link|secondary|secondary_link|trunk|trunk_link|tertiary|tertiary_link|residential|living_street"]'
    ways = ox.graph_from_bbox(bbox_4326[3], bbox_4326[1], bbox_4326[2], bbox_4326[0], network_type='drive',
            simplify=True, retain_all=True, truncate_by_edge=True, clean_periphery=True, custom_filter = cust_fil)
    # Remove OSM nodes not connected to intersections
    ways_project = ox.project_graph(ways)
    ways_int = ox.consolidate_intersections(ways_project, tolerance=10, rebuild_graph=True, dead_ends=True, reconnect_edges=True)
    gdf_streets = ox.graph_to_gdfs(ways_int, nodes=False)
    # Convert road type to code
    typeRoad = {'"motorway"': 1, '"motorway_link"':2, '"primary"':3, '"primary_link"':4, '"secondary"':5, '"secondary_link"':6,
            '"trunk"': 7, '"trunk_link"': 8, '"tertiary"' : 9, '"tertiary_link"': 10, '"residential"': 11, '"living_street"': 12}

    gdf_streets['code_type'] = gdf_streets.apply(lambda row:road_type_code(row, typeRoad), axis=1)
    gdf_streets['code_type'] = gdf_streets['code_type'].astype('int')
    gdf_streets = gdf_streets.to_crs(int(epsg))
    # Rasterize OSM
    rasterize_clip_shp(job_parameters, 'osm_proxy.tif', gdf_streets, 'code_type', bbox, epsg)

