"""

"""
import os
import netCDF4
import numpy as np
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from patchify import patchify, unpatchify
from geocube.api.core import make_geocube

def bbox_transform(in_crs, out_crs, cell_minx, cell_miny, cell_maxx, cell_maxy, order_xy = True):
    transformer = Transformer.from_crs(out_crs, in_crs, always_xy = order_xy)
    xmin, ymin = transformer.transform(cell_minx, cell_miny)
    xmax, ymax = transformer.transform(cell_maxx, cell_maxy)
    bbox_out = [xmin, ymin, xmax, ymax]

    return bbox_out

def nfr_to_gnfr(job_parameters, gdf_data, sectors):
    coln = list(gdf_data.columns)
    src_type= [item[0] for item in sectors]
    col_sum = []
    for spec in job_parameters['species']:
        for sec in src_type:
            sidx = src_type.index(sec)
            subsec = sectors[sidx][1:]
            gfd = []
            for n in subsec:
                gf =  'E_'+ n +'_' + spec.upper()
                if gf in coln:
                    gfd.append(gf)
                elif gf not in coln:
                    continue
            # Sum columns for sector
            gdf_data[sec +'_'+ spec] = gdf_data[gfd].sum(axis=1)
            col_sum.append(sec +'_'+ spec)
    # Calcualte actual pm10
    if 'pm10' in job_parameters['species']:
        for sec in src_type:
            gdf_data[sec + '_pm10'] = gdf_data[sec + '_pm10'] + gdf_data[sec + '_pm2_5']
    # Copy selected columns for point and grid
    if 'plant_id_left' in list(gdf_data.columns):
        col_out = ['plant_id_left', 'plant_name_left', 'prtr_code_left', 'prtr_sector_left', 'maingroup_left',
                'emission_height_left', 'wgs84_x_left', 'wgs84_y_left', 'geometry'] + col_sum
        gdf_out = gdf_data[col_out].copy()

    elif 'ID_RASTER_left' in list(gdf_data.columns):
        col_out = ['ID_RASTER_left', 'geometry'] + col_sum
        gdf_out = gdf_data[col_out].copy()
    # Remove duplicates
    gdf_out = gdf_out.drop_duplicates()
    return gdf_out

def prep_greta_data(data_parameters, job_parameters, sectors):
    bbox_greta = bbox_transform(25832, job_parameters['epsg_code'],
            job_parameters['min_lon'], job_parameters['min_lat'],
            job_parameters['max_lon'], job_parameters['max_lat'])

    # Point soures - read and merge
    prtr1_greta = gpd.read_file(data_parameters['emiss_dir'], layer=2, bbox=(bbox_greta[0], bbox_greta[1], bbox_greta[2], bbox_greta[3]))
    prtr2_greta = gpd.read_file(data_parameters['emiss_dir'], layer=3, bbox=(bbox_greta[0], bbox_greta[1], bbox_greta[2], bbox_greta[3]))
    prtr_greta = gpd.sjoin(prtr1_greta, prtr2_greta, how="left")

    # Gridded data - read and merge
    rast1_greta = gpd.read_file(data_parameters['emiss_dir'], layer=0, bbox=(bbox_greta[0], bbox_greta[1], bbox_greta[2], bbox_greta[3]))
    rast2_greta = gpd.read_file(data_parameters['emiss_dir'], layer=1, bbox=(bbox_greta[0], bbox_greta[1], bbox_greta[2], bbox_greta[3]))
    rast_greta = gpd.sjoin(rast1_greta, rast2_greta, how="left")

    ## Convert emissions sectors (NFR to GNFR)
    # Point sources
    gdf_prtr = nfr_to_gnfr(job_parameters, prtr_greta, sectors)
    gdf_prtr.drop('geometry',axis=1).to_csv(os.path.join(job_parameters['job_path'], 'point_sources.csv'))

    # Gridded emissions
    gdf_grid = nfr_to_gnfr(job_parameters, rast_greta, sectors)
    # Get extent of GRETA dataframe
    grid_ext = gdf_grid.total_bounds
    bbox_grid = [grid_ext[0], grid_ext[1], grid_ext[2], grid_ext[3]]
    bbox_epsg = str(gdf_grid.crs)
    bbox_epsg = bbox_epsg[5:]

    return gdf_grid, bbox_grid, bbox_epsg

#def rasterize_clip_gdf(job_parameters, spec, sec, nsec):
#    tmp_raster = 'tmp_emission'+str(nsec)+'.tif'
#    out_raster = 'emission'+str(nsec)+'.tif'
#    out_grid = make_geocube(vector_data=gdf_grid,
#            measurements=[sec +'_'+ spec],
#            resolution=(-1*1000, 1000),
#            output_crs=job_parameters['epsg_code'])
#    rast_mem = out_grid.rio.to_raster(tmp_raster)

    #ds = gda.Open(tmp_raster)
    #gdal.Warp(out_raster, ds,
    #        xRes=job_parameters['resol'], yRes=job_parameters['resol'],
    #        resampleAlg='mode', format='GTiff',
    #        dstSRS ='EPSG:'+str(job_parameters['epsg_code']),
    #        outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]),
    #        outputBoundsSRS='EPSG:'+ str(job_parameters['epsg_code']))
    #ds = None
    #os.remove(tmp_fn)


def downscale_emissions(job_parameters, sectors, gdf_grid, bbox, epsg):
    print('Downscaling Emissions...')
    # Read proxies
    clc_ds  = gdal.Open('clc_proxy.tif')
    clc_arr = clc_ds.ReadAsArray().astype(int)

    osm_ds = gdal.Open('osm_proxy.tif')
    osm_arr = osm_ds.ReadAsArray().astype(int)

    pop_ds  = gdal.Open('pop_proxy.tif')
    pop_arr = pop_ds.ReadAsArray().astype(int)

    #
    patch_size = 100
    main_sectors = [item[0] for item in sectors]
    #
    for spec in job_parameters['species']:
        empty_arr = np.zeros((clc_arr.shape[0], clc_arr.shape[1], len(sectors)))
        # By sector
        for sec in main_sectors:
            nsec = main_sectors.index(sec)
            # GRETA emission geodataframe to raster
            tmp_raster = 'tmp_emission'+str(nsec)+'.tif'
            out_raster = 'emission'+str(nsec)+'.tif'
            out_grid = make_geocube(vector_data=gdf_grid,
                    measurements=[sec +'_'+ spec],
                    resolution=(-1*1000, 1000),
                    output_crs=int(epsg))
            rast_mem = out_grid.rio.to_raster(tmp_raster)

            ds = gdal.Open(tmp_raster)
            gdal.Warp(out_raster, ds,
                    xRes=job_parameters['resol'], yRes=job_parameters['resol'],
                    resampleAlg='mode', format='GTiff',
                    dstSRS ='EPSG:'+epsg,
                    outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    outputBoundsSRS='EPSG:'+ epsg)
            ds = None
            os.remove(tmp_raster)

            # Downscale using patches
            sector_arr    = empty_array[:,:,nsec]
            empty_patches = patchify(sector_arr, (patch_size, patch_size), step = patch_size)
            clc_patches   = patchify(clc_arr, (patch_size, patch_size), step = patch_size)
            pop_patches   = patchify(pop_arr, (patch_size, patch_size), step = patch_size)
            osm_patches   = patchify(osm_arr, (patch_size, patch_size), step = patch_size)
            for i in range(empty_patches.shape[0]):
                for j in range(empty_patches.shape[1]):
                    single_patch_empty = empty_patches[i,j,:,:]
                    single_patch_clc = clc_patches[i,j,:,:]
                    single_patch_pop = pop_patches[i,j,:,:]
                    single_patch_osm = osm_patches[i,j,:,:]

        ## Wrtite array to netCDF file for each species
        #nc_out = netCDF4.Dataset(out_driver,'w',format='NETCDF4')
        # Attributes
        # Dimensions
        # Variables
        #nc_out.close()
    

    # Tidy-up temporay files
    clc_ds = None
    osm_ds = None
    pop_ds = None
    #os.remove('clc_proxy.tif')
    #os.remove('osm_proxy.tif')
    #os.remove('pop_proxy.tif')
