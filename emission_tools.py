"""

"""
import os
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

## Downscaling if emissions

def update_patch_with_clc(single_patch_empty, single_patch_clc, greta_array, i, j, clc_code):
    if len(clc_code)==1:
        clc_patch = np.where(single_patch_clc == clc_code[0])
        clc_count = np.count_nonzero(single_patch_clc == clc_code[0])

    elif len(clc_code)==2:
        clc_patch = np.where((single_patch_clc == clc_code[0]) | (single_patch_clc == clc_code[1]))
        clc_count = np.count_nonzero(single_patch_clc == clc_code[0])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[1])

    elif len(clc_code)==3:
        clc_patch = np.where((single_patch_clc == clc_code[0]) | (single_patch_clc == clc_code[1]) | (single_patch_clc == clc_code[2]))
        clc_count = np.count_nonzero(single_patch_clc == clc_code[0])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[1])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[2])

    elif len(clc_code)==7:
        clc_patch = np.where((single_patch_clc == clc_code[0]) | (single_patch_clc == clc_code[1]) | (single_patch_clc == clc_code[2]) | (single_patch_clc == clc_code[3] | (single_patch_clc == clc_code[4]) | (single_patch_clc == clc_code[5]) | (single_patch_clc == clc_code[6])))
        clc_count = np.count_nonzero(single_patch_clc == clc_code[0])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[1])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[2])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[3])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[4])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[5])
        clc_count = clc_count + np.count_nonzero(single_patch_clc == clc_code[6])
    
    greta_value_clc = greta_array[i][j]
    if clc_count != 0 and greta_value_clc != 0:
        single_patch_empty[clc_patch] = greta_value_clc/clc_count

    elif greta_value_clc != 0 and clc_count == 0: # equally distribute emissions through grid
        clc_patch = np.where(single_patch_clc > 0)
        single_patch_empty[clc_patch] = greta_value_clc/100

    elif greta_value_clc == 0 and clc_count == 0:
        single_patch_empty[clc_patch] = 0

    single_patch_update = single_patch_empty

    return single_patch_update

def update_patch_with_population(single_patch_empty, single_patch_pop, greta_array, i, j):
    pop_patch = np.where(single_patch_pop != -1)
    pop_count = np.count_nonzero(single_patch_pop != -1)

    greta_value_pop = greta_array[i][j]
    if pop_count != 0 and greta_value_pop !=0:
        single_patch_empty[pop_patch] = greta_value_pop/pop_count

    else:
        single_patch_empty[pop_patch] = 0

    single_patch_update = single_patch_empty

    return single_patch_update

def update_patch_with_osm(single_patch_empty, single_patch_osm, greta_array, i, j):
    osm_patch = np.where(single_patch_osm > 0)

    motorway    = np.count_nonzero((single_patch_osm == 1) | (single_patch_osm == 2))  # motorway + motorway_link
    primary     = np.count_nonzero((single_patch_osm == 3) | (single_patch_osm == 4))  # primary + primary_link
    secondary   = np.count_nonzero((single_patch_osm == 5) | (single_patch_osm == 6))  # secondary + secondary_link
    trunk       = np.count_nonzero((single_patch_osm == 7) | (single_patch_osm == 8))  # trunk # trunk_link
    tertiary    = np.count_nonzero((single_patch_osm == 9) | (single_patch_osm == 10)) # tertiary + tertiary_link
    residential = np.count_nonzero(single_patch_osm == 11)                             # residential
    living      = np.count_nonzero(single_patch_osm == 12)                             # living_street

    greta_value_osm = greta_array[i][j]
    weighted_value = (motorway * 0.4 + primary * 0.3 + secondary * 0.2 + tertiary * 0.25 + trunk * 0.25 + residential * 0.25 + living * 0.25)
    #weighted_value = (motorway * 1.0 + primary * 1.0 + secondary * 1.0 + tertiary * 1.0 + trunk * 1.0 + residential * 1.0 + living * 1.0)

    if weighted_value != 0 and greta_value_osm != 0:
         single_patch_empty[osm_patch] = greta_value_osm/weighted_value
    else:
         single_patch_empty[osm_patch] = 0
    single_patch_update = single_patch_empty

    return single_patch_update

##------------------------------------------------------------------------
def downscale_emissions(job_parameters, sectors, gdf_grid, bbox, epsg):
    print('Downscaling Emissions...')
    # Read proxies
    clc_ds  = gdal.Open('clc_proxy.tif')
    clc_arr = clc_ds.ReadAsArray().astype(int)
    clc_trans = clc_ds.GetGeoTransform()
    clc_wkt =  clc_ds.GetProjection()

    osm_ds = gdal.Open('osm_proxy.tif')
    osm_arr = osm_ds.ReadAsArray().astype(int)

    pop_ds  = gdal.Open('pop_proxy.tif')
    pop_arr = pop_ds.ReadAsArray().astype(int)

    ## Downscale by species and sector
    patch_size = 100
    main_sectors = [item[0] for item in sectors]
    for spec in job_parameters['species']:
        empty_arr = np.zeros((clc_arr.shape[0], clc_arr.shape[1], len(sectors)))
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
            emis_ds  = gdal.Open(out_raster)
            emis_arr = emis_ds.ReadAsArray()
            os.remove(tmp_raster)

            # Downscale using patches
            sector_arr    = empty_arr[:,:,nsec]
            empty_patches = patchify(sector_arr, (patch_size, patch_size), step = patch_size)
            clc_patches   = patchify(clc_arr, (patch_size, patch_size), step = patch_size)
            pop_patches   = patchify(pop_arr, (patch_size, patch_size), step = patch_size)
            osm_patches   = patchify(osm_arr, (patch_size, patch_size), step = patch_size)
            for i in range(empty_patches.shape[0]):
                for j in range(empty_patches.shape[1]):
                    single_patch_empty = empty_patches[i,j,:,:]
                    single_patch_clc   = clc_patches[i,j,:,:]
                    single_patch_pop   = pop_patches[i,j,:,:]
                    single_patch_osm   = osm_patches[i,j,:,:]

                    if sec == 'A_PublicPower':
                        code = [121]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'B_Industry':
                        code = [121, 131, 133]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'C_OtherStationaryComb':
                        empty_patches[i,j,:,:] = update_patch_with_population(single_patch_empty, single_patch_pop, emis_arr, i, j)
                    elif sec == 'D_Fugitives':
                        code = [121, 131]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'E_Solvents':
                        empty_patches[i,j,:,:] = update_patch_with_population(single_patch_empty, single_patch_pop, emis_arr, i, j)
                    elif sec == 'F_RoadTransport':
                        empty_patches[i,j,:,:] = update_patch_with_osm(single_patch_empty, single_patch_osm, emis_arr, i, j)
                    elif sec == 'G_Shipping':
                        code = [123]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'H_Aviation':
                        code = [124]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'I_OffRoad':
                        code = [122, 244]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'J_Waste':
                        code = [132]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'K_AgriLivestock':
                        code = [231]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)
                    elif sec == 'L_AgriOther':
                        code = [211, 212, 213, 221, 222, 223, 241]
                        empty_patches[i,j,:,:] = update_patch_with_clc(single_patch_empty, single_patch_clc, emis_arr, i, j, code)

            # Unpatchify
            if sec != 'SumAllSectors':
               empty_arr[:,:,nsec] = unpatchify(empty_patches, sector_arr.shape)
            elif sec == 'SumAllSectors' and nsec+1 == emis_arr.shape[0]:
                empty_arr[:,:,nsec] = np.sum(empty_arr, axis =2)
            updated_arr = empty_arr

            emis_ds = None
            os.remove(out_raster)

        # Final output: raster & clip to input domain
        driver = gdal.GetDriverByName('MEM')
        dst_ds = driver.Create('', clc_ds.GetRasterBand(1).XSize,
                clc_ds.GetRasterBand(1).YSize, updated_arr.shape[2], gdal.GDT_Float64)
        dst_ds.SetGeoTransform(clc_trans)
        dst_ds.SetProjection(clc_wkt)
        for nband in range(0, len(main_sectors)):
            dst_ds.GetRasterBand(nband+1).WriteArray(updated_arr[:,:,nband])

        output_fn = os.path.join(job_parameters['job_path'], 'emission_'+ spec +'_downscaled.tif')
        gdal.Warp(output_fn, dst_ds, xRes=job_parameters['resol'], yRes=job_parameters['resol'], resampleAlg=None, format="GTiff", 
            outputBounds=(job_parameters['min_lon'], job_parameters['min_lat'], job_parameters['max_lon'], job_parameters['max_lat']))

        rds = gdal.Open(output_fn, gdal.GA_Update)
        for band in main_sectors:
            idx = main_sectors.index(band)+1
            rd = rds.GetRasterBand(idx)
            rd.SetDescription(band)
        rds = None
        dst_ds = None
    
    # Tidy-up temporay files
    clc_ds = None
    osm_ds = None
    pop_ds = None
    os.remove('clc_proxy.tif')
    os.remove('osm_proxy.tif')
    os.remove('pop_proxy.tif')
