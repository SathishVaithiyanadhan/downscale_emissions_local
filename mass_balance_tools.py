
import numpy as np
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
import os
import fiona
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd

def process_species(args):
    """Parallel processing function for each species"""
    spec, job_params, data_params, sectors, gdf_grid = args
    
    print(f"\nProcessing species: {spec}")
    spec_upper = spec.upper()
    input_masses = []
    
    # Calculate input masses for each sector
    for sec_idx, sec in enumerate(sectors[:13]):
        sector_name = sec[0]
        nfr_codes = sec[1:]
        total_mass = 0.0
        
        for nfr in nfr_codes:
            col_name = f"E_{nfr}_{spec_upper}"
            if col_name in gdf_grid.columns:
                emissions = gdf_grid[col_name] * gdf_grid['area_m2']
                total_mass += emissions.sum()
        
        input_masses.append(total_mass)
        print(f"Input mass for {sector_name}_{spec}: {total_mass:.6f} kg")

    # Process downscaled GeoTIFF
    tiff_path = os.path.join(job_params['job_path'], f'emission_{spec}_combined.tif')
    if not os.path.exists(tiff_path):
        print(f"Warning: Downscaled file not found: {tiff_path}")
        return None

    try:
        ds = gdal.Open(tiff_path, gdal.GA_Update)
        if ds is None:
            print(f"Error: Could not open {tiff_path}")
            return None

        # Calculate output masses
        transform = ds.GetGeoTransform()
        pixel_area = abs(transform[1] * transform[5])
        output_masses = []
        
        for i in range(1, 14):  # Bands 1-13
            band = ds.GetRasterBand(i)
            arr = band.ReadAsArray()
            output_mass = np.nansum(arr) * pixel_area
            output_masses.append(output_mass)

        # Apply scaling factors
        for i in range(13):
            band_idx = i + 1
            in_mass = input_masses[i]
            out_mass = output_masses[i]
            
            if out_mass > 0 and in_mass > 0:
                scaling_factor = in_mass / out_mass
                band = ds.GetRasterBand(band_idx)
                arr = band.ReadAsArray()
                band.WriteArray(arr * scaling_factor)

        # Update metadata
        metadata = ds.GetMetadata()
        metadata[f'scaling_{spec}'] = str({
            sectors[i][0]: {
                'input_mass': input_masses[i],
                'output_mass': output_masses[i],
                'scaling_factor': input_masses[i]/output_masses[i] if output_masses[i] > 0 else 1.0
            }
            for i in range(13)
        })
        ds.SetMetadata(metadata)
        ds = None
        
        return (spec, input_masses, output_masses)
    
    except Exception as e:
        print(f"Error processing {spec}: {str(e)}")
        if 'ds' in locals():
            ds = None
        return None

def calculate_mass_balance(job_parameters, data_parameters, sectors, species):
    """Parallel mass balance calculation"""
    print("\n=== Starting Parallel Mass Balance Correction ===")
    
    # Create bbox polygon
    bbox_poly = gpd.GeoDataFrame(geometry=[box(
        job_parameters['min_lon'],
        job_parameters['min_lat'],
        job_parameters['max_lon'],
        job_parameters['max_lat']
    )], crs=f"EPSG:{job_parameters['epsg_code']}")

    # Load GRETA data from all layers
    print("\nLoading GRETA data from all layers...")
    gdf_list = []
    for layer in fiona.listlayers(data_parameters['emiss_dir']):
        try:
            gdf = gpd.read_file(data_parameters['emiss_dir'], layer=layer)
            gdf = gdf.to_crs(epsg=job_parameters['epsg_code'])
            gdf = gdf[gdf.intersects(bbox_poly.geometry[0])].copy()
            if not gdf.empty:
                gdf_list.append(gdf)
        except Exception as e:
            print(f"Error loading layer {layer}: {str(e)}")

    if not gdf_list:
        print("Error: No data found within the bbox")
        return

    gdf_grid = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    gdf_grid['area_m2'] = gdf_grid.geometry.area

    # Prepare parallel processing
    num_workers = min(len(species), cpu_count() - 1 if cpu_count() > 1 else 1)
    print(f"\nUsing {num_workers} parallel workers for {len(species)} species")
    
    # Create argument tuples for parallel processing
    args_list = [(spec, job_parameters, data_parameters, sectors, gdf_grid) 
                for spec in species]

    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_species, args_list)

    # Print summary
    print("\n=== Mass Balance Summary ===")
    for result in results:
        if result:
            spec, input_masses, output_masses = result
            print(f"\nSpecies: {spec}")
            for i in range(13):
                sector = sectors[i][0]
                print(f"{sector}: Input={input_masses[i]:.2f}kg, Output={output_masses[i]:.2f}kg, Ratio={input_masses[i]/output_masses[i]:.4f}")

    print("\n=== Parallel Mass Balance Correction Completed ===")