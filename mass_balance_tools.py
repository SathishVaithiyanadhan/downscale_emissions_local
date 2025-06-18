#with NOx and PM (fine+coarse) speciation
import numpy as np
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
import os
import fiona
from multiprocessing import Pool, cpu_count
import pandas as pd
import rasterio
import time
from pathlib import Path
import tempfile

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
    tiff_path = os.path.join(job_params['job_path'], f'emission_{spec}_yearly.tif')
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

def process_yearly_emissions(input_path, output_no_path, output_no2_path):
    """Process yearly multi-sector NOx emissions to calculate NO and NO₂ emissions."""
    start_time = time.time()
    
    # Define sector names (excluding 'SumAllSectors' for processing)
    all_sector_names = [
        'A_PublicPower', 'B_Industry', 'C_OtherStationaryComb', 'D_Fugitives',
        'E_Solvents', 'F_RoadTransport', 'G_Shipping', 'H_Aviation',
        'I_OffRoad', 'J_Waste', 'K_AgriLivestock', 'L_AgriOther', 'SumAllSectors'
    ]
    processing_sectors = all_sector_names[:-1]  # Exclude 'SumAllSectors'
    
    def get_no2_fraction(sector):
        """Returns annual average NO₂ fraction of NOx for given sector."""
        if sector == 'F_RoadTransport':
            return 0.20  # Annual average for road transport
        elif sector in ['G_Shipping', 'H_Aviation']:
            return 0.20  # Shipping and aviation
        elif sector == 'I_OffRoad':
            return 0.15  # Off-road mobile
        elif sector in ['A_PublicPower', 'B_Industry', 'C_OtherStationaryComb']:
            return 0.08  # Stationary combustion
        else:
            return 0.0   # Other sectors (negligible NO₂)

    with rasterio.open(input_path) as src:
        total_bands = src.count
        if total_bands != len(all_sector_names):
            raise ValueError(f"Expected {len(all_sector_names)} bands for yearly data, got {total_bands}")
        
        print(f"Processing yearly data with {len(all_sector_names)} sectors")
        
        # Get optimal block size (multiple of 16)
        blockxsize = min(256, src.width // 16 * 16)
        blockysize = min(256, src.height // 16 * 16)
        
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            count=total_bands,
            nodata=src.nodata if src.nodata is not None else -9999.0,
            compress='deflate',
            tiled=True,
            blockxsize=blockxsize,
            blockysize=blockysize
        )
        
        try:
            with rasterio.open(output_no_path, 'w', **profile) as no_dst, \
                 rasterio.open(output_no2_path, 'w', **profile) as no2_dst:
                
                # Copy all metadata including band descriptions
                no_dst.update_tags(**src.tags())
                no2_dst.update_tags(**src.tags())
                
                # Initialize arrays for totals
                annual_total_no = np.zeros(src.shape, dtype=np.float32)
                annual_total_no2 = np.zeros(src.shape, dtype=np.float32)
                
                # Process each sector (excluding 'SumAllSectors')
                for sector_idx, sector in enumerate(processing_sectors):
                    nox = src.read(sector_idx + 1)
                    
                    # Split NOx into NO and NO₂ using annual fractions
                    no2_frac = get_no2_fraction(sector)
                    no = nox * (1 - no2_frac)
                    no2 = nox * no2_frac
                    
                    # Write sector bands
                    no_dst.write(no, sector_idx + 1)
                    no2_dst.write(no2, sector_idx + 1)
                    
                    # Set band descriptions
                    band_desc = src.descriptions[sector_idx]
                    no_dst.set_band_description(sector_idx + 1, band_desc)
                    no2_dst.set_band_description(sector_idx + 1, band_desc)
                    
                    # Accumulate for total calculation
                    annual_total_no += no
                    annual_total_no2 += no2
                
                # Write recalculated totals (ignore input value)
                total_band_idx = len(processing_sectors)
                no_dst.write(annual_total_no, total_band_idx + 1)
                no2_dst.write(annual_total_no2, total_band_idx + 1)
                
                # Copy description for SumAllSectors band
                total_band_desc = src.descriptions[total_band_idx]
                no_dst.set_band_description(total_band_idx + 1, total_band_desc)
                no2_dst.set_band_description(total_band_idx + 1, total_band_desc)
                
                print("Sector processing completed")
        
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            # Clean up potentially corrupted files
            if os.path.exists(output_no_path):
                os.remove(output_no_path)
            if os.path.exists(output_no2_path):
                os.remove(output_no2_path)
            raise
    
    print(f"\nProcessing completed in {time.time()-start_time:.2f} seconds")
    print(f"Annual NO emissions saved to: {output_no_path}")
    print(f"Annual NO₂ emissions saved to: {output_no2_path}")

def process_pm_speciation(pm10_path, pm2_5_path, speciation_file, country_code, year, output_dir):
    """
    Full PM processing pipeline:
    1. Speciate PM2.5 (fine)
    2. Calculate PM coarse (PM10 - PM2.5) and speciate
    3. Combine fine and coarse species
    """
    start_time = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load PM speciation profiles from Excel
    try:
        # Load fine and coarse profiles
        pm_fine = pd.read_excel(speciation_file, sheet_name='fine')
        pm_coarse = pd.read_excel(speciation_file, sheet_name='coarse')
        
        # Filter for the specific country and year
        pm_fine = pm_fine[(pm_fine['ISO3'] == country_code) & (pm_fine['Year'] == year)]
        pm_coarse = pm_coarse[(pm_coarse['ISO3'] == country_code) & (pm_coarse['Year'] == year)]
        
        if pm_fine.empty or pm_coarse.empty:
            raise ValueError(f"No data found for country {country_code} and year {year}")
        
        # Get species names and columns
        fine_species_cols = ['EC_fine', 'OC_fine', 'SO4_fine', 'Na_fine', 'OthMin_fine']
        fine_species_names = ['ec', 'oc', 'so4', 'na', 'othmin']
        coarse_species_cols = ['EC_coarse', 'OC_coarse', 'SO4_coarse', 'Na_coarse', 'OthMin_coarse']
        
        # Sector mapping between input bands and GNFR sectors
        gnfr_band_mapping = {
            1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F',
            7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L'
        }
        
        # Open input files
        with rasterio.open(pm2_5_path) as src_pm25, rasterio.open(pm10_path) as src_pm10:
            # Validate inputs
            if src_pm25.count != 13 or src_pm10.count != 13:
                raise ValueError("Inputs must have 13 bands")
            if src_pm25.shape != src_pm10.shape:
                raise ValueError("Raster dimensions don't match")
            
            profile = src_pm25.profile
            profile.update(dtype=rasterio.float32, nodata=-9999.0, compress='deflate')
            
            # Create temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as temp_dir:
                # STEP 1: Speciate PM2.5 (fine)
                fine_species_data = {}
                
                for species_col, species_name in zip(fine_species_cols, fine_species_names):
                    # Initialize arrays for this species
                    fine_bands = []
                    fine_total = np.zeros(src_pm25.shape, dtype=np.float32)
                    
                    # Process each sector band
                    for band_idx in range(1, 13):  # Only process first 12 bands
                        sector_code = gnfr_band_mapping[band_idx]
                        
                        # Get PM2.5 emissions for this sector
                        pm25 = src_pm25.read(band_idx).astype(np.float32)
                        
                        # Get speciation fraction
                        if sector_code == 'F':
                            # For road transport, use average of F1-F4
                            f_fractions = pm_fine[pm_fine['GNFR_Sector'].str.startswith('F')][species_col]
                            fraction = np.average(f_fractions)
                        else:
                            fraction = pm_fine.loc[
                                pm_fine['GNFR_Sector'] == sector_code, species_col
                            ].values[0]
                        
                        # Calculate species emissions
                        species_emissions = np.where(
                            pm25 == src_pm25.nodata,
                            src_pm25.nodata,
                            pm25 * fraction
                        ).astype(np.float32)
                        
                        fine_bands.append(species_emissions)
                        
                        # Accumulate for total band (skip nodata values)
                        valid_mask = species_emissions != src_pm25.nodata
                        fine_total[valid_mask] += species_emissions[valid_mask]
                    
                    fine_species_data[species_name] = {
                        'bands': fine_bands,
                        'total': fine_total
                    }
                
                # STEP 2: Calculate PM coarse and speciate
                coarse_species_data = {}
                
                for species_col in coarse_species_cols:
                    species_name = species_col.split('_')[0].lower()
                    
                    # Initialize arrays for this species
                    coarse_bands = []
                    coarse_total = np.zeros(src_pm10.shape, dtype=np.float32)
                    
                    # Process each sector band
                    for band_idx in range(1, 13):
                        sector_code = gnfr_band_mapping[band_idx]
                        
                        # Calculate coarse PM (PM10 - PM2.5)
                        pm10 = src_pm10.read(band_idx).astype(np.float32)
                        pm25 = src_pm25.read(band_idx).astype(np.float32)
                        coarse = np.where(
                            (pm10 == src_pm10.nodata) | (pm25 == src_pm25.nodata),
                            src_pm10.nodata,
                            pm10 - pm25
                        ).astype(np.float32)
                        
                        # Get speciation fraction
                        if sector_code == 'F':
                            # For road transport, use average of F1-F4
                            f_fractions = pm_coarse[pm_coarse['GNFR_Sector'].str.startswith('F')][species_col]
                            fraction = np.average(f_fractions)
                        else:
                            fraction = pm_coarse.loc[
                                pm_coarse['GNFR_Sector'] == sector_code, species_col
                            ].values[0]
                        
                        # Calculate species emissions
                        species_emissions = np.where(
                            coarse == src_pm10.nodata,
                            src_pm10.nodata,
                            coarse * fraction
                        ).astype(np.float32)
                        
                        coarse_bands.append(species_emissions)
                        
                        # Accumulate for total band
                        valid_mask = species_emissions != src_pm10.nodata
                        coarse_total[valid_mask] += species_emissions[valid_mask]
                    
                    coarse_species_data[species_name] = {
                        'bands': coarse_bands,
                        'total': coarse_total
                    }
                
                # STEP 3: Combine fine and coarse species
                for species_name in fine_species_names:
                    if species_name in coarse_species_data:
                        output_path = os.path.join(output_dir, f"emission_{species_name}_yearly.tif")
                        
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            # Process each sector band
                            for band_idx in range(12):
                                # Get fine and coarse emissions for this band
                                fine_emissions = fine_species_data[species_name]['bands'][band_idx]
                                coarse_emissions = coarse_species_data[species_name]['bands'][band_idx]
                                
                                # Combine them
                                combined = np.where(
                                    (fine_emissions == src_pm25.nodata) | (coarse_emissions == src_pm10.nodata),
                                    profile['nodata'],
                                    fine_emissions + coarse_emissions
                                ).astype(np.float32)
                                
                                dst.write(combined, band_idx + 1)
                            
                            # Write total band (combined fine + coarse totals)
                            fine_total = fine_species_data[species_name]['total']
                            coarse_total = coarse_species_data[species_name]['total']
                            combined_total = np.where(
                                (fine_total == src_pm25.nodata) | (coarse_total == src_pm10.nodata),
                                profile['nodata'],
                                fine_total + coarse_total
                            ).astype(np.float32)
                            
                            dst.write(combined_total, 13)
                            
                            # Set band descriptions from original file
                            if src_pm25.descriptions:
                                for i in range(13):
                                    dst.set_band_description(i + 1, src_pm25.descriptions[i])
                            
                            print(f"Saved combined {species_name} emissions to {output_path}")
    
    except Exception as e:
        print(f"Error in PM speciation: {str(e)}")
        raise
    
    print(f"\nPM speciation completed in {time.time()-start_time:.2f} seconds")
    print(f"Output files saved to: {output_dir}")

def calculate_mass_balance(job_parameters, data_parameters, sectors, species):
    """Parallel mass balance calculation"""
    print("\n=== Starting Parallel Mass Balance Correction ===")
    
    # Create bbox polygon
    bbox_poly = gpd.GeoDataFrame(geometry=[box(
        job_parameters['min_lon'],
        job_parameters['min_lat'],
        job_parameters['max_lon'],
        job_parameters['max_lat'])
    ], crs=f"EPSG:{job_parameters['epsg_code']}")

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

    # Perform NOx splitting if needed
    if 'nox' in species:
        print("\n=== Starting NOx Splitting ===")
        nox_path = os.path.join(job_parameters['job_path'], 'emission_nox_yearly.tif')
        no_path = os.path.join(job_parameters['job_path'], 'emission_no_yearly.tif')
        no2_path = os.path.join(job_parameters['job_path'], 'emission_no2_yearly.tif')
        
        if os.path.exists(nox_path):
            try:
                process_yearly_emissions(nox_path, no_path, no2_path)
                print("nox splitting completed successfully to NO and NO2.")
            except Exception as e:
                print(f"Error during NOx splitting: {str(e)}")
        else:
            print(f"Warning: Mass-balanced NOx file not found: {nox_path}")

    # Perform PM speciation if both PM10 and PM2.5 are present
    if 'pm10' in species and 'pm2_5' in species:
        print("\n=== Starting PM Speciation ===")
        pm10_path = os.path.join(job_parameters['job_path'], 'emission_pm10_yearly.tif')
        pm25_path = os.path.join(job_parameters['job_path'], 'emission_pm2_5_yearly.tif')
        
        if os.path.exists(pm10_path) and os.path.exists(pm25_path):
            try:
                process_pm_speciation(
                    pm10_path=pm10_path,
                    pm2_5_path=pm25_path,
                    speciation_file=data_parameters['pm_speciation_file'],
                    country_code=job_parameters['pm_speciation']['country_code'],
                    year=job_parameters['pm_speciation']['year'],
                    output_dir=job_parameters['job_path']
                )
                print("PM speciation completed successfully.")
            except Exception as e:
                print(f"Error during PM speciation: {str(e)}")
        else:
            missing = []
            if not os.path.exists(pm10_path):
                missing.append('pm10')
            if not os.path.exists(pm25_path):
                missing.append('pm2_5')
            print(f"Warning: Mass-balanced PM files not found: {', '.join(missing)}")