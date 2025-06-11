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


def speciate_pm25(input_path, output_dir, pm_speciation_file, country_code, year):
    """
    Process PM2.5 emissions by speciating them according to sector-specific profiles.
    
    Args:
        input_path (str): Path to input PM2.5 GeoTIFF file
        output_dir (str): Directory to save speciated outputs
        pm_speciation_file (str): Path to Excel file with speciation profiles
        country_code (str): Country code to use (default 'DEU')
        year (int): Year to use (default 2018)
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load PM speciation profiles from Excel
    try:
        pm_species = pd.read_excel(pm_speciation_file, sheet_name='fine')
        
        # Filter for the specific country and year
        pm_species = pm_species[(pm_species['ISO3'] == country_code) & 
                                (pm_species['Year'] == year)]
        
        if pm_species.empty:
            raise ValueError(f"No data found for country {country_code} and year {year}")
        
        # Verify we have the expected columns
        required_columns = ['Year', 'ISO3', 'GNFR_Sector', 'EC_fine', 'OC_fine', 
                          'SO4_fine', 'Na_fine', 'OthMin_fine']
        if not all(col in pm_species.columns for col in required_columns):
            raise ValueError("Excel file doesn't contain expected columns")
            
        # Get list of all species (from column names)
        species_columns = ['EC_fine', 'OC_fine', 'SO4_fine', 'Na_fine', 'OthMin_fine']
        species_names = ['ec', 'oc', 'so4', 'na', 'othmin']  # Simplified names
        print(f"Found {len(species_names)} PM2.5 species in speciation profile")
        
        # Calculate average fractions for road transport (F) sector
        # We'll use these as the default when processing the F band
        road_fractions = {}
        for species_col in species_columns:
            # Get fractions for all road subsectors (F1-F4)
            f_fractions = pm_species[pm_species['GNFR_Sector'].str.startswith('F')][species_col]
            # Calculate weighted average (weighted by typical emission distribution)
            road_fractions[species_col] = np.average(f_fractions)
        
        print("\nRoad transport sector average fractions:")
        for species, fraction in zip(species_names, road_fractions.values()):
            print(f"  {species}: {fraction:.4f}")
            
    except Exception as e:
        print(f"Error loading speciation profiles: {str(e)}")
        raise
    
    # Define sector mapping between input bands and GNFR sectors
    # This maps the band order in the input GeoTIFF to GNFR sectors
    gnfr_band_mapping = {
        1: 'A',  # Public Power
        2: 'B',  # Industry
        3: 'C',  # Other Stationary Comb
        4: 'D',  # Fugitives
        5: 'E',  # Solvents
        6: 'F',  # Road Transport (will use average of F1-F4)
        7: 'G',  # Shipping
        8: 'H',  # Aviation
        9: 'I',  # Off-road
        10: 'J', # Waste
        11: 'K', # Agri Livestock
        12: 'L'  # Agri Other
    }
    
    # Open input PM2.5 file
    with rasterio.open(input_path) as src:
        # Verify we have the expected number of bands (12 sectors + SumAllSectors)
        if src.count != 13:
            raise ValueError(f"Expected 13 bands in input file (12 sectors + SumAllSectors), found {src.count}")
        
        # Get the profile for output files
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            count=src.count,  # Same number of bands (13)
            nodata=src.nodata if src.nodata is not None else -9999.0,
            compress='deflate'
        )
        
        # Get band descriptions from input
        band_descriptions = src.descriptions if src.descriptions else [f"Band_{i}" for i in range(1, 14)]
        
        # Process each species
        for species_col, species_name in zip(species_columns, species_names):
            output_path = os.path.join(output_dir, f"emission_{species_name}_yearly.tif")
            print(f"\nProcessing species: {species_name}")
            
            try:
                with rasterio.open(output_path, 'w', **profile) as dst:
                    # Copy metadata from input
                    dst.update_tags(**src.tags())
                    
                    # Initialize sum for recalculating SumAllSectors
                    sum_all = np.zeros(src.shape, dtype=np.float32)
                    
                    # Process each sector band (bands 1-12)
                    for band_idx in range(1, 13):  # Only process first 12 bands
                        # Get the GNFR sector for this band
                        sector_code = gnfr_band_mapping[band_idx]
                        
                        # Read the PM2.5 emissions for this sector
                        pm25 = src.read(band_idx)
                        
                        if sector_code == 'F':
                            # For road transport, use the average fractions
                            fraction = road_fractions[species_col]
                        else:
                            # For non-road sectors
                            fraction = pm_species.loc[
                                pm_species['GNFR_Sector'] == sector_code,
                                species_col
                            ].values[0]
                        
                        # Calculate species emissions
                        species_emissions = pm25 * fraction
                        
                        # Write to output
                        dst.write(species_emissions, band_idx)
                        dst.set_band_description(band_idx, band_descriptions[band_idx-1])
                        
                        # Add to total sum
                        sum_all += species_emissions
                    
                    # Write the recalculated SumAllSectors band (band 13)
                    dst.write(sum_all, 13)
                    dst.set_band_description(13, band_descriptions[12])  # Keep original description
                    
                    print(f"Saved: {output_path}")
            
            except Exception as e:
                print(f"Error processing {species_name}: {str(e)}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise
    
    # Verification of mass conservation
    verify_mass_conservation(input_path, output_dir, species_names)
    
    print(f"\nPM2.5 speciation completed in {time.time()-start_time:.2f} seconds")
    print(f"Output files saved to: {output_dir}")

def verify_mass_conservation(input_path, output_dir, species_names):
    """Verify that the sum of speciated emissions equals the original PM2.5"""
    print("\nVerifying mass conservation...")
    
    with rasterio.open(input_path) as src:
        original_total = src.read(13)  # SumAllSectors band
        
        # Initialize sum of all species
        species_sum = np.zeros(original_total.shape, dtype=np.float32)
        
        # Sum all species files
        for species in species_names:
            species_path = os.path.join(output_dir, f"emission_{species}_yearly.tif")
            with rasterio.open(species_path) as species_src:
                species_sum += species_src.read(13)  # SumAllSectors band for this species
        
        # Calculate differences
        diff = np.abs(original_total - species_sum)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Find where differences exceed 1%
        threshold = original_total * 0.01
        large_diff_count = np.sum(diff > threshold)
        total_pixels = original_total.size
        
        print(f"Mass conservation results:")
        print(f"  Maximum difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Pixels with >1% difference: {large_diff_count}/{total_pixels} ({large_diff_count/total_pixels:.2%})")
        
        if max_diff > 0.01:  # Absolute tolerance
            print("Warning: Significant differences detected in mass conservation")
        else:
            print("Mass conservation verified successfully")

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

    # Perform PM2.5 speciation if needed
    if 'pm2_5' in species:
        print("\n=== Starting PM2.5 Speciation ===")
        pm25_path = os.path.join(job_parameters['job_path'], 'emission_pm2_5_yearly.tif')
        
        if os.path.exists(pm25_path):
            try:
                speciate_pm25(
                    input_path=pm25_path,
                    output_dir=job_parameters['job_path'],
                    pm_speciation_file=data_parameters['pm_speciation_file'],
                    country_code=job_parameters['pm_speciation']['country_code'],
                    year=job_parameters['pm_speciation']['year']
                )
                print("PM2.5 speciation completed successfully.")
            except Exception as e:
                print(f"Error during PM2.5 speciation: {str(e)}")
        else:
            print(f"Warning: Mass-balanced PM2.5 file not found: {pm25_path}")