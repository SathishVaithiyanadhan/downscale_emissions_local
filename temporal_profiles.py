import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
import os
import calendar
from datetime import datetime, timedelta

class TemporalProfiler:
    def __init__(self, data_parameters, job_parameters):
        """Initialize with EDGAR profile paths and load sector mapping"""
        self.edgar_monthly = data_parameters['edgar_monthly']
        self.edgar_hourly = data_parameters['edgar_hourly']
        self.edgar_weekly = data_parameters['edgar_weekly']
        self.weekend_types = data_parameters['weekend_types']
        self.daytype_mapping = data_parameters['daytype_mapping']
        
        # Get country and year from job parameters
        self.country = job_parameters['temporal']['country']
        self.profile_year = job_parameters['temporal']['profile_year']
        
        # Load GRETA to EDGAR sector mapping
        self.sector_mapping = self._load_sector_mapping(data_parameters['greta_to_edgar'])
        
        # Initialize profile data
        self.monthly_profiles = None
        self.weekly_profiles = None
        self.hourly_profiles = None
        self.weekend_defs = None
        self.daytype_map = None

    def _load_sector_mapping(self, mapping_file):
        """Load GRETA to EDGAR sector mapping from Excel file"""
        try:
            df = pd.read_excel(mapping_file, sheet_name='Emission_groups')
            return dict(zip(df['Emission_category'], df['EDGAR_sector']))
        except Exception as e:
            raise RuntimeError(f"Failed to load sector mapping: {str(e)}")

    def _load_profiles(self):
        """Load all EDGAR temporal profiles with verification"""
        try:
            # 1. Load and verify monthly profiles
            monthly_df = pd.read_excel(self.edgar_monthly, sheet_name='monthly & hourly temp profiles')
            monthly_df = monthly_df[(monthly_df['country'] == self.country) & 
                                  (monthly_df['Year'] == self.profile_year)]
            
            print("\nMonthly Profile Sample:")
            print(monthly_df[['IPCC_2006_source_category', 'Jan', 'Jul']].head(3))
            
            # 2. Load and verify weekly profiles
            weekly_df = pd.read_csv(self.edgar_weekly)
            weekly_df = weekly_df[weekly_df['Country_code_A3'] == self.country]
            print("\nWeekly Profile Sample:")
            print(weekly_df[['activity_code', 'Weekday_id', 'daily_factor']].head(6))
            
            # 3. Load and verify hourly profiles
            hourly_df = pd.read_csv(self.edgar_hourly)
            hourly_df = hourly_df[hourly_df['Country_code_A3'] == self.country]
            print("\nHourly Profile Sample (Transport Sector - Weekday):")
            transport_weekday = hourly_df[(hourly_df['activity_code'] == 'TRO') & 
                                        (hourly_df['Daytype_id'] == 1)]
            print(transport_weekday.iloc[0][['h7', 'h8', 'h9', 'h17', 'h18', 'h19']])
            
            # Normalization
            month_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            monthly_df[month_cols] = monthly_df[month_cols].div(
                monthly_df[month_cols].sum(axis=1), axis=0)
            
            hour_cols = [f'h{h}' for h in range(1, 25)]
            hourly_df[hour_cols] = hourly_df[hour_cols].div(
                hourly_df[hour_cols].sum(axis=1), axis=0)
                
            return monthly_df, weekly_df, hourly_df, None, None
            
        except Exception as e:
            raise RuntimeError(f"Failed to load EDGAR profiles: {str(e)}")

    def _get_edgar_sector(self, greta_sector):
        """Get corresponding EDGAR sector for GRETA sector"""
        base_sector = greta_sector.split('_')[0]
        return self.sector_mapping.get(base_sector, 'TRO')

    def _get_daytype(self, date):
        """Determine day type (1=weekday, 2=saturday, 3=sunday)"""
        weekday = date.weekday()
        if weekday == 5:  # Saturday
            return 2
        elif weekday == 6:  # Sunday
            return 3
        return 1  # Weekday

    def _get_time_factors(self, greta_sector, date):
        """Get time factors with verification prints"""
        edgar_sector = self._get_edgar_sector(greta_sector)
        month = date.month
        daytype = self._get_daytype(date)
        
        # Get monthly factor
        monthly_data = self.monthly_profiles[
            self.monthly_profiles['IPCC_2006_source_category'].str.startswith(edgar_sector)
        ]
        month_col = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'][month-1]
        monthly_factor = monthly_data[month_col].values[0] if not monthly_data.empty else 1.0
        
        # Get weekly factor
        weekly_data = self.weekly_profiles[
            (self.weekly_profiles['activity_code'] == edgar_sector) &
            (self.weekly_profiles['Weekday_id'] == daytype)
        ]
        weekly_factor = weekly_data['daily_factor'].values[0] if not weekly_data.empty else 1.0
        
        # Get hourly factors
        hourly_data = self.hourly_profiles[
            (self.hourly_profiles['activity_code'] == edgar_sector) &
            (self.hourly_profiles['Daytype_id'] == daytype)
        ]
        if not hourly_data.empty:
            hour_cols = [f'h{h}' for h in range(1, 25)]
            hourly_factors = hourly_data[hour_cols].values[0]
        else:
            hourly_factors = np.ones(24)/24
            print(f"No hourly profile for {edgar_sector}, using uniform distribution")
        
        days_in_month = calendar.monthrange(date.year, date.month)[1]
        annual_weight = (monthly_factor / days_in_month) * weekly_factor
        
        return annual_weight, hourly_factors

    def _create_output_files(self, input_fn, base_output_fn, sectors, start_date, end_date, max_bands_per_file=60000):
        """
        Create multiple output files to handle the band limit
        Returns a list of created file paths
        """
        # Calculate total bands needed
        total_days = (end_date - start_date).days + 1
        bands_per_day = 24 * len(sectors)
        total_bands = bands_per_day * total_days
        
        # Determine how many files we need
        num_files = (total_bands + max_bands_per_file - 1) // max_bands_per_file
        bands_per_file = (total_bands + num_files - 1) // num_files
        
        # Calculate days per file
        days_per_file = bands_per_file // bands_per_day
        if days_per_file < 1:
            days_per_file = 1
        
        print(f"\nSplitting output into {num_files} files with max {bands_per_file} bands each")
        
        output_files = []
        current_date = start_date
        file_index = 1
        
        while current_date <= end_date:
            file_start_date = current_date
            file_end_date = current_date + timedelta(days=days_per_file - 1)
            if file_end_date > end_date:
                file_end_date = end_date
            
            output_fn = f"{base_output_fn[:-4]}_part{file_index:02d}.tif"
            output_files.append(output_fn)
            
            current_date = file_end_date + timedelta(days=1)
            file_index += 1
        
        return output_files, days_per_file

    def _process_temporal_disaggregation(self, input_fn, base_output_fn, sectors, start_date, end_date):
        """Helper function to perform temporal disaggregation with smart splitting"""
        try:
            ds = gdal.Open(input_fn)
            if ds is None:
                print(f"Error: Could not open {input_fn}")
                return False
                
            yearly_bands = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, 14)]
            
            # Calculate total bands needed
            total_days = (end_date - start_date).days + 1
            bands_per_day = 24 * len(sectors)
            total_bands = bands_per_day * total_days
            
            # Determine if we need to split
            max_bands_per_file = 60000  # Conservative limit
            needs_split = total_bands > max_bands_per_file
            
            if needs_split:
                print(f"\nTotal bands ({total_bands}) exceeds limit, splitting output")
                # Calculate how many days we can fit in first file
                days_in_first_file = max_bands_per_file // bands_per_day
                if days_in_first_file < 1:
                    days_in_first_file = 1
                
                file_ranges = [
                    (start_date, start_date + timedelta(days=days_in_first_file - 1)),
                    (start_date + timedelta(days=days_in_first_file), end_date)
                ]
                output_fns = [
                    base_output_fn,
                    f"{base_output_fn[:-4]}_part02.tif"
                ]
            else:
                file_ranges = [(start_date, end_date)]
                output_fns = [base_output_fn]
            
            # Track min/max values for verification
            min_max_values = {}

            for file_idx, (file_start, file_end) in enumerate(file_ranges):
                output_fn = output_fns[file_idx]
                file_days = (file_end - file_start).days + 1
                file_bands = bands_per_day * file_days
                
                print(f"\nCreating {output_fn} with {file_bands} bands "
                      f"({file_start.strftime('%Y-%m-%d')} to {file_end.strftime('%Y-%m-%d')})")
                
                # Create the output file
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(
                    output_fn,
                    ds.RasterXSize,
                    ds.RasterYSize,
                    file_bands,
                    gdal.GDT_Float32,
                    options=['COMPRESS=LZW', 'PREDICTOR=2', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS']
                )
                out_ds.SetGeoTransform(ds.GetGeoTransform())
                out_ds.SetProjection(ds.GetProjection())
                
                band_idx = 1
                current_date = file_start
                
                with tqdm(total=file_days, desc=f"Processing {os.path.basename(output_fn)}") as pbar:
                    while current_date <= file_end:
                        for sec_idx, sec in enumerate(sectors):
                            yearly_emis = yearly_bands[sec_idx]
                            annual_weight, hourly_factors = self._get_time_factors(sec, current_date)
                            daily_emiss = yearly_emis * annual_weight
                            
                            for hour in range(24):
                                hour_emis = daily_emiss * hourly_factors[hour]
                                out_ds.GetRasterBand(band_idx).WriteArray(hour_emis)
                                
                                hour_str = f"{hour+1:02d}"
                                out_ds.GetRasterBand(band_idx).SetDescription(
                                    f"{sec}_h{hour_str}_{current_date.strftime('%Y%m%d')}")
                                
                                # Track min/max values
                                band_desc = f"{sec}_h{hour_str}_{current_date.strftime('%Y%m%d')}"
                                min_max_values[band_desc] = {
                                    'min': np.nanmin(hour_emis),
                                    'max': np.nanmax(hour_emis),
                                    'mean': np.nanmean(hour_emis)
                                }
                                band_idx += 1
                        
                        current_date += timedelta(days=1)
                        pbar.update(1)
                
                # Set metadata for this file
                metadata = {
                    'temporal_disaggregation': 'EDGAR',
                    'temporal_status': 'Hourly',
                    'start_date': file_start.strftime('%Y-%m-%d'),
                    'end_date': file_end.strftime('%Y-%m-%d'),
                    'country': self.country,
                    'profile_year': str(self.profile_year),
                    'total_days': str(file_days),
                    'total_bands': str(file_bands),
                    'band_organization': 'sector_hourly_daily',
                    'file_part': f"{file_idx + 1} of {len(file_ranges)}" if needs_split else "1 of 1",
                    'original_filename': os.path.basename(base_output_fn)
                }
                out_ds.SetMetadata(metadata)
                
                out_ds = None
                print(f"Successfully created: {output_fn}")
            
            # Print min/max values for verification
            print(f"\nTemporal variation for {os.path.basename(base_output_fn)}:")
            for band, values in list(min_max_values.items())[:5] + list(min_max_values.items())[-5:]:
                print(f"{band}: min={values['min']:.9f}, max={values['max']:.9f}, mean={values['mean']:.9f}")

            ds = None
            return True
            
        except Exception as e:
            print(f"\nError processing {input_fn}: {str(e)}")
            if 'ds' in locals(): ds = None
            if 'out_ds' in locals(): out_ds = None
            return False

    def apply_temporal_profiles(self, job_parameters, sectors):
        """Apply temporal profiles to downscaled emissions"""
        print('\n=== Starting Temporal Disaggregation ===')
        
        # Load EDGAR profiles
        (self.monthly_profiles, self.weekly_profiles, 
         self.hourly_profiles, self.weekend_defs, self.daytype_map) = self._load_profiles()
        
        main_sectors = [item[0] for item in sectors]
        start_date = datetime.strptime(job_parameters['temporal']['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(job_parameters['temporal']['end_date'], "%Y-%m-%d")
        
        # Calculate total bands needed to warn user
        total_days = (end_date - start_date).days + 1
        total_bands = 24 * len(main_sectors) * total_days
        print(f"\nTotal bands needed: {total_bands} (will be split into multiple files if needed)")
        
        # Process all species including NO/NO2 if NOx is present and PM components if PM2.5 is present
        species_to_process = job_parameters['species'].copy()
        
        # Add NO and NO2 if NOx is present
        if 'nox' in species_to_process:
            if 'no' not in species_to_process: species_to_process.append('no')
            if 'no2' not in species_to_process: species_to_process.append('no2')
        
        # Add PM2.5 components if PM2.5 is present
        if 'pm2_5' in species_to_process and 'pm10' in species_to_process:
            pm_components = ['EC', 'OC', 'SO4', 'Na', 'OthMin']
            for comp in pm_components:
                if comp.lower() not in species_to_process:
                    species_to_process.append(comp.lower())
        
        for spec in tqdm(species_to_process, desc="Processing species"):
            # Skip if this is NO/NO2 but we don't have NOx file
            if spec in ['no', 'no2'] and 'nox' not in job_parameters['species']:
                continue
                
            # Skip if this is a PM component but we don't have PM2.5 file
            if spec in ['EC', 'OC', 'SO4', 'Na', 'OthMin'] and 'pm2_5' not in job_parameters['species']:
                continue
                
            input_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_yearly.tif')
            base_output_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_temporal.tif')
            
            if not os.path.exists(input_fn):
                print(f"Warning: Input file not found: {input_fn}")
                continue
            
            self._process_temporal_disaggregation(input_fn, base_output_fn, main_sectors, start_date, end_date)
                
        print("\n=== Mass Conserved downscaling of GRETA emissions with Temporal Disaggregation Completed ===")