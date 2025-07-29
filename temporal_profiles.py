#CST CEST
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
import os
import calendar
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TemporalProfiler:
    def __init__(self, data_parameters, job_parameters):
        """Initialize with EDGAR profile paths and load sector mapping"""
        self.edgar_monthly = data_parameters['edgar_monthly']
        self.edgar_hourly = data_parameters['edgar_hourly']
        self.edgar_weekly = data_parameters['edgar_weekly']
        self.weekend_types = data_parameters['weekend_types']
        self.daytype_mapping = data_parameters['daytype_mapping']
        
        self.country = job_parameters['temporal']['country']
        self.profile_year = job_parameters['temporal']['profile_year']
        
        # Preload all data to avoid repeated loading
        self.sector_mapping = self._load_sector_mapping(data_parameters['greta_to_edgar'])
        self._load_profiles()
        self._precompute_timezone_data()

    def _load_sector_mapping(self, mapping_file):
        """Load GRETA to EDGAR sector mapping efficiently"""
        try:
            df = pd.read_excel(mapping_file, sheet_name='Emission_groups', usecols=['Emission_category', 'EDGAR_sector'])
            return dict(zip(df['Emission_category'], df['EDGAR_sector']))
        except Exception as e:
            raise RuntimeError(f"Failed to load sector mapping: {str(e)}")

    def _load_profiles(self):
        """Load and preprocess EDGAR temporal profiles"""
        try:
            # Load monthly profiles
            monthly_df = pd.read_excel(self.edgar_monthly, sheet_name='monthly & hourly temp profiles',
                                     usecols=['country', 'Year', 'IPCC_2006_source_category'] + 
                                             ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            monthly_df = monthly_df[(monthly_df['country'] == self.country) & 
                                  (monthly_df['Year'] == self.profile_year)]
            print("\nMonthly Profile Sample:")
            print(monthly_df[['IPCC_2006_source_category', 'Jan', 'Jul']].head(3))
            
            # Load weekly profiles
            weekly_df = pd.read_csv(self.edgar_weekly, usecols=['Country_code_A3', 'activity_code', 'Weekday_id', 'daily_factor'])
            weekly_df = weekly_df[weekly_df['Country_code_A3'] == self.country]
            print("\nWeekly Profile Sample:")
            print(weekly_df[['activity_code', 'Weekday_id', 'daily_factor']].head(6))
            
            # Load hourly profiles
            expected_cols = ['Country_code_A3', 'activity_code', 'Daytype_id', 'month_id'] + [f'h{h}' for h in range(1, 25)]
            hourly_df = pd.read_csv(self.edgar_hourly, usecols=[col for col in expected_cols if col in pd.read_csv(self.edgar_hourly, nrows=0).columns])
            hourly_df = hourly_df[hourly_df['Country_code_A3'] == self.country]
            print("\nHourly Profile Sample (Transport Sector - Weekday):")
            transport_weekday = hourly_df[(hourly_df['activity_code'] == 'TRO') & 
                                       (hourly_df['Daytype_id'] == 1)]
            if not transport_weekday.empty:
                print(transport_weekday.iloc[0][[col for col in ['h7', 'h8', 'h9', 'h17', 'h18', 'h19'] if col in transport_weekday.columns]])
            else:
                print("No transport weekday data available.")
            
            # Load and merge weekend/weekday definitions
            weekend_days = pd.read_csv(self.weekend_types, sep=';', header=0)
            week_days = pd.read_csv(self.daytype_mapping, sep=';', header=0)
            week_days_per_country = weekend_days.merge(week_days, how='left', on=['Weekend_type_id'])
            
            # Normalize profiles
            month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_df[month_cols] = monthly_df[month_cols].div(
                monthly_df[month_cols].sum(axis=1), axis=0).fillna(1.0 / 12)
            
            hour_cols = [f'h{h}' for h in range(1, 25)]
            if all(col in hourly_df.columns for col in hour_cols):
                hourly_df[hour_cols] = hourly_df[hour_cols].div(
                    hourly_df[hour_cols].sum(axis=1), axis=0).fillna(1.0 / 24)
            else:
                print("Warning: Some hourly columns missing. Using uniform distribution.")
                for col in hour_cols:
                    if col not in hourly_df.columns:
                        hourly_df[col] = 1.0 / 24
                
            self.monthly_profiles = monthly_df
            self.weekly_profiles = weekly_df
            self.hourly_profiles = hourly_df
            self.week_days_per_country = week_days_per_country
            self.has_tz_id = False  # Set to False as TZ_id is not needed
            
        except Exception as e:
            raise RuntimeError(f"Failed to load profiles: {str(e)}")

    def _precompute_timezone_data(self):
        """Set minimal timezone data since no adjustments are needed"""
        self.is_multi_timezone = False

    def _get_edgar_sector(self, greta_sector):
        """Get corresponding EDGAR sector for GRETA sector"""
        base_sector = greta_sector.split('_')[0]
        return self.sector_mapping.get(base_sector, 'TRO')

    def _get_daytype(self, date):
        """Determine day type (1=weekday, 2=saturday, 3=sunday)"""
        weekday = date.weekday()
        return 2 if weekday == 5 else 3 if weekday == 6 else 1

    def _apply_dst_adjust(self, hourly_factors, utc_reference, current_date):
        """Return hourly factors without any timezone or DST shift to preserve local time"""
        return hourly_factors

    def _get_time_factors(self, greta_sector, date, lat=None, lon=None):
        """Get monthly, daily, and hourly factors without timezone adjustments"""
        edgar_sector = self._get_edgar_sector(greta_sector)
        month = date.month
        daytype = self._get_daytype(date)
        
        # Get monthly factor
        monthly_data = self.monthly_profiles[
            self.monthly_profiles['IPCC_2006_source_category'].str.startswith(edgar_sector)
        ]
        month_col = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
        monthly_factor = monthly_data[month_col].values[0] if not monthly_data.empty else 1.0
        
        # Get weekly factor
        weekly_data = self.weekly_profiles[
            (self.weekly_profiles['activity_code'] == edgar_sector) &
            (self.weekly_profiles['Weekday_id'] == daytype)
        ]
        weekly_factor = weekly_data['daily_factor'].values[0] if not weekly_data.empty else 1.0
        
        # Normalize weekly factors
        days_in_month = calendar.monthrange(date.year, date.month)[1]
        month_df = pd.DataFrame({
            'day_in_month': range(1, days_in_month + 1),
            'Weekday_id': [self._get_daytype(date.replace(day=d)) for d in range(1, days_in_month + 1)]
        })
        month_with_countries = month_df.merge(self.week_days_per_country, how='left', 
                                           on=['Weekday_id'])
        monthly_profiled = month_with_countries.merge(
            self.weekly_profiles[self.weekly_profiles['activity_code'] == edgar_sector], 
            how='left', on=['Country_code_A3', 'Weekday_id']
        )
        sum_daily_factors = monthly_profiled['daily_factor'].sum()
        weekly_factor = weekly_factor / sum_daily_factors if sum_daily_factors != 0 else 1.0
        
        # Get hourly factors
        hourly_data = self.hourly_profiles[
            (self.hourly_profiles['activity_code'] == edgar_sector) &
            (self.hourly_profiles['Daytype_id'] == daytype) &
            (self.hourly_profiles['month_id'] == month)
        ]
        
        if not hourly_data.empty:
            hour_cols = [f'h{h}' for h in range(1, 25)]
            hourly_factors = hourly_data[hour_cols].values[0]
            hourly_factors = self._apply_dst_adjust(hourly_factors, None, date)
        else:
            hourly_factors = np.ones(24) / 24
            print(f"No hourly profile for {edgar_sector}, using uniform distribution")
        
        annual_weight = (monthly_factor / days_in_month) * weekly_factor
        return annual_weight, hourly_factors

    def _create_output_files(self, input_fn, base_output_fn, sectors, start_date, end_date, max_bands_per_file=60000):
        """Create output file structure"""
        total_days = (end_date - start_date).days + 1
        bands_per_day = 24 * len(sectors)
        total_bands = bands_per_day * total_days
        
        # Determine if splitting is needed based on band count
        if total_bands <= max_bands_per_file:
            # Use single file without part suffix
            output_files = [(base_output_fn, start_date, end_date)]
            days_per_file = total_days
            print(f"\nCreating single output file with {total_bands} bands")
        else:
            # Split into multiple files with part suffix
            num_files = (total_bands + max_bands_per_file - 1) // max_bands_per_file
            bands_per_file = (total_bands + num_files - 1) // num_files
            days_per_file = max(bands_per_file // bands_per_day, 1)
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
                output_files.append((output_fn, file_start_date, file_end_date))
                current_date = file_end_date + timedelta(days=1)
                file_index += 1
        
        return output_files, days_per_file

    def _process_day(self, current_date, sectors, yearly_bands, x_size, y_size, lats, lons):
        """Process a single day's data for all sectors"""
        band_data = []
        band_descs = []
        min_max_values = {}
        
        for sec_idx, sec in enumerate(sectors):
            yearly_emis = yearly_bands[sec_idx]
            annual_weight, hourly_factors_base = self._get_time_factors(sec, current_date)
            
            daily_emis = yearly_emis * annual_weight
            for hour in range(24):
                hour_emis = daily_emis * hourly_factors_base[hour]
                band_data.append(hour_emis)
                hour_str = f"{hour+1:02d}"
                band_desc = f"{sec}_h{hour_str}_{current_date.strftime('%Y%m%d')}"
                band_descs.append(band_desc)
                min_max_values[band_desc] = {
                    'min': np.nanmin(hour_emis),
                    'max': np.nanmax(hour_emis),
                    'mean': np.nanmean(hour_emis)
                }
        
        return band_data, band_descs, min_max_values

    def _process_file(self, input_fn, output_fn, sectors, start_date, end_date, x_size, y_size, geotransform, projection, output_files):
        """Process a single output file"""
        try:
            ds = gdal.Open(input_fn)
            yearly_bands = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)]
            
            lon_start, lon_step = geotransform[0], geotransform[1]
            lat_start, lat_step = geotransform[3], geotransform[5]
            lons = np.linspace(lon_start, lon_start + lon_step * (x_size - 1), x_size)
            lats = np.linspace(lat_start, lat_start + lat_step * (y_size - 1), y_size)
            
            file_days = (end_date - start_date).days + 1
            file_bands = 24 * len(sectors) * file_days
            
            print(f"\nCreating {output_fn} with {file_bands} bands "
                  f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_fn, x_size, y_size, file_bands, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'PREDICTOR=2', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS']
            )
            out_ds.SetGeoTransform(geotransform)
            out_ds.SetProjection(projection)
            
            band_idx = 1
            min_max_values = {}
            
            with tqdm(total=file_days, desc=f"Processing {os.path.basename(output_fn)}") as pbar:
                current_date = start_date
                while current_date <= end_date:
                    band_data, band_descs, day_min_max = self._process_day(
                        current_date, sectors, yearly_bands, x_size, y_size, lats, lons)
                    for data, desc in zip(band_data, band_descs):
                        out_ds.GetRasterBand(band_idx).WriteArray(data)
                        out_ds.GetRasterBand(band_idx).SetDescription(desc)
                        min_max_values[desc] = day_min_max[desc]
                        band_idx += 1
                    current_date += timedelta(days=1)
                    pbar.update(1)
            
            metadata = {
                'temporal_disaggregation': 'EDGAR_no_timezone_adjustment',
                'temporal_status': 'Hourly',
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'country': self.country,
                'profile_year': str(self.profile_year),
                'total_days': str(file_days),
                'total_bands': str(file_bands),
                'band_organization': 'sector_hourly_daily',
                'file_part': f"{output_files.index((output_fn, start_date, end_date)) + 1} of {len(output_files)}",
                'original_filename': os.path.basename(output_fn)
            }
            out_ds.SetMetadata(metadata)
            out_ds = None
            ds = None
            print(f"Successfully created: {output_fn}")
            return min_max_values
        except Exception as e:
            print(f"Error processing {output_fn}: {str(e)}")
            return None

    def _process_temporal_disaggregation(self, input_fn, base_output_fn, sectors, start_date, end_date):
        """Perform temporal disaggregation with parallel processing"""
        try:
            ds = gdal.Open(input_fn)
            if ds is None:
                print(f"Error: Could not open {input_fn}")
                return False
                
            geotransform = ds.GetGeoTransform()
            x_size, y_size = ds.RasterXSize, ds.RasterYSize
            projection = ds.GetProjection()
            
            output_files, days_per_file = self._create_output_files(
                input_fn, base_output_fn, sectors, start_date, end_date)
            
            min_max_values = {}
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_file, input_fn, output_fn, sectors, 
                                  file_start, file_end, x_size, y_size, geotransform, projection, output_files)
                    for output_fn, file_start, file_end in output_files
                ]
                for future in futures:
                    result = future.result()
                    if result:
                        min_max_values.update(result)
            
            print(f"\nTemporal variation for {os.path.basename(base_output_fn)}:")
            for band, values in list(min_max_values.items())[:5] + list(min_max_values.items())[-5:]:
                print(f"{band}: min={values['min']:.9f}, max={values['max']:.9f}, mean={values['mean']:.9f}")
            
            return True
        except Exception as e:
            print(f"Error processing {input_fn}: {str(e)}")
            return False

    def apply_temporal_profiles(self, job_parameters, sectors):
        """Apply temporal profiles to downscaled emissions"""
        print('\n=== Starting Temporal Disaggregation ===')
        
        main_sectors = [item[0] for item in sectors]
        start_date = datetime.strptime(job_parameters['temporal']['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(job_parameters['temporal']['end_date'], "%Y-%m-%d")
        
        total_days = (end_date - start_date).days + 1
        total_bands = 24 * len(main_sectors) * total_days
        print(f"\nTotal bands needed: {total_bands} (will be split into multiple files if needed)")
        
        species_to_process = job_parameters['species'].copy()
        
        if 'nox' in species_to_process:
            if 'no' not in species_to_process: species_to_process.append('no')
            if 'no2' not in species_to_process: species_to_process.append('no2')
            if 'o3' not in species_to_process: species_to_process.append('o3')
        
        if 'pm2_5' in species_to_process and 'pm10' in species_to_process:
            pm_components = ['EC', 'OC', 'SO4', 'Na', 'OthMin']
            for comp in pm_components:
                if comp.lower() not in species_to_process:
                    species_to_process.append(comp.lower())
        
        for spec in tqdm(species_to_process, desc="Processing species"):
            if spec in ['no', 'no2'] and 'nox' not in job_parameters['species']:
                continue
            if spec in ['ec', 'oc', 'so4', 'na', 'othmin'] and 'pm2_5' not in job_parameters['species']:
                continue
                
            input_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_yearly.tif')
            base_output_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_temporal.tif')
            
            if not os.path.exists(input_fn):
                print(f"Warning: Input file not found: {input_fn}")
                continue
            
            self._process_temporal_disaggregation(input_fn, base_output_fn, main_sectors, start_date, end_date)
                
        print("\n=== Mass Conserved downscaling of GRETA emissions with Temporal Disaggregation Completed ===")