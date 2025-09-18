#mass conservation check
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
import os
import calendar
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import warnings
import sys
import time
from functools import lru_cache
import multiprocessing as mp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TemporalProfiler:
    def __init__(self, data_parameters, job_parameters):
        """Initialize with EDGAR profile paths and load sector mapping"""
        # Set up logging to file
        self.log_file = os.path.join(job_parameters['job_path'], 'temporal_profiler_log.txt')
        self.original_stdout = sys.stdout
        sys.stdout = self
        
        self.edgar_monthly = data_parameters['edgar_monthly']
        self.edgar_hourly = data_parameters['edgar_hourly']
        self.edgar_weekly = data_parameters['edgar_weekly']
        self.weekend_types = data_parameters['weekend_types']
        self.daytype_mapping = data_parameters['daytype_mapping']
        
        self.country = job_parameters['temporal']['country']
        self.profile_year = job_parameters['temporal']['profile_year']
        
        # Preload all data to avoid repeated loading
        self.sector_mapping = self._load_sector_mapping(data_parameters['greta_to_edgar'])
        self.sector_year_mapping = self._create_sector_year_mapping()
        self.ipcc_category_mapping = self._create_ipcc_category_mapping()
        self.activity_code_mapping = self._create_activity_code_mapping()
        self._load_profiles()
        self._precompute_timezone_data()
        
        # Precompute and cache frequently used data
        self._precompute_cached_data()
    
    def write(self, text):
        """Write to both console and log file"""
        self.original_stdout.write(text)
        with open(self.log_file, 'a') as f:
            f.write(text)
    
    def flush(self):
        """Flush both console and log file"""
        self.original_stdout.flush()
        with open(self.log_file, 'a') as f:
            f.flush()

    def __del__(self):
        """Restore stdout when object is destroyed"""
        sys.stdout = self.original_stdout

    def _load_sector_mapping(self, mapping_file):
        """Load GRETA to EDGAR sector mapping efficiently"""
        try:
            df = pd.read_excel(mapping_file, sheet_name='Emission_groups', usecols=['Emission_category', 'EDGAR_sector'])
            # Convert to dictionary with proper GRETA sector names
            sector_map = {}
            for _, row in df.iterrows():
                sector_map[row['Emission_category']] = row['EDGAR_sector']
            return sector_map
        except Exception as e:
            raise RuntimeError(f"Failed to load sector mapping: {str(e)}")

    def _create_sector_year_mapping(self):
        """Create mapping of GRETA sectors to their profile years"""
        return {
            'A_PublicPower': 2017,
            'B_Industry': 0,
            'C_OtherStationaryComb': 2017,
            'D_Fugitives': 0,
            'E_Solvents': 0,
            'F_RoadTransport': 0,
            'G_Shipping': 0,
            'H_Aviation': 0,
            'I_OffRoad': 0,
            'J_Waste': 0,
            'K_AgriLivestock': 0,
            'L_AgriOther': 0
        }

    def _create_ipcc_category_mapping(self):
        """Create mapping of GRETA sectors to their IPCC categories"""
        return {
            'A_PublicPower': ['1.A.1'],
            'B_Industry': ['1A2a', '1A2f', '1A2b', '1A2d', '1B1', '1B2'],
            'C_OtherStationaryComb': ['1A4', '3.C.1'],
            'D_Fugitives': ['1B'],
            'E_Solvents': ['2D'],
            'F_RoadTransport': ['1A3b', '1A3b v'],
            'G_Shipping': ['1.A.3.d.i', '1.A.3.d.ii'],
            'H_Aviation': ['1.A.3.a.ii', '1.A.3.a.i'],
            'I_OffRoad': ['1A3e'],
            'J_Waste': ['4A', '4B', '4C', '4F', '4D'],
            'K_AgriLivestock': ['3C4', '3A2'],
            'L_AgriOther': ['3.C.1.b', '3.C.2', '3.C.3', '3.C.4', '3C7', '1A3c', '3.C.5']
        }

    def _create_activity_code_mapping(self):
        """Create mapping of GRETA sectors to their activity codes for weekly/hourly profiles"""
        return {
            'A_PublicPower': ['ENE'],
            'B_Industry': ['IND', 'CHE', 'FOO', 'NFE', 'MNM', 'IRO'],
            'C_OtherStationaryComb': ['BMB', 'IDE', 'RCO'],
            'D_Fugitives': ['ENF'],
            'E_Solvents': ['SOL'],
            'F_RoadTransport': ['TRO', 'TRF'],
            'G_Shipping': ['SHP'],
            'H_Aviation': ['AVT'],
            'I_OffRoad': ['TNR'],
            'J_Waste': ['SWD', 'WWT'],
            'K_AgriLivestock': ['PRO'],
            'L_AgriOther': ['AGS', 'AWB', 'N2O']
        }

    def _load_profiles(self):
        """Load and preprocess EDGAR temporal profiles for multiple years"""
        try:
            # Load monthly profiles for all years
            monthly_df = pd.read_excel(self.edgar_monthly, sheet_name='monthly & hourly temp profiles',
                                     usecols=['country', 'Year', 'IPCC_2006_source_category'] + 
                                             ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            # Filter for Germany using both country code formats (DEU and 22)
            monthly_df = monthly_df[((monthly_df['country'] == self.country) | 
                                   (monthly_df['country'] == '22') |
                                   (monthly_df['country'] == 22))]
            
            print("\nMonthly Profile Sample:")
            print(monthly_df[['country', 'Year', 'IPCC_2006_source_category', 'Jan', 'Jul']].head(6))
            
            # Load weekly profiles
            weekly_df = pd.read_csv(self.edgar_weekly, usecols=['Country_code_A3', 'activity_code', 'Weekday_id', 'daily_factor'])
            weekly_df = weekly_df[weekly_df['Country_code_A3'] == self.country]
            print("\nWeekly Profile Sample:")
            print(weekly_df[['activity_code', 'Weekday_id', 'daily_factor']].head(6))
            print(f"Available weekly activity codes: {weekly_df['activity_code'].unique()}")
            
            # Load hourly profiles
            expected_cols = ['Country_code_A3', 'activity_code', 'Daytype_id', 'month_id'] + [f'h{h}' for h in range(1, 25)]
            hourly_df = pd.read_csv(self.edgar_hourly, usecols=[col for col in expected_cols if col in pd.read_csv(self.edgar_hourly, nrows=0).columns])
            hourly_df = hourly_df[hourly_df['Country_code_A3'] == self.country]
            print("\nHourly Profile Sample (Transport Sector - Weekday):")
            transport_weekday = hourly_df[(hourly_df['activity_code'] == 'TRO') & 
                                       (hourly_df['Daytype_id'] == 1)]
            if not transport_weekday.empty:
                print(transport_weekday.iloc[0][[col for col in ['h7', 'h8', 'h9', 'h17', 'h18', 'h19'] if col in transport_weekday.columns]])
            print(f"Available hourly activity codes: {hourly_df['activity_code'].unique()}")
            
            # Load and merge weekend/weekday definitions
            weekend_days = pd.read_csv(self.weekend_types, sep=';', header=0)
            week_days = pd.read_csv(self.daytype_mapping, sep=';', header=0)
            week_days_per_country = weekend_days.merge(week_days, how='left', on=['Weekend_type_id'])
            
            # Normalize monthly profiles properly - ensure each profile sums to 1.0 across months
            month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            def normalize_monthly(group):
                row_sums = group[month_cols].sum(axis=1)
                # Avoid division by zero
                row_sums = row_sums.replace(0, 1)
                return group[month_cols].div(row_sums, axis=0)
            
            monthly_normalized = monthly_df.groupby(['country', 'Year', 'IPCC_2006_source_category']).apply(normalize_monthly)
            monthly_df[month_cols] = monthly_normalized.reset_index(level=[0, 1, 2], drop=True)
            monthly_df[month_cols] = monthly_df[month_cols].fillna(1.0 / len(month_cols))
            
            # Normalize weekly profiles properly - ensure each profile has proper daily factors
            weekly_df['daily_factor'] = weekly_df.groupby(['Country_code_A3', 'activity_code'])['daily_factor'].transform(
                lambda x: x / x.sum() * 7 if x.sum() > 0 else 1.0
            )
            
            # Normalize hourly profiles properly - ensure each profile sums to 1.0 across hours
            hour_cols = [f'h{h}' for h in range(1, 25)]
            if all(col in hourly_df.columns for col in hour_cols):
                def normalize_hourly(group):
                    row_sums = group[hour_cols].sum(axis=1)
                    # Avoid division by zero
                    row_sums = row_sums.replace(0, 1)
                    return group[hour_cols].div(row_sums, axis=0)
                
                hourly_normalized = hourly_df.groupby(['Country_code_A3', 'activity_code', 'Daytype_id', 'month_id']).apply(normalize_hourly)
                hourly_df[hour_cols] = hourly_normalized.reset_index(level=[0, 1, 2, 3], drop=True)
                hourly_df[hour_cols] = hourly_df[hour_cols].fillna(1.0 / len(hour_cols))
            else:
                print("Warning: Some hourly columns missing. Using uniform distribution.")
                for col in hour_cols:
                    if col not in hourly_df.columns:
                        hourly_df[col] = 1.0 / len(hour_cols)
                
            self.monthly_profiles = monthly_df
            self.weekly_profiles = weekly_df
            self.hourly_profiles = hourly_df
            self.week_days_per_country = week_days_per_country
            self.has_tz_id = False
            
            print(f"\nLoaded {len(monthly_df)} monthly profiles")
            print(f"Loaded {len(weekly_df)} weekly profiles")
            print(f"Loaded {len(hourly_df)} hourly profiles")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load profiles: {str(e)}")

    def _precompute_timezone_data(self):
        """Set minimal timezone data since no adjustments are needed"""
        self.is_multi_timezone = False

    def _precompute_cached_data(self):
        """Precompute and cache frequently used data for faster access"""
        # Precompute monthly profile lookup dictionaries
        self.monthly_profile_cache = {}
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for _, row in self.monthly_profiles.iterrows():
            key = (str(row['country']), int(row['Year']), str(row['IPCC_2006_source_category']))
            self.monthly_profile_cache[key] = row[month_cols].values.astype(np.float32)
        
        # Precompute weekly profile lookup dictionaries
        self.weekly_profile_cache = {}
        for _, row in self.weekly_profiles.iterrows():
            key = (str(row['Country_code_A3']), str(row['activity_code']), int(row['Weekday_id']))
            self.weekly_profile_cache[key] = float(row['daily_factor'])
        
        # Precompute hourly profile lookup dictionaries
        self.hourly_profile_cache = {}
        hour_cols = [f'h{h}' for h in range(1, 25)]
        for _, row in self.hourly_profiles.iterrows():
            key = (str(row['Country_code_A3']), str(row['activity_code']), int(row['Daytype_id']), int(row['month_id']))
            if all(col in row.index for col in hour_cols):
                self.hourly_profile_cache[key] = row[hour_cols].values.astype(np.float32)
            else:
                self.hourly_profile_cache[key] = np.ones(24, dtype=np.float32) / 24

    def _get_edgar_sector(self, greta_sector):
        """Get corresponding EDGAR sector for GRETA sector"""
        # Use the full GRETA sector name for mapping
        if greta_sector in self.sector_mapping:
            return self.sector_mapping[greta_sector]
        else:
            # Try to find the base sector name
            for sector_key in self.sector_mapping.keys():
                if greta_sector.startswith(sector_key):
                    return self.sector_mapping[sector_key]
            raise ValueError(f"No mapping found for GRETA sector {greta_sector}")

    def _get_activity_codes(self, greta_sector):
        """Get activity codes for a GRETA sector"""
        # Use the full GRETA sector name for mapping
        if greta_sector in self.activity_code_mapping:
            return self.activity_code_mapping[greta_sector]
        else:
            # Try to find the base sector name
            for sector_key in self.activity_code_mapping.keys():
                if greta_sector.startswith(sector_key):
                    return self.activity_code_mapping[sector_key]
            return ['TRO']  # Default to transport if not found

    def _get_profile_year(self, greta_sector):
        """Get the profile year for a GRETA sector"""
        # Use the full GRETA sector name for mapping
        if greta_sector in self.sector_year_mapping:
            return self.sector_year_mapping[greta_sector]
        else:
            # Try to find the base sector name
            for sector_key in self.sector_year_mapping.keys():
                if greta_sector.startswith(sector_key):
                    return self.sector_year_mapping[sector_key]
            return 0  # Default to year 0 if not found

    def _get_ipcc_categories(self, greta_sector):
        """Get IPCC categories for a GRETA sector"""
        # Use the full GRETA sector name for mapping
        if greta_sector in self.ipcc_category_mapping:
            return self.ipcc_category_mapping[greta_sector]
        else:
            # Try to find the base sector name
            for sector_key in self.ipcc_category_mapping.keys():
                if greta_sector.startswith(sector_key):
                    return self.ipcc_category_mapping[sector_key]
            return ['1A3b']  # Default to transport if not found

    def _get_daytype(self, date):
        """Determine day type (1=weekday, 2=saturday, 3=sunday)"""
        weekday = date.weekday()
        return 2 if weekday == 5 else 3 if weekday == 6 else 1

    def _apply_dst_adjust(self, hourly_factors, utc_reference, current_date):
        """Return hourly factors without any timezone or DST shift to preserve local time"""
        return hourly_factors

    def _get_time_factors(self, greta_sector, date):
        """Get monthly, daily, and hourly factors without timezone adjustments"""
        try:
            activity_codes = self._get_activity_codes(greta_sector)
            profile_year = self._get_profile_year(greta_sector)
            ipcc_categories = self._get_ipcc_categories(greta_sector)
            month = date.month
            daytype = self._get_daytype(date)
            
            # Get monthly factor using cached lookup
            monthly_factor_sum = 0.0
            monthly_count = 0
            
            for ipcc_category in ipcc_categories:
                key = (self.country, profile_year, ipcc_category)
                if key in self.monthly_profile_cache:
                    monthly_values = self.monthly_profile_cache[key]
                    monthly_factor_sum += monthly_values[month - 1]  # month-1 for 0-based index
                    monthly_count += 1
            
            if monthly_count > 0:
                monthly_factor = monthly_factor_sum / monthly_count
            else:
                monthly_factor = 1.0 / 12
            
            # Get weekly factor using cached lookup
            weekly_factors = []
            for activity_code in activity_codes:
                key = (self.country, activity_code, daytype)
                if key in self.weekly_profile_cache:
                    weekly_factors.append(self.weekly_profile_cache[key])
            
            if weekly_factors:
                weekly_factor = np.mean(weekly_factors)
            else:
                weekly_factor = 1.0 / 7
            
            # Normalize weekly factors for the month
            days_in_month = calendar.monthrange(date.year, date.month)[1]
            total_daily_factors = 0.0
            
            for day in range(1, days_in_month + 1):
                day_date = date.replace(day=day)
                day_daytype = self._get_daytype(day_date)
                
                day_weekly_factors = []
                for activity_code in activity_codes:
                    key = (self.country, activity_code, day_daytype)
                    if key in self.weekly_profile_cache:
                        day_weekly_factors.append(self.weekly_profile_cache[key])
                
                if day_weekly_factors:
                    total_daily_factors += np.mean(day_weekly_factors)
                else:
                    total_daily_factors += 1.0 / 7
            
            if total_daily_factors > 0:
                weekly_factor = weekly_factor / total_daily_factors
            else:
                weekly_factor = 1.0 / days_in_month
            
            # Get hourly factors using cached lookup
            hourly_factors_list = []
            for activity_code in activity_codes:
                key = (self.country, activity_code, daytype, month)
                if key in self.hourly_profile_cache:
                    hourly_factors_list.append(self.hourly_profile_cache[key])
            
            if hourly_factors_list:
                hourly_factors = np.mean(hourly_factors_list, axis=0)
                hourly_factors = self._apply_dst_adjust(hourly_factors, None, date)
            else:
                hourly_factors = np.ones(24) / 24
            
            annual_weight = monthly_factor * weekly_factor
            
            return annual_weight, hourly_factors
            
        except Exception as e:
            print(f"Error getting time factors for {greta_sector}: {str(e)}")
            # Return uniform distribution as fallback
            return 1.0 / (365 * 24), np.ones(24) / 24

    def _precompute_yearly_factors(self, sectors, year):
        """Precompute yearly factor sums for mass conservation check"""
        yearly_factor_sums = {}
        
        for sector in sectors:
            total_factor = 0.0
            for month in range(1, 13):
                days_in_month = calendar.monthrange(year, month)[1]
                for day in range(1, days_in_month + 1):
                    annual_weight, hourly_factors = self._get_time_factors(sector, datetime(year, month, day))
                    daily_contribution = annual_weight * sum(hourly_factors)
                    total_factor += daily_contribution
            yearly_factor_sums[sector] = total_factor
        
        return yearly_factor_sums

    def _check_mass_conservation(self, input_fn, sectors, year):
        """
        Verify mass conservation by comparing input yearly emissions with
        sum of temporally disaggregated emissions for the entire year
        """
        try:
            # Load input yearly emissions
            ds = gdal.Open(input_fn)
            if ds is None:
                print(f"Error: Could not open {input_fn}")
                return False
                
            yearly_emissions = []
            for i in range(1, ds.RasterCount + 1):
                band_data = ds.GetRasterBand(i).ReadAsArray()
                yearly_emissions.append(band_data)
            
            geotransform = ds.GetGeoTransform()
            x_size, y_size = ds.RasterXSize, ds.RasterYSize
            ds = None
            
            # Calculate total mass for each sector from yearly emissions
            yearly_mass = {}
            total_yearly_mass = 0.0
            
            for i, sector in enumerate(sectors):
                sector_mass = np.nansum(yearly_emissions[i])
                yearly_mass[sector] = sector_mass
                total_yearly_mass += sector_mass
            
            print(f"Total yearly mass: {total_yearly_mass:.6f} kg/m²")
            
            # Precompute yearly factor sums for all sectors
            yearly_factor_sums = self._precompute_yearly_factors(sectors, year)
            
            # Calculate expected mass from temporal disaggregation for entire year
            disaggregated_mass = {}
            total_disaggregated_mass = 0.0
            
            for i, sector in enumerate(sectors):
                sector_yearly_emissions = yearly_emissions[i]
                yearly_factor_sum = yearly_factor_sums[sector]
                
                # Calculate expected disaggregated mass
                sector_disaggregated_mass = np.nansum(sector_yearly_emissions) * yearly_factor_sum
                disaggregated_mass[sector] = sector_disaggregated_mass
                total_disaggregated_mass += sector_disaggregated_mass
                
                # Check mass conservation
                mass_diff = abs(yearly_mass[sector] - sector_disaggregated_mass)
                mass_rel_diff = mass_diff / yearly_mass[sector] * 100 if yearly_mass[sector] > 0 else 0
                
                if mass_rel_diff < 0.1:  # 0.1% tolerance
                    print(f"✓ Mass conserved for {sector}: {mass_rel_diff:.4f}% difference")
                else:
                    print(f"✗ Mass conservation warning for {sector}: {mass_rel_diff:.4f}% difference")
            
            # Check total mass conservation
            total_mass_diff = abs(total_yearly_mass - total_disaggregated_mass)
            total_mass_rel_diff = total_mass_diff / total_yearly_mass * 100 if total_yearly_mass > 0 else 0
            
            print(f"\nTotal yearly mass: {total_yearly_mass:.6f} kg/m²")
            print(f"Total disaggregated mass: {total_disaggregated_mass:.6f} kg/m²")
            print(f"Total mass difference: {total_mass_diff:.6f} kg/m²")
            print(f"Total relative difference: {total_mass_rel_diff:.4f}%")
            
            if total_mass_rel_diff < 0.1:
                print("✓ Total mass conservation verified")
                return True
            else:
                print("✗ Total mass conservation warning")
                return False
                
        except Exception as e:
            print(f"Error in mass conservation check: {str(e)}")
            return False

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

    def _process_day(self, current_date, sectors, yearly_bands):
        """Process a single day's data for all sectors - optimized version"""
        band_data = []
        band_descs = []
        
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
        
        return band_data, band_descs

    def _process_file_worker(self, args):
        """Worker function for processing files - avoids pickling issues"""
        input_fn, output_fn, sectors, start_date, end_date, x_size, y_size, geotransform, projection, output_files = args
        
        try:
            # Load all bands at once for better performance
            ds = gdal.Open(input_fn)
            yearly_bands = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)]
            ds = None
            
            file_days = (end_date - start_date).days + 1
            file_bands = 24 * len(sectors) * file_days
            
            print(f"\nCreating {output_fn} with {file_bands} bands "
                  f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            
            # Create output file with optimal settings
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_fn, x_size, y_size, file_bands, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'PREDICTOR=2', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS', 'TILED=YES']
            )
            out_ds.SetGeoTransform(geotransform)
            out_ds.SetProjection(projection)
            
            band_idx = 1
            min_max_values = {}
            
            # Process days sequentially
            current_date = start_date
            while current_date <= end_date:
                band_data, band_descs = self._process_day(current_date, sectors, yearly_bands)
                
                # Write all bands for this day at once
                for i, (data, desc) in enumerate(zip(band_data, band_descs)):
                    out_ds.GetRasterBand(band_idx + i).WriteArray(data)
                    out_ds.GetRasterBand(band_idx + i).SetDescription(desc)
                    
                    # Only calculate min/max for first and last few bands to save time
                    if i < 5 or i >= len(band_descs) - 5:
                        min_max_values[desc] = {
                            'min': np.nanmin(data),
                            'max': np.nanmax(data),
                            'mean': np.nanmean(data)
                        }
                
                band_idx += len(band_data)
                current_date += timedelta(days=1)
            
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
            print(f"Successfully created: {output_fn}")
            return min_max_values
        except Exception as e:
            print(f"Error processing {output_fn}: {str(e)}")
            return None

    def _process_temporal_disaggregation(self, input_fn, base_output_fn, sectors, start_date, end_date):
        """Perform temporal disaggregation with sequential processing to avoid pickling issues"""
        try:
            ds = gdal.Open(input_fn)
            if ds is None:
                print(f"Error: Could not open {input_fn}")
                return False
                
            geotransform = ds.GetGeoTransform()
            x_size, y_size = ds.RasterXSize, ds.RasterYSize
            projection = ds.GetProjection()
            ds = None
            
            # Perform detailed mass conservation check
            year = start_date.year
            print(f"\n=== Performing Detailed Mass Conservation Check ===")
            mass_conserved = self._check_mass_conservation(input_fn, sectors, year)
            
            if not mass_conserved:
                print("Warning: Mass conservation check failed!")
            
            output_files, days_per_file = self._create_output_files(
                input_fn, base_output_fn, sectors, start_date, end_date)
            
            min_max_values = {}
            
            # Process files sequentially to avoid pickling issues
            for output_fn, file_start, file_end in output_files:
                args = (input_fn, output_fn, sectors, file_start, file_end, x_size, y_size, geotransform, projection, output_files)
                result = self._process_file_worker(args)
                if result:
                    min_max_values.update(result)
            
            print(f"\nTemporal variation for {os.path.basename(base_output_fn)}:")
            for band, values in list(min_max_values.items())[:5] + list(min_max_values.items())[-5:]:
                print(f"{band}: min={values['min']:.9f}, max={values['max']:.9f}, mean={values['mean']:.9f}")
            
            # Add mass conservation status to output
            if mass_conserved:
                print("Mass conservation verified")
            else:
                print("Mass conservation warnings present")
            
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