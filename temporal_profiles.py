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

    def apply_temporal_profiles(self, job_parameters, sectors):
        """Apply temporal profiles to downscaled emissions"""
        print('\n=== Starting Temporal Disaggregation ===')
        
        # Load EDGAR profiles
        (self.monthly_profiles, self.weekly_profiles, 
         self.hourly_profiles, self.weekend_defs, self.daytype_map) = self._load_profiles()
        
        main_sectors = [item[0] for item in sectors]
        start_date = datetime.strptime(job_parameters['temporal']['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(job_parameters['temporal']['end_date'], "%Y-%m-%d")
        delta = end_date - start_date
        
        for spec in tqdm(job_parameters['species'], desc="Processing species"):
            input_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_yearly.tif')
            output_fn = os.path.join(job_parameters['job_path'], f'emission_{spec}_temporal.tif')
            
            if not os.path.exists(input_fn):
                print(f"Warning: Input file not found: {input_fn}")
                continue
                
            try:
                ds = gdal.Open(input_fn)
                if ds is None:
                    print(f"Error: Could not open {input_fn}")
                    continue
                    
                yearly_bands = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, 14)]
                
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(
                    output_fn,
                    ds.RasterXSize,
                    ds.RasterYSize,
                    24 * len(main_sectors) * (delta.days + 1),
                    gdal.GDT_Float32,
                    options=['COMPRESS=LZW', 'PREDICTOR=2', 'BIGTIFF=YES']
                )
                out_ds.SetGeoTransform(ds.GetGeoTransform())
                out_ds.SetProjection(ds.GetProjection())
                
                band_idx = 1
                current_date = start_date
                
                # Track min/max values for verification
                min_max_values = {}

                for day in range(delta.days + 1):
                    for sec_idx, sec in enumerate(main_sectors):
                        yearly_emis = yearly_bands[sec_idx]
                        annual_weight, hourly_factors = self._get_time_factors(sec, current_date)
                        daily_emiss = yearly_emis * annual_weight
                        
                        for hour in range(24):
                            hour_emis = daily_emiss * hourly_factors[hour]
                            out_ds.GetRasterBand(band_idx).WriteArray(hour_emis)
                            out_ds.GetRasterBand(band_idx).SetDescription(
                                f"{sec}_h{hour+1}_{current_date.strftime('%Y%m%d')}")
                            # Track min/max values
                            band_desc = f"{sec}_h{hour+1}_{current_date.strftime('%Y%m%d')}"
                            min_max_values[band_desc] = {
                                'min': np.nanmin(hour_emis),
                                'max': np.nanmax(hour_emis),
                                'mean': np.nanmean(hour_emis)
                            }
                            band_idx += 1
                    
                    current_date += timedelta(days=1)
                # Print min/max values for verification
                print(f"\nTemporal variation for {spec}:")
                for band, values in list(min_max_values.items())[:5] + list(min_max_values.items())[-5:]:
                    print(f"{band}: min={values['min']:.9f}, max={values['max']:.9f}, mean={values['mean']:.9f}")

                metadata = {
                    'temporal_disaggregation': 'EDGAR',
                    'temporal_status': 'Hourly',
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'country': self.country,
                    'profile_year': str(self.profile_year),
                    'total_days': str(delta.days + 1)
                }
                out_ds.SetMetadata(metadata)
                
                ds = None
                out_ds = None
                print(f"\nSuccessfully created: {output_fn}")
                
            except Exception as e:
                print(f"\nError processing {spec}: {str(e)}")
                if 'ds' in locals(): ds = None
                if 'out_ds' in locals(): out_ds = None
                
        print("\n=== Mass Conserved Downscaling with Temporal Disaggregation Completed ===")

##########