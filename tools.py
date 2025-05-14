import os
import yaml
import sys
import getopt

def config_parameters():
    configname = ''
    if sys.argv[0].endswith('pydevconsole.py'):
        configname = "example_a.yaml"
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hc:w)", ["help=", "config="])
        except getopt.GetoptError as err:
            print("Error:", err)
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                sys.exit(0)
            elif opt in ("-c", "--config"):
                configname = arg + ".yaml"

    if configname == '':
        print("This script requires input parameter -c <config_name>")
        exit(2)

    print('Reading Configuration File...')
    with open(configname, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        data_parameters = config['data_settings']
        job_parameters = config['job_settings']

        # Add EDGAR file paths with validation
        required_temporal_files = {
            'edgar_hourly': 'auxiliary_tables/hourly_profiles.csv',
            'edgar_weekly': 'auxiliary_tables/weekly_profiles.csv',
            'weekend_types': 'auxiliary_tables/weekenddays.csv',
            'daytype_mapping': 'auxiliary_tables/weekdays.csv',
            'edgar_monthly': 'EDGAR_temporal_profiles_r1.xlsx',
            'greta_to_edgar': 'emission_time_factors.xlsx'
        }

        for key, rel_path in required_temporal_files.items():
            full_path = os.path.join(data_parameters['edgar_dir'], rel_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Temporal profile file not found: {full_path}")
            data_parameters[key] = full_path

    # Add explicit bbox validation
    data_parameters['bbox'] = [
        job_parameters['min_lon'],
        job_parameters['min_lat'],
        job_parameters['max_lon'],
        job_parameters['max_lat']
    ]
    print(f"Validated BBOX: {data_parameters['bbox']} (EPSG:{job_parameters['epsg_code']})")

    return data_parameters, job_parameters