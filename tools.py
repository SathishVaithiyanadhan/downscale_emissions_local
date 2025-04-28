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

        # Add EDGAR file paths
        data_parameters['edgar_hourly'] = os.path.join(data_parameters['edgar_dir'], 'auxiliary_tables/hourly_profiles.csv')
        data_parameters['edgar_weekly'] = os.path.join(data_parameters['edgar_dir'], 'auxiliary_tables/weekly_profiles.csv')
        data_parameters['edgar_monthly'] = os.path.join(data_parameters['edgar_dir'], 'EDGAR_temporal_profiles_r1.xlsx')
        data_parameters['weekend_types'] = os.path.join(data_parameters['edgar_dir'], 'auxiliary_tables/weekenddays.csv')
        data_parameters['daytype_mapping'] = os.path.join(data_parameters['edgar_dir'], 'auxiliary_tables/weekdays.csv')

    # Add explicit bbox validation
    data_parameters['bbox'] = [
        job_parameters['min_lon'],
        job_parameters['min_lat'],
        job_parameters['max_lon'],
        job_parameters['max_lat']
    ]
    print(f"Validated BBOX: {data_parameters['bbox']} (EPSG:{job_parameters['epsg_code']})")

    return data_parameters, job_parameters
