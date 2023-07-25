"""
"""
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
            opts, args = getopt.getopt(sys.argv[1:],"hc:w)",["help=","config="])
        except getopt.GetoptError as err:
            print("Error:", err)
            sys.exit(2)
        # parse options
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                sys.exit(0)
            elif opt in ("-c", "--config"):
                configname = arg + ".yaml"

    if configname == '':
        print("This script requires input parameter -c <config_name>")
        exit(2)

    # Read config file
    print('Reading Configuration File...')
    with open(configname, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        data_parameters = config['data_settings']
        job_parameters = config['job_settings']

    return data_parameters, job_parameters
