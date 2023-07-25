# downscale_emissions_local


## Description
This code repository downscales a gridded emission inventory using surface characteristics such as land cover types, population density and street types. The sector specific emissions are downscaled using a similar method as used in the [UrbEm Hybrid Method to Derive High-Resolution Emissions for City-Scale Air Quality Modeling](https://www.mdpi.com/2073-4433/12/11/1404).

## Input data
The following data is required to downscale a emission inventory using this tool.

1. Emission inventory
	* GRETA Emission inventory - point and gridded data as Geodatabse.
	* TNO Emisison inventory
2. Population density
	* GHSL - Global Human Settlement Raster with 100 m resolution.
	* [Download](https://ghsl.jrc.ec.europa.eu/download.php?ds=pop)
3. CORINE Land Cover
	* Geodatabse land cover dataset with 100 m resolution.
	* Surface characteristic used to downscale emissions for specific sectors.
	* [Download](https://land.copernicus.eu/pan-european/corine-land-cover)
4. EDGAR high resolution temporal profiles
	* Temporally distribute emissions for specific sectors and country location.
	* **Not yet implemented**
	* [Download](https://edgar.jrc.ec.europa.eu/dataset_temp_profile)
5. OpenStreetMap(OSM)
	* Retrieved using the OSMnx API

## Usage
* To use this tool, one can use the default configuration file (`default_config.yaml`) as a template to provide the input parameters. The input parameters are described in the table below.
* The code must be excuted by providing the name of the configuration file in the command file. For example, `python main.py -c default_config`.

## Configuration file

| **Parameter** | **Description**|
| --- | --- |
| **data_settings** | |
| emiss_dir  | Path to GRETA Geodatabase or TNO files. |
| corine_dir | Path to CORINE Geodatabase. |
| popul_dir  | Path to population density raster. |
| edagr_dir  | Path to EDGAR temporal profile. |
| **job_settings** | | 
| job_path | Path to simulation data. |
| job_name | Simulation name. |
| min_lon | Min. longitude of model domain. |
| max_lon | Max. longitude of model domain. |
| min_lat | Min. latitude of model domain. |
| max_lat | Max. latitude of model domain. |
| resol | Domain resolution (meters). |
| epsg_code | EPSG of domain. |
| species | List of species to be downscaled. **Hint:** GRETA requires both PM10  & PM2.5 |
| country | Country location of model domain. |
| inventory | Emission inventory used. `GRETA or TNO` |


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## TO-DO
* Add code to prepare TNO emissions as input
* Add option to temporally distribute emissions
* Add checks to verify input parameter in configuration file.
* Add option to merge point sources and gridded data
* Add aerosol size distribution
