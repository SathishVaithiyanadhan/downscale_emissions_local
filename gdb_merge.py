
#!/usr/bin/env python3
# merging the multiple layers of the UrbanAtlas_Germany.gdb into a single layer was very complicated, so i found to use this method (multi GDB to GPKG to singlelayer GDB).
#if you find a better way to do this, go for it. 
import shutil
import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkb
import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# INPUT / OUTPUT
# ===============================================================

input_gdb = "/hpc/gpfs2/home/u/vaithisa/UniA/GDB/UrbanAtlas_Germany.gdb"   #Raw data with multiple layers.
output_gdb = "/hpc/gpfs2/home/u/vaithisa/UniA/GDB/UrbanAtlas_Germany_singlelayer.gdb"   #Output GDB with single merged layer. Will be converted to FileGDB using GDAL command line.
output_layer = "UrbanAtlas_Merged"

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def fix_geometry(geom):
    """Fix invalid or mixed-dimension geometries"""
    if geom is None or geom.is_empty:
        return None
    
    try:
        # Fix invalid geometries
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        # Force 2D (remove Z coordinates if present)
        if geom.has_z:
            # Convert to 2D by dropping Z
            if geom.geom_type == 'Polygon':
                # Create new 2D polygon from exterior and interiors
                exterior_2d = [(x, y) for x, y, z in geom.exterior.coords]
                interiors_2d = [[(x, y) for x, y, z in interior.coords] 
                               for interior in geom.interiors]
                geom = Polygon(exterior_2d, interiors_2d)
            elif geom.geom_type == 'MultiPolygon':
                # Convert each polygon to 2D
                polygons_2d = []
                for poly in geom.geoms:
                    exterior_2d = [(x, y) for x, y, z in poly.exterior.coords]
                    interiors_2d = [[(x, y) for x, y, z in interior.coords] 
                                   for interior in poly.interiors]
                    polygons_2d.append(Polygon(exterior_2d, interiors_2d))
                geom = MultiPolygon(polygons_2d)
        
        # Ensure geometry is valid after fixing
        if not geom.is_valid:
            geom = geom.buffer(0)
            
        return geom
        
    except Exception as e:
        print(f"  Warning: Could not fix geometry: {e}")
        return None

def make_2d_gdf(gdf):
    """Convert entire GeoDataFrame to 2D geometries"""
    gdf['geometry'] = gdf['geometry'].apply(fix_geometry)
    # Drop rows with None geometry
    gdf = gdf.dropna(subset=['geometry'])
    return gdf

# ===============================================================
# CHECK INPUT
# ===============================================================

if not os.path.exists(input_gdb):
    raise FileNotFoundError(f"Input GDB not found:\n{input_gdb}")

# ===============================================================
# LIST ALL LAYERS
# ===============================================================

layers = fiona.listlayers(input_gdb)

print("\n===================================================")
print(f"Found {len(layers)} layers")
print("===================================================\n")

# ===============================================================
# READ AND STORE LAYERS
# ===============================================================

gdfs = []
failed_layers = []
reference_crs = None
total_valid_features = 0

for i, layer in enumerate(layers, 1):
    print(f"[{i}/{len(layers)}] Reading layer: {layer}")

    try:
        # Read with pyogrio (more robust for large data)
        gdf = gpd.read_file(input_gdb, layer=layer, engine='pyogrio')
        
        # Skip empty layers
        if len(gdf) == 0:
            print(f"  Skipping empty layer")
            continue
        
        # Fix geometry issues
        original_count = len(gdf)
        gdf = make_2d_gdf(gdf)
        
        if len(gdf) == 0:
            print(f" No valid geometries after fixing")
            continue
        
        # Store source layer name
        gdf['source_layer'] = layer
        
        # Set reference CRS
        if reference_crs is None:
            reference_crs = gdf.crs
            print(f"  Reference CRS: {reference_crs}")
        
        # Reproject if necessary
        if gdf.crs != reference_crs:
            print(f"  Reprojecting to reference CRS")
            gdf = gdf.to_crs(reference_crs)
        
        # Remove duplicate ID columns
        drop_cols = [col for col in gdf.columns 
                    if col.lower() in ['fid', 'objectid', 'oid']]
        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)
        
        # Reset index
        gdf = gdf.reset_index(drop=True)
        
        # Append
        gdfs.append(gdf)
        total_valid_features += len(gdf)
        print(f" Added {len(gdf)} features (skipped {original_count - len(gdf)} invalid)")
        
    except Exception as e:
        print(f" ERROR: {str(e)[:100]}")
        failed_layers.append(layer)

print(f"\n Successfully read {len(gdfs)} layers")
if failed_layers:
    print(f" Failed to read {len(failed_layers)} layers")

# ===============================================================
# CHECK
# ===============================================================

if len(gdfs) == 0:
    raise RuntimeError("No valid layers found.")

# ===============================================================
# MERGE ALL LAYERS
# ===============================================================

print("\n===================================================")
print("Merging layers...")
print("===================================================\n")

# Use pandas concat for large datasets
merged_gdf = pd.concat(gdfs, ignore_index=True, sort=False)

# Convert back to GeoDataFrame
merged_gdf = gpd.GeoDataFrame(
    merged_gdf,
    geometry='geometry',
    crs=reference_crs
)

# Final cleanup
merged_gdf = merged_gdf.reset_index(drop=True)

# Remove any remaining invalid geometries
merged_gdf['geometry'] = merged_gdf['geometry'].apply(fix_geometry)
merged_gdf = merged_gdf.dropna(subset=['geometry'])

print(f"Merged features: {len(merged_gdf)}")

# ===============================================================
# DELETE OUTPUT IF EXISTS
# ===============================================================

if os.path.exists(output_gdb):
    print(f"\nRemoving existing output:\n{output_gdb}")
    shutil.rmtree(output_gdb)

# ===============================================================
# WRITE OUTPUT - TRY MULTIPLE FORMATS
# ===============================================================

print("\n===================================================")
print("Writing merged data...")
print("===================================================\n")

# Try different output formats in order of preference
output_formats = [
    ('GPKG', f"{output_gdb.replace('.gdb', '.gpkg')}", 'GeoPackage'),
    ('GeoJSON', f"{output_gdb.replace('.gdb', '.geojson')}", 'GeoJSON'),
    ('ESRI Shapefile', f"{output_gdb.replace('.gdb', '.shp')}", 'Shapefile')
]

success = False
for driver, output_path, format_name in output_formats:
    try:
        print(f"Trying {format_name} ({driver})...")
        
        # Remove if exists
        if os.path.exists(output_path):
            if driver == 'ESRI Shapefile':
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    shp_file = output_path.replace('.shp', ext)
                    if os.path.exists(shp_file):
                        os.remove(shp_file)
            else:
                os.remove(output_path)
        
        # Write file
        merged_gdf.to_file(output_path, driver=driver, layer=output_layer)
        
        print(f" Success! Saved as {format_name}")
        print(f"  Output: {output_path}")
        print(f"  Total features: {len(merged_gdf)}")
        success = True
        break
        
    except Exception as e:
        print(f" Failed with {driver}: {str(e)[:100]}")
        continue

# ===============================================================
# FINISHED
# ===============================================================

if success:
    print("\n===================================================")
    print(" DONE")
    print("===================================================\n")
    print(f"CRS: {merged_gdf.crs}")
    print(f"Columns: {list(merged_gdf.columns)}")
    print(f"Geometry type: {merged_gdf.geometry.geom_type.unique()}")
else:
    print("\n All output formats failed!")
    print("Saving as pickle for debugging...")
    merged_gdf.to_pickle(f"{input_gdb.replace('.gdb', '')}_merged.pkl")
    print("Saved as pickle file for further analysis.")


'''then perform:
# Step 2: Convert GeoPackage to FileGDB using GDAL command line
ogr2ogr -f OpenFileGDB \
    /hpc/gpfs2/home/u/vaithisa/UniA/GDB/UrbanAtlas_Germany_singlelayer.gdb \
    /hpc/gpfs2/home/u/vaithisa/UniA/GDB/UrbanAtlas_Germany_singlelayer.gpkg \
    -nln UrbanAtlas_Merged \
    -lco COMPRESSION=YES'''