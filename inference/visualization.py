import pandas as pd


def process_storm_data(input_csv, output_csv, output_ori, predicted_latitudes, predicted_longitudes):
    """
    Reads a CSV file, extracts specific columns, and appends predicted latitudes and longitudes.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the processed CSV file.
        output_ori (str): Path to save the original extracted CSV file.
        predicted_latitudes (list[float]): Predicted latitude values to append.
        predicted_longitudes (list[float]): Predicted longitude values to append.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv).head(8)
    df_ori = pd.read_csv(input_csv).head(10)

    # Extract required columns
    columns_to_keep = ['Storm Name', 'Latitude (°N)', 'Longitude (°E)']
    if not all(col in df.columns for col in columns_to_keep):
        raise ValueError(f"One or more required columns {columns_to_keep} are missing in the input CSV.")

    extracted_df = df[columns_to_keep].copy()
    extracted_df_ori = df_ori[columns_to_keep].copy()

    # Ensure predicted data has the same length
    if len(predicted_latitudes) != len(predicted_longitudes):
        raise ValueError("Predicted latitudes and longitudes must have the same length.")

    # Create a DataFrame for predicted data
    predicted_data = pd.DataFrame({
        'Storm Name': extracted_df['Storm Name'][0],
        'Latitude (°N)': predicted_latitudes,
        'Longitude (°E)': predicted_longitudes
    })

    # Append predicted data using pd.concat
    updated_df = pd.concat([extracted_df, predicted_data], ignore_index=True)

    # Save the updated DataFrame to a new CSV file
    updated_df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

    # Save the original extracted data to another CSV file
    extracted_df_ori.to_csv(output_ori, index=False)
    print(f"Processed data saved to {output_ori}")


# Example usage
input_csv_path = '../train/best_track_records_test.csv'
output_csv_path = '../inference/predicted_typhoon_path.csv'
output_ori_path = '../inference/original_typhoon_path.csv'
csv_path = "../train/best_track_records_test.csv"
sp_json_path = "../data_preprocess/sp_data_matrix.json"
sst_json_path = "../data_preprocess/sst_data_matrix.json"
typhoon_name = "Carmen"
# Assuming infer_typhoon_next_track is a function that returns predicted_latitudes and predicted_longitudes
predicted_latitudes, predicted_longitudes = infer_typhoon_next_track(typhoon_name, csv_path, sp_json_path,
                                                                     sst_json_path)
process_storm_data(input_csv_path, output_csv_path, output_ori_path, predicted_latitudes, predicted_longitudes)

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 读取CSV文件并提取经纬度列
def plot_coordinates_on_map(csv_file, csv_file1, lat_col='Latitude (°N)', lon_col='Longitude (°E)'):
    data = pd.read_csv(csv_file)
    data1 = pd.read_csv(csv_file1)
    lats = data[lat_col]
    lons = data[lon_col]
    lats1 = data1[lat_col]
    lons1 = data1[lon_col]
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    min_lat -= 5
    max_lat += 5
    min_lon -= 5
    max_lon += 5
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(5, 5))
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='k', alpha=0.5, linestyle='--')
    gl.xlabel_style={'size':5}
    gl.ylabel_style={'size':5}
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, edgecolor='blue')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')
    ax.add_feature(cfeature.OCEAN)
    ax.scatter(lons1, lats1, color='yellow', s=10, transform=ccrs.PlateCarree(), label='predicted')
    ax.scatter(lons, lats, color='red', s=10, transform=ccrs.PlateCarree(), label='original')
    ax.legend(loc='lower right')

    plt.title("Carmen")
    plt.savefig('Carmen.png')

ori_csv_file_path = "../inference/original_typhoon_path.csv"
pre_csv_file_path = "../inference/predicted_typhoon_path.csv"
plot_coordinates_on_map(ori_csv_file_path, pre_csv_file_path, lat_col='Latitude (°N)', lon_col='Longitude (°E)')