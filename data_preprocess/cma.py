import os
import pandas as pd

# Define the directory containing the txt files
directory = "/home/dl392/data/yiwei/typhoon/CMABSTdata"

# Prepare a list to store all best track records across files
all_best_tracks = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            
            # Extract the number of data lines and storm name from the header
            header_data = lines[0].split()
            num_data_lines = int(header_data[2])  # CCC field in the header
            storm_name = header_data[7]  # Storm name field in the header
            
            # Process each Best Track Record line (after the header)
            for i in range(1, 1 + num_data_lines):
                data = lines[i].split()
                
                # Extract fields from the Best Track Record
                date_time = data[0]  # YYYYMMDDHH
                intensity_category = int(data[1])  # I
                latitude = float(data[2]) * 0.1  # Convert LAT to degrees
                latitude = round(latitude, 1)
                longitude = float(data[3]) * 0.1  # Convert LONG to degrees
                longitude = round(longitude, 1)
                pressure = int(data[4])  # PRES
                max_sustained_wind = int(data[5])  # WND
                #other_wind = int(data[6])  # OWD
                
                # Add extracted data to all_best_tracks list
                all_best_tracks.append([
                    storm_name, date_time, intensity_category, latitude, longitude, pressure, max_sustained_wind
                ])

# Create a DataFrame to organize data for export
columns = ["Storm Name", "DateTime", "Intensity Category", "Latitude (°N)", "Longitude (°E)",
           "Pressure (hPa)", "Max Sustained Wind (m/s)"]
df = pd.DataFrame(all_best_tracks, columns=columns)

# Save the DataFrame to a CSV file
output_csv = "best_track_records.csv"
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"Best track records saved to {output_csv}")

