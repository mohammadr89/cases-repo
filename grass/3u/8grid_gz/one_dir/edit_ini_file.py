import os
import re

total_simulation_time = 86400  # 24 hours in seconds
segment_end_times = [
    10800,   # 3 hours (segment 1 ; 0-3)
    21600,   # 3 hours (segment 2 ; 3-6)
    32400,   # 3 hours (segment 3 ; 6-9)
    43200,   # 3 hours (segment 4 ; 9-12)
    54000,   # 3 hours (segment 5 ; 12-15)
    64800,   # 3 hours (segment 6 ; 15-18)
    75600,   # 3 hours (segment 7 ; 18-21)
    86400    # 3 hours (segment 8 ; 21-24)
]

file_path = os.path.join(os.getcwd(), 'plume_chem.ini')
with open(file_path, 'r') as file:
    ini_content = file.read()

endtime_match = re.search(r'endtime=(\d+)', ini_content)
if endtime_match:
    current_endtime = int(endtime_match.group(1))
else:
    current_endtime = total_simulation_time
    print(f"Could not detect endtime in ini file, using default: {current_endtime} seconds")

# Create the 9 ini files
ini_contents = []
start_time = 0

for i, end_time in enumerate(segment_end_times):
    segment_content = ini_content
    segment_content = re.sub(r'starttime=\d+', f'starttime={start_time}', segment_content)
    segment_content = re.sub(r'endtime=\d+', f'endtime={end_time}', segment_content)
    hours = start_time / 3600  # Convert seconds to hours
    segment_content = re.sub(r'start_hour=\d+', f'start_hour={int(hours)}', segment_content)
    ini_contents.append(segment_content)
    start_time = end_time

# Write all the files
for i, content in enumerate(ini_contents):
    file_path = os.path.join(os.getcwd(), f'plume_chem{i+1}.ini')
    with open(file_path, 'w') as file:
        file.write(content)
