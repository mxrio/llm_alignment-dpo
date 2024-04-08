import time
import psutil
import os
import subprocess
import datetime
import csv

# Function to measure CPU, memory, swap, and network usage
def measure_system_performance():
    cpu_percent = psutil.cpu_percent(interval=0)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    swap = psutil.swap_memory()
    swap_percent = swap.percent
    network_io = psutil.net_io_counters()
    return cpu_percent, memory_percent, swap_percent, network_io.bytes_sent, network_io.bytes_recv

# Function to measure GPU usage (if available)
def measure_gpu_performance():
    try:
        gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')
        gpu_usage = [int(utilization) for utilization in gpu_usage]
    except Exception as e:
        print("Error while measuring GPU usage:", e)
        gpu_usage = [0] * psutil.cpu_count()
    return gpu_usage

# Function to write data to CSV
def write_to_csv(data, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

# Main function
def main(measurement_duration):
    start_time = time.time()

    # Create CSV file for logging
    csv_filename = "measurement_logs.csv"
    csv_header = ['Timestamp', 'Duration', 'CPU Usage (%)', 'Memory Usage (%)', 'Swap Usage (%)', 'GPU Usage (%)', 'Network Sent (bytes)', 'Network Received (bytes)']
    if not os.path.isfile(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

    # Measure system performance
    while (time.time() - start_time) < measurement_duration:
        cpu_percent, memory_percent, swap_percent, bytes_sent, bytes_received = measure_system_performance()
        gpu_usage = measure_gpu_performance()
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = round(time.time() - start_time, 2)
        data_row = [current_time, duration, cpu_percent, memory_percent, swap_percent] + gpu_usage + [bytes_sent, bytes_received]
        write_to_csv(data_row, csv_filename)
        print('Measurement cycle completed. Data is stored.')
        time.sleep(5)  # Adjust sleep time to control the measurement frequency

if __name__ == "__main__":
    measurement_duration = 60  # Duration in seconds for measurement
    main(measurement_duration)