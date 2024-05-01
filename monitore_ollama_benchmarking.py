import subprocess
import time

# Define the command to run your Python script
# Replace 'python script.py' with the actual command to run your script
command = 'python benchmark_bbq_pipeline_ollama.py'

# Define the time interval for checking the output (in seconds)
check_interval = 15  # Check every minute

def monitor_process(command, check_interval):
    start_time = time.time()

    while True:
        # Start the process and capture its output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('Started Benchmarking')

        # Read the output while the process is running
        while process.poll() is None:
            print('1')
            output = process.stdout.readline()
            print('2')
            print('Output:', output.decode().strip())
            if output:
                print(output.decode().strip())  # Print the output to the console
            time.sleep(0.1)  # Sleep briefly to avoid consuming too much CPU

            # Check the output at regular intervals
            if time.time() - start_time >= check_interval:
                start_time = time.time()
                if not output:
                    print("No output received. Restarting the process...")
                    process.kill()  # Kill the process if it's hanging
                    break  # Exit the inner loop to restart the process

        # If the process exits, print the exit code and restart it
        exit_code = process.poll()
        print(f"Process exited with code {exit_code}. Restarting...")
        time.sleep(1)  # Wait for a moment before restarting to avoid rapid restarts

# Start monitoring the process
monitor_process(command, check_interval)
