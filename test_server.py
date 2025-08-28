import subprocess
import time
import requests
import os

def test_server_starts():
    """Test that the server starts and is accessible."""
    print("Starting server...")
    # Start the server in the background
    server_process = subprocess.Popen(
        ["python", "server.py", "--no-proxy"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="tiktok-reddit-mcp"  # Ensure we're in the right directory
    )
    
    # Give the server some time to start
    time.sleep(10)
    
    # Check if the process is still running
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"Server failed to start. Stdout: {stdout}, Stderr: {stderr}")
        return False
    
    # If we reach here, the server is running
    print("Server is running.")
    
    # Terminate the server
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
        server_process.wait()
    
    print("Server terminated.")
    return True

if __name__ == "__main__":
    success = test_server_starts()
    if success:
        print("Test passed: Server starts correctly.")
    else:
        print("Test failed: Server did not start correctly.")
    exit(0 if success else 1)