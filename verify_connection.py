import time
try:
    from cortex import CortexClient
    print("Successfully imported CortexClient from actiancortex package.")
except ImportError:
    print("Error: 'actiancortex' package not found. Please install the .whl from the repo.")
    exit(1)

def verify_connection(host="localhost", port=50051):
    """
    Verifies connection to the Actian VectorAI DB.
    """
    addr = f"{host}:{port}"
    print(f"Connecting to Actian VectorAI DB at {addr}...")
    
    try:
        # Connect using context manager
        with CortexClient(addr) as client:
            # Health check
            version, uptime = client.health_check()
            print("--- Connection Successful! ---")
            print(f"Database Version: {version}")
            print(f"System Uptime: {uptime}")
            
            # List collections (just to see if it works)
            collections = client.list_collections()
            print(f"Found {len(collections)} collections.")
            
    except Exception as e:
        print(f"--- Connection Failed ---")
        print(f"Error: {str(e)}")
        print("Make sure your Docker container is running:")
        print(f"docker run -d --name actian-vectorai -p {port}:50051 williamimoh/actian-vectorai-db:1.0b")

if __name__ == "__main__":
    verify_connection()
