#!/usr/bin/env python3
"""
Simple test script to verify HLS stream availability
"""

import requests
from urllib.parse import quote

def test_hls_stream(camera_name, host="127.0.0.1", port=8888):
    """Test if HLS stream is available for a camera"""
    encoded_name = quote(camera_name)
    hls_url = f"http://{host}:{port}/{encoded_name}/index.m3u8"
    
    print(f"Testing HLS stream for camera: {camera_name}")
    print(f"URL: {hls_url}")
    
    try:
        response = requests.head(hls_url, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        
        if response.status_code == 200:
            print("✅ HLS stream is available!")
            return True
        else:
            print(f"❌ HLS stream not available (HTTP {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to MediaMTX server (connection refused)")
        print("   - Check if MediaMTX is running")
        print("   - Check if port 8888 is open")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=== HLS Stream Test ===\n")
    
    # Test cameras from MediaMTX config
    cameras = [
        "Bardi-Carport",
        "Bardi-Kamar", 
        "Bardi-Living-Room",
        "D1-CAM-1",
        "DS-CAM-2"
    ]
    
    results = {}
    for camera in cameras:
        print("-" * 50)
        results[camera] = test_hls_stream(camera)
        print()
    
    print("=== SUMMARY ===")
    available_count = sum(results.values())
    total_count = len(results)
    
    print(f"Available: {available_count}/{total_count} cameras")
    
    for camera, available in results.items():
        status = "✅ Available" if available else "❌ Unavailable"
        print(f"  {camera}: {status}")
    
    if available_count == 0:
        print("\n💡 Troubleshooting:")
        print("1. Make sure MediaMTX is running")
        print("2. Check MediaMTX logs for errors")
        print("3. Verify camera RTSP connections")
        print("4. Check MediaMTX HLS configuration")

if __name__ == "__main__":
    main()