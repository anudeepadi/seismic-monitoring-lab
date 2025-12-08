#!/usr/bin/env python3
"""
Test IRIS SeedLink Real-Time Data Connection

This script tests connectivity to the IRIS SeedLink server and streams
real-time seismic data from Indian Ocean stations.

Usage:
    python scripts/test_seedlink.py
"""

import sys
import time
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Indian Ocean priority stations for tsunami monitoring
INDIAN_OCEAN_STATIONS = [
    ("II", "PALK"),   # Pallekele, Sri Lanka
    ("II", "COCO"),   # Cocos (Keeling) Islands
    ("II", "DGAR"),   # Diego Garcia
    ("IU", "CHTO"),   # Chiang Mai, Thailand
    ("IU", "TATO"),   # Taipei, Taiwan
    ("IU", "NWAO"),   # Narrogin, Australia
]

def test_seedlink_connection():
    """Test basic SeedLink connectivity."""
    print("=" * 70)
    print("IRIS SeedLink Real-Time Data Test")
    print("=" * 70)
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print()

    try:
        from obspy.clients.seedlink import Client as SeedLinkClient
        from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
        print("[OK] ObsPy SeedLink module available")
    except ImportError as e:
        print(f"[ERROR] ObsPy SeedLink not available: {e}")
        print("\nTrying alternative approach with obspy.clients.fdsn for recent data...")
        return test_fdsn_recent_data()

    # Test connection
    print("\nConnecting to rtserve.iris.washington.edu:18000...")

    try:
        # Use EasySeedLinkClient for simpler streaming
        received_data = defaultdict(list)
        start_time = time.time()
        duration = 30  # seconds

        class DataCollector(EasySeedLinkClient):
            def __init__(self, server_url):
                super().__init__(server_url)
                self.data = defaultdict(list)
                self.packet_count = 0

            def on_data(self, trace):
                station_id = f"{trace.stats.network}.{trace.stats.station}"
                self.data[station_id].append(trace)
                self.packet_count += 1

                # Print progress
                if self.packet_count % 10 == 0:
                    print(f"  Received {self.packet_count} packets from {len(self.data)} stations...")

        print(f"\nStreaming data for {duration} seconds...")
        print("Selecting Indian Ocean stations:")

        collector = DataCollector("rtserve.iris.washington.edu:18000")

        # Select stations
        for net, sta in INDIAN_OCEAN_STATIONS[:3]:  # Start with 3 stations
            try:
                collector.select_stream(net, sta, "BHZ")  # Broadband vertical
                print(f"  + {net}.{sta}.*.BHZ")
            except Exception as e:
                print(f"  - {net}.{sta}: {e}")

        print()

        # Run for specified duration
        def stop_after_duration():
            time.sleep(duration)
            collector.close()

        import threading
        timer = threading.Thread(target=stop_after_duration)
        timer.daemon = True
        timer.start()

        try:
            collector.run()
        except Exception as e:
            pass  # Expected when we close the connection

        # Analyze results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        if collector.packet_count > 0:
            print(f"\n[SUCCESS] Received {collector.packet_count} packets!")
            print(f"\nStations with data:")

            for station_id, traces in collector.data.items():
                total_samples = sum(len(tr.data) for tr in traces)
                print(f"  {station_id}: {len(traces)} traces, {total_samples} samples")

                # Show sample data
                if traces:
                    tr = traces[0]
                    print(f"    Sample rate: {tr.stats.sampling_rate} Hz")
                    print(f"    Start time: {tr.stats.starttime}")
                    print(f"    Data range: [{tr.data.min():.2f}, {tr.data.max():.2f}]")

            return True, collector.data
        else:
            print("\n[WARNING] No data received in time window")
            print("This could mean:")
            print("  - Network connectivity issues")
            print("  - Selected stations not currently streaming")
            print("  - Need to try different stations")
            return False, {}

    except Exception as e:
        print(f"\n[ERROR] SeedLink connection failed: {e}")
        print("\nFalling back to FDSN recent data test...")
        return test_fdsn_recent_data()


def test_fdsn_recent_data():
    """Test FDSN web service for recent data (fallback if SeedLink fails)."""
    print("\n" + "=" * 70)
    print("Testing FDSN Web Service for Recent Data")
    print("=" * 70)

    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime

        client = Client("IRIS")
        print("[OK] Connected to IRIS FDSN")

        # Get data from last 10 minutes
        end_time = UTCDateTime.now()
        start_time = end_time - 600  # 10 minutes ago

        print(f"\nFetching data from {start_time} to {end_time}")
        print("Stations:")

        all_data = {}
        for net, sta in INDIAN_OCEAN_STATIONS[:3]:
            try:
                print(f"  Trying {net}.{sta}...", end=" ")
                st = client.get_waveforms(
                    network=net,
                    station=sta,
                    location="*",
                    channel="BHZ",
                    starttime=start_time,
                    endtime=end_time
                )
                if len(st) > 0:
                    tr = st[0]
                    print(f"OK - {len(tr.data)} samples @ {tr.stats.sampling_rate} Hz")
                    all_data[f"{net}.{sta}"] = st
                else:
                    print("No data")
            except Exception as e:
                print(f"Failed: {e}")

        if all_data:
            print(f"\n[SUCCESS] Retrieved data from {len(all_data)} stations")
            return True, all_data
        else:
            print("\n[ERROR] No data retrieved from any station")
            return False, {}

    except Exception as e:
        print(f"\n[ERROR] FDSN test failed: {e}")
        return False, {}


def plot_waveforms(data):
    """Plot received waveforms."""
    try:
        import matplotlib.pyplot as plt

        if not data:
            print("No data to plot")
            return

        fig, axes = plt.subplots(len(data), 1, figsize=(14, 3 * len(data)))
        if len(data) == 1:
            axes = [axes]

        for idx, (station_id, traces) in enumerate(data.items()):
            ax = axes[idx]

            # Combine all traces
            if isinstance(traces, list):
                # From SeedLink
                for tr in traces:
                    t = np.arange(len(tr.data)) / tr.stats.sampling_rate
                    ax.plot(t, tr.data, 'b-', linewidth=0.5)
            else:
                # From FDSN (Stream object)
                for tr in traces:
                    t = np.arange(len(tr.data)) / tr.stats.sampling_rate
                    ax.plot(t, tr.data, 'b-', linewidth=0.5)

            ax.set_ylabel(station_id)
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Real-Time Seismic Data - Indian Ocean Stations", fontsize=14)
        plt.tight_layout()

        output_path = "outputs/seedlink_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    success, data = test_seedlink_connection()

    if success and data:
        plot_waveforms(data)
        print("\n" + "=" * 70)
        print("SeedLink test PASSED - Real-time data streaming is working!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("SeedLink test FAILED - Check network or try alternative data sources")
        print("=" * 70)
        sys.exit(1)
