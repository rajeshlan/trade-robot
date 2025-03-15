#python exchanges\synchronize_exchange_time.py

import ntplib
import logging
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synchronize_system_time(retries=3):
    """
    Synchronize system time with an NTP server, with retries and alternate servers.
    Returns the time offset in milliseconds.
    """
    ntp_servers = ['pool.ntp.org', 'time.google.com', 'time.windows.com']
    logging.info("Starting system time synchronization...")

    for attempt in range(retries):
        logging.info(f"Attempt {attempt + 1} of {retries}.")
        for server in ntp_servers:
            try:
                response = ntplib.NTPClient().request(server, timeout=5)
                # Convert NTP time to UTC datetime
                current_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
                local_time = datetime.now(timezone.utc)
                offset = (response.tx_time - local_time.timestamp()) * 1000  # Convert to milliseconds
                logging.info(f"System time synchronized: {current_time} using server {server}")
                return offset
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for server {server}: {e}")

    # Handle all failures
    logging.error("All attempts to synchronize time failed. Using zero offset as fallback.")
    return 0  # Return zero offset if synchronization fails

if __name__ == "__main__":
    # Call the function and log the result
    offset = synchronize_system_time()
    if offset == 0:
        logging.warning("Time offset is zero. Ensure system time is accurate to avoid API errors.")
    logging.info(f"Time offset: {offset:.2f} ms")
