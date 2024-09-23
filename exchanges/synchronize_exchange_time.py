import ntplib
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synchronize_system_time(retries=3):
    """
    Synchronize system time with an NTP server, with retries and alternate servers.
    Returns the time offset in milliseconds.
    """
    ntp_servers = ['pool.ntp.org', 'time.google.com', 'time.windows.com']
    for attempt in range(retries):
        for server in ntp_servers:
            try:
                response = ntplib.NTPClient().request(server, timeout=5)
                current_time = datetime.fromtimestamp(response.tx_time)
                local_time = datetime.utcnow()
                offset = (current_time - local_time).total_seconds() * 1000  # Convert to milliseconds
                logging.info(f"System time synchronized: {current_time} using server {server}")
                return offset
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for server {server}: {e}")
    logging.error("All attempts to synchronize time failed")
    return 0  # Return zero offset if synchronization fails
