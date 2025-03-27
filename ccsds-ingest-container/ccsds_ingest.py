# REFERENCE: https://public.ccsds.org/Pubs/132x0b3.pdf

import json
import struct
import time
import logging
from datetime import datetime
from collections import defaultdict
from confluent_kafka import Consumer, KafkaError
from influxdb import InfluxDBClient

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ccsds_ingest")

with open('./ccsds_config/sample_config.json', 'r') as f:
    config = json.load(f)

ccsds_conf = config['ccsds']
aggregation_window = config.get('aggregation', {}).get('time_window', 1)

# expected packet length = header length + data field length + crc length
expected_packet_length = ccsds_conf['header']['length'] + ccsds_conf['data_field']['length'] + ccsds_conf.get('crc_length', 0)

def parse_field(data, offset, length, field_type):
    segment = data[offset:offset+length]
    if field_type == "uint8":
        return struct.unpack("B", segment)[0]
    elif field_type == "uint32":
        return struct.unpack(">I", segment)[0]  # big-endian unsigned int (assume big-endian format for CCSDS)
    elif field_type == "float":
        return struct.unpack(">f", segment)[0]  # big-endian float
    else:
        raise ValueError(f"Unsupported field type: {field_type}")

def parse_packet(packet_bytes):
    if len(packet_bytes) != expected_packet_length:
        logger.error("Packet length does not match expected length")
        return None

    # parse header
    header_conf = ccsds_conf['header']
    header = {}
    for field, spec in header_conf['fields'].items():
        offset = spec['offset']
        length = spec['length']
        try:
            header[field] = parse_field(packet_bytes, offset, length, spec['type'])
        except Exception as e:
            logger.error(f"Error parsing header field {field}: {e}")
            return None

    # use header timestamp if available or current time
    packet_time = header.get("timestamp")
    if not packet_time or packet_time == 0:
        packet_time = int(time.time())
    else:
        # if needed, convert the header timestamp to Unix seconds
        packet_time = int(packet_time)

    # parse data field
    data_field_conf = ccsds_conf['data_field']
    data_field = {}
    data_field_offset = header_conf['length']  # daata field starts immediately after header
    for field in data_field_conf['fields']:
        name = field['name']
        offset = data_field_offset + field['offset']
        length = field['length']
        try:
            data_field[name] = parse_field(packet_bytes, offset, length, field['type'])
        except Exception as e:
            logger.error(f"Error parsing data field {name}: {e}")

    # assume CRC error correction is valid

    # use satellite_id from header; if not present, assign one externally
    satellite_id = header.get("satellite_id", None)
    return {
        "time": packet_time,
        "satellite_id": satellite_id,
        "data": data_field
    }

# Buffer for aggregation - keyed by satellite_id and timestamp rounded to the aggregation window
aggregation_buffer = defaultdict(dict)

def aggregate_packet(parsed_packet):

    key = (parsed_packet["satellite_id"], parsed_packet["time"] // aggregation_window)

    # merge new fields into existing aggregated data
    aggregation_buffer[key]["time"] = parsed_packet["time"]
    aggregation_buffer[key]["satellite_id"] = parsed_packet["satellite_id"]

    if "data" not in aggregation_buffer[key]:
        aggregation_buffer[key]["data"] = {}

    aggregation_buffer[key]["data"].update(parsed_packet["data"])

def flush_aggregated_packets(influx_client):

    current_time = int(time.time())
    keys_to_flush = []
    for key, record in aggregation_buffer.items():
        # if the record is older than the aggregation window, flush it
        if current_time - record["time"] >= aggregation_window:
            write_to_influx(influx_client, record)
            keys_to_flush.append(key)

    # Remove flushed records from buffer
    for key in keys_to_flush:
        del aggregation_buffer[key]

def write_to_influx(client, record):
    # prepare InfluxDB point (using measurement "telemetry_data")
    json_body = [
        {
            "measurement": "telemetry_data",
            "tags": {
                "satellite_id": str(record["satellite_id"])
            },
            "time": datetime.utcfromtimestamp(record["time"]).isoformat() + "Z",
            "fields": {
                "data": json.dumps(record["data"])
            }
        }
    ]
    try:
        client.write_points(json_body)
        logger.info(f"Wrote record to InfluxDB: {json_body}")
    except Exception as e:
        logger.error(f"Error writing to InfluxDB: {e}")

def main():
    # Kafka consumer configuration
    consumer_conf = {
        'bootstrap.servers': 'kafka:9092',
        'group.id': 'ccsds_ingest_group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(['raw_telemetry'])  # topic that has raw telemetry streams

    # Connect to InfluxDB - influx binds to port 8086 in its own Dockerfile
    influx_client = InfluxDBClient(host='influx_telegraf', port=8086, database='telemetry_db')

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # no message received, flush any aggregated packets periodically.
                flush_aggregated_packets(influx_client)
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error("Kafka error: {}".format(msg.error()))
                    continue

            raw_bytes = msg.value()
            # determine if the message is a CCSDS bytestream by checking the packet length.
            if len(raw_bytes) == expected_packet_length:
                parsed = parse_packet(raw_bytes)
                if parsed:
                    # Only store the translated/parsed version in the aggregation buffer
                    # The original raw_bytes are not stored or sent to InfluxDB
                    aggregate_packet(parsed)
                    logger.info("Processed CCSDS packet and queued translated data for storage")
                else:
                    logger.warning("Failed to parse CCSDS packet, skipping")
            else:
                # bypass: forward the message as-is or ignore.
                logger.info("Received non-CCSDS stream - bypassing translation.")
                # Note: We don't write non-CCSDS data to InfluxDB

            # Flush aggregated records at each iteration.
            flush_aggregated_packets(influx_client)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
