'''
Script that starts a GCN Kafka listener for LVK Public Alerts, and makes GWSkyNet-Multi predictions for when a candidate CBC event alert is issued. The user must provide their own GCN client id and secret to run the listener.
This script has been adapted from the IGWN Public Alerts User Guide sample code at https://emfollow.docs.ligo.org/userguide/tutorial/receiving/gcn.html
'''

import json
import os
import argparse
from gcn_kafka import Consumer
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument('--client-id', required=True, 
                    help='The alphanumeric GCN client ID of the user.')
parser.add_argument('--client-secret', required=True, 
                    help='The alphanumeric GCN client secret of the user.')
parser.add_argument('--output-path',
                    help='Provide the output path to the directory where the prediction results for all events will be saved. If not provided, it is assumed that the directory exists in the project top-level working directory. If no default directory is found then one is created.')
parser.add_argument('--test-mode', action='store_true', 
                    help='Run the script in test mode on hourly mock alerts sent by the LVK instead of on real events.')
opts = parser.parse_args()

drive_path = os.path.dirname(os.getcwd())
output_path = opts.output_path if opts.output_path else drive_path + '/predictions'
if not os.path.exists(output_path):
    os.mkdir(output_path)

def parse_notice(record):
    record = json.loads(record)
    event_id = record['superevent_id']
    print('Received an LVK public alert for candidate event {}:'.format(event_id))
    
    # For testing mock alerts.
    if (opts.test_mode and event_id[0] != 'M'):
        return
    
    # For parsing real events.
    if ((not opts.test_mode) and event_id[0] != 'S'):
        return

    if record['alert_type'] == 'RETRACTION':
        print('{} was retracted.'.format(event_id))
        return

    # Respond only to modeled search pipeline CBC events.
    if record['event']['group'] != 'CBC' or record['event']['pipeline'] == 'CWB':
        print('This alert is not triggered by a CBC modeled search pipeline, so no associated BAYESTAR sky map is available for prediction.')
        return

    # Call GWSkyNet-Multi to download the BAYESTAR FITS file, make predictions on this event, and save the results
    print('This is a candidate CBC event with BAYESTAR localization! Running GWSkyNet-Multi...')
    run(['python', 'GWSkyNet_Multi_predict.py', '--superevent-id', record['superevent_id'], '--output-path', output_path])


# Connect as a consumer
consumer = Consumer(client_id=opts.client_id, client_secret=opts.client_secret)

# Subscribe to topics and receive alerts
consumer.subscribe(['igwn.gwalert'])

while True:
    for message in consumer.consume(timeout=1):
        parse_notice(message.value())
