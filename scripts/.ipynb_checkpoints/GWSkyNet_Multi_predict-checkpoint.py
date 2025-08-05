'''
Script to make GWSkyNet-Multi classification predictions for candidate CBC events from LVK Public Alerts.
The script expects the candidate event ID to be provided as an argument.
The prediction results are saved in a json file, and optionally a text file.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from ligo.skymap import io
from ligo.skymap import postprocess
from subprocess import run
import json
import os
import sys
import argparse

# Specify the event name from the command line, and optionally the paths to the sky map file, models, data, and output.

parser = argparse.ArgumentParser()
parser.add_argument('--superevent-id', required=True, 
                    help='The name of the LVK candidate event as identified on GraceDB. This is usually a string that starts with S, followed by the year-month-day of the event in UTC, and then one or two alphabets.')
parser.add_argument('--skymap-file', 
                    help='The path to the BAYESTAR generated sky map FITS file for the event. If this is not provided, the latest sky map file will be downloaded from GraceDB and saved in the skymaps directory.')
parser.add_argument('--output-path', 
                    help='Provide the output path to the directory where the prediction results for this event will be saved. If not provided, it is assumed that the directory exists in the project top-level working directory. If no default directory is found then one is created.')
parser.add_argument('--save-text', action='store_true', 
                    help='Save the prediction results for this event in a text file for easy human readability.')
parser.add_argument('--models-path', 
                    help='Provide the path to the directory which contains the models. If not provided, it is assumed that the directory exists in the project top-level working directory.')
parser.add_argument('--data-path', 
                    help='Provide the path to the directory which contains the relevant input normalizations data from training. If not provided, it is assumed that the directory exists in the project top-level working directory.')
opts = parser.parse_args()


event_name = opts.superevent_id
drive_path = os.path.dirname(os.getcwd())
output_path = opts.output_path if opts.output_path else drive_path + '/predictions'
if not os.path.exists(output_path):
    os.mkdir(output_path)
models_path = '{}/'.format(opts.models_path) if opts.models_path else drive_path + '/models/'
data_path = '{}/'.format(opts.data_path) if opts.data_path else drive_path + '/data/'

if opts.skymap_file:
    skymap_file = opts.skymap_file
else:
    ## DOWNLOAD THE CANDIDATE EVENT BAYESTAR FITS FILE IF NOT PROVIDED
    event_url = 'https://gracedb.ligo.org/apiweb/superevents/{}/files/'.format(event_name)
    skymap_url = event_url + 'bayestar.multiorder.fits'
    skymap_download_path = drive_path + '/skymaps/'
    skymap_file = skymap_download_path+'{}_bayestar.multiorder.fits'.format(event_name)
    print('Downloading the sky map FITS file for candidate event '+event_name+'...')
    run(['curl', '-s', '-o', skymap_file, skymap_url])


## FUNCTIONALITY TO READ FITS FILE, PROCESS AND PREPARE DATA FOR GWSKYNET-MULTI INPUT

# Normalization factors (from the training) to normalize input values
input_norms_file = data_path+'input_normalization_factors.json'
with open(input_norms_file, 'r') as f:
    input_norms = json.load(f)

def prepare_data(fits_file):
    (prob, mu, sigma, norm), metadata = io.read_sky_map(fits_file, distances=True, nest=None)
    
    # Read and normalize metadata
    dist_mean = np.log10(metadata['distmean']) / input_norms['LogDistance']
    dist_std = np.log10(metadata['diststd']) / input_norms['LogDistance']
    logBCI = metadata['log_bci'] / input_norms['LogBCI']
    if metadata['log_bsn'] < 100:
        logBSN = metadata['log_bsn'] / input_norms['LogBSN']
    else:
        logBSN = 100 / input_norms['LogBSN']
    
    network = metadata['instruments']
    dets = []
    for ifo in ['H1', 'L1', 'V1']:
        dets.append(1) if ifo in network else dets.append(0)
    if np.sum(dets) < 2:
        print('WARNING: this is a single detector event. GWSkyNet-Multi is only trained to predict on multi-detector events. Results may be unreliable.')

    # Calculate the localization area and volume from the maps, and normalize
    skymap = io.read_sky_map(fits_file, distances=True, nest=None, moc=True)
    credible_regions = postprocess.crossmatch(skymap, contours=(0.9,))
    sky_area = np.log10(credible_regions.contour_areas[0]) / input_norms['LogSkyArea90']
    volume = np.log10(credible_regions.contour_vols[0]) / input_norms['LogVolume90']
    
    # Combine all data for model input, in correct order and array shape
    data = [np.reshape(volume, (1,1)), np.reshape(sky_area, (1,1)), np.reshape(dets, (1,3)), np.reshape(dist_mean, (1,1)), np.reshape(dist_std, (1,1)), np.reshape(logBSN, (1,1)), np.reshape(logBCI, (1,1))]
    #data = [volume, sky_area, np.array(dets), dist_mean, dist_std, logBSN, logBCI]
    return data


## FUNCTIONALITY TO MAKE THE GWSKYNET-MULTI PREDICTIONS

n_models = 20
output_classes = ['Glitch', 'BBH', 'NSBH', 'BNS']
def make_predictions(data):
    
    # Load all the models from training
    models = []
    #weights = []
    model_names = np.array(['Multi-class_{}'.format(i+1) for i in range(n_models)])
    for i,name in enumerate(model_names):
        with open(models_path+'Multi-class.json', 'r') as json_file:
            models.append(tf.keras.models.model_from_json(json_file.read()))
            models[i].load_weights(models_path+name+'.h5')
    
    # Make the predictions for each model, and find the mean and standard deviation
    prediction_probs = np.zeros((n_models,4))
    for i,model in enumerate(models):
        prediction_probs[i] = 100 * model(data).numpy()
    
    prediction_prob_means = np.mean(prediction_probs,axis=0)
    prediction_prob_stds = np.std(prediction_probs,axis=0)
    
    return prediction_prob_means, prediction_prob_stds


## PROCESS THE FILE TO PREPARE DATA, AND MAKE THE PREDICTIONS
print('Processing the data...')
data = prepare_data(skymap_file)
print('Making the predictions...')
prediction_probs, prediction_probs_err = make_predictions(data)
# Provide the classification label determined by the class with the highest probability
prediction_class = output_classes[np.argmax(prediction_probs)]

## SAVE THE PREDICTION RESULTS
# Save the predictions for this event to a json file
prediction_results = {}
for i, class_type in enumerate(output_classes):
    prediction_results[class_type] = int(np.round(prediction_probs[i]))
    prediction_results[class_type+'_uncertainty'] = int(np.round(prediction_probs_err[i]))
prediction_results['Classification'] = prediction_class
results_save_file = '{}/{}.json'.format(output_path, event_name)
with open(results_save_file, 'w') as f:
    json.dump(prediction_results, f, sort_keys=False, indent=4)
print('Done! The prediction results can be found in {}'.format(results_save_file))

# Optionally save the results to a text file
if opts.save_text:
    results_save_file = '{}/{}.txt'.format(output_path, event_name)
    with open(results_save_file, 'w') as f:
        f.write('Predicted probabilities (%):\n')
        for i, class_type in enumerate(output_classes):
            f.write('{}: {:.0f} +/- {:.0f}\n'.format(class_type, prediction_probs[i], prediction_probs_err[i]))
        f.write('Classification: {}'.format(prediction_class))
