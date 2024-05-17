'''
Script to make GWSkyNet-Multi classification predictions for candidate CBC events from LVK Public Alerts.
The script expects the candidate event ID to be provided as an argument.
The prediction results are saved in a text file.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from ligo.skymap import io, distance
from reproject import reproject_from_healpix
from skimage import transform
from subprocess import run
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
parser.add_argument('--models-path', 
                    help='Provide the path to the directory which contains the models. If not provided, it is assumed that the directory exists in the project top-level working directory.')
parser.add_argument('--data-path', 
                    help='Provide the path to the directory which contains the relevant data from training used in processing the sky map data and making the predictions. If not provided, it is assumed that the directory exists in the project top-level working directory.')
opts = parser.parse_args()


event_name = opts.superevent_id
drive_path = os.path.dirname(os.getcwd())
output_path = opts.output_path if opts.output_path else drive_path + '/predictions'
if not os.path.exists(output_path):
    os.mkdir(output_path)
models_path = '{}/'.format(opts.models_path) if opts.models_path else drive_path + '/models/'
training_data_path = '{}/'.format(opts.data_path) if opts.data_path else drive_path + '/data/'

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

target_header = fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                  360
NAXIS2  =                  180
CTYPE1  = 'RA---CAR'
CRPIX1  =                180.5
CRVAL1  =                180.0
CDELT1  =                   -1
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--CAR'
CRPIX2  =                 90.5
CRVAL2  =                  0.0
CDELT2  =                    1
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
""", sep='\n')

# Normalization factors (from the training) to normalize distances and Bayes factors
data_table = pd.read_csv(training_data_path+'data_summary_table.csv')
training_norms = {'mean_distance': np.max(data_table.DistMean), 'max_distance': np.max(data_table.DistMax),
                  'skymap': np.max(data_table.ProbMax), 'vol0': np.max(data_table.Vol0Max),
                  'vol1': np.max(data_table.Vol1Max), 'vol2': np.max(data_table.Vol2Max),
                  'logBCI': np.max(data_table.LogBCI), 'logBSN': np.max(data_table.LogBSN)}

def prepare_data(fits_file):
    (prob, mu, sigma, norm), metadata = io.read_sky_map(fits_file, distances=True, nest=None)
    
    # Read and normalize metadata
    meandist, std = metadata['distmean'], metadata['diststd']
    maxdist = meandist + 2.5 * std
    mean_distance = meandist / training_norms['mean_distance']
    max_distance = maxdist / training_norms['max_distance']
    logBCI = metadata['log_bci'] / training_norms['logBCI']
    logBSN = metadata['log_bsn'] / training_norms['logBSN']
    network = metadata['instruments']
    dets = []
    for ifo in ['H1', 'L1', 'V1']:
        dets.append(1) if ifo in network else dets.append(0)
    if np.sum(dets) < 2: print('WARNING: this is a single detector event. GWSkyNet-Multi is only trained to predict on multi-detector events. Results may be unreliable.')
    
    # Process sky map and volume images
    img_data = dict()
    img_cols = ['skymap', 'vol0', 'vol1', 'vol2']

    # reproject skymap data to rectangle
    with np.errstate(invalid='ignore'):
        img, mask = reproject_from_healpix((prob, 'ICRS'), target_header,
                                    nested=metadata['nest'], hdu_in=None,
                                    order='bilinear', field=0)
        img_data['skymap'] = img
        
    # calculate volume projections    
    rot = np.ascontiguousarray(distance.principal_axes(prob, mu, sigma, nest=metadata['nest']))
    prob_sums = np.zeros(3)
    
    imgwidth = 524 #dpi=300
    s = np.linspace(-maxdist, maxdist, imgwidth)
    Mpc2_per_pix = (s[1]-s[0])**2
    xx, yy = np.meshgrid(s, s)
    for iface, (axis0, axis1) in enumerate(((1,0), (0,2), (1,2))):
        density = distance.volume_render(xx.ravel(), yy.ravel(), maxdist,
                                            axis0, axis1, rot, metadata['nest'], prob, mu,
                                            sigma, norm).reshape(xx.shape)
        vol_prob = density*Mpc2_per_pix
        prob_sums[iface] = np.sum(vol_prob)
        vol_prob_ds = transform.downscale_local_mean(vol_prob, (4,4)) * 16.0
        img_data['vol{}'.format(iface)] = vol_prob_ds
        
    # re-calculate volume projections with higher resolution
    # if the probability sums are not within acceptable tolerance
    if (np.min(prob_sums) <= 0.98 or np.max(prob_sums) >= 1.005):
        imgwidth = 1048 #dpi=600
        s = np.linspace(-maxdist, maxdist, imgwidth)
        Mpc2_per_pix = (s[1]-s[0])**2
        xx, yy = np.meshgrid(s, s)
        for iface, (axis0, axis1) in enumerate(((1,0), (0,2), (1,2))):
            density = distance.volume_render(xx.ravel(), yy.ravel(), maxdist,
                                                axis0, axis1, rot, metadata['nest'], prob, mu,
                                                sigma, norm).reshape(xx.shape)
            vol_prob = density*Mpc2_per_pix
            prob_sums[iface] = np.sum(vol_prob)
            vol_prob_ds = transform.downscale_local_mean(vol_prob, (8,8)) * 64.0
            img_data['vol{}'.format(iface)] = vol_prob_ds
    
    # Normalize sky map and volume images
    norm_img, norms = dict(), dict()
    
    for column in img_cols:
        # Normalize img data
        norm = np.max(img_data[column])
        img = img_data[column] / norm

        # Downsize img data using maxpooling
        x = np.reshape(img, (1, len(img), len(img[0]), 1))
        # To avoid tensorflow warnings
        x = tf.cast(x, tf.float32)
        if column == 'skymap':
            maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        else:
            maxpool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))
        norm_img[column] = np.array(maxpool(x))
        # Normalize norms
        norms[column] = norm / training_norms[column]
    
    # Stack volume images
    # img_data has shape (1, 131, 131, 1), we need to reshape to (1, 131, 131) for stacking
    stacked_volume = np.stack([np.reshape(norm_img[column], (1, 131, 131)) for column in img_cols[1:]], axis=-1)
    stacked_volnorms = np.stack([norms[column] for column in img_cols[1:]], axis=-1)
    
    # Stack distances
    distances = np.stack((mean_distance, max_distance), axis=-1)
    
    return [stacked_volume, norm_img['skymap'], np.reshape(dets, (1,3)), np.reshape(distances, (1,2)),
            np.reshape(norms['skymap'], (1,1)), np.reshape(stacked_volnorms, (1,3)), np.reshape(logBSN, (1,1)), np.reshape(logBCI, (1,1))]


## FUNCTIONALITY TO MAKE THE GWSKYNET-MULTI PREDICTIONS

model_types = ['Glitch', 'NS', 'BBH']
n_models = 20
def make_predictions(data):
    
    # Load all the models and their thresholds from training
    model_names = []
    model_thresholds = []
    for model_type in model_types:
        model_names.append([model_type+'_{}'.format(i) for i in range(n_models)])
        model_thresholds.append(np.load(training_data_path+model_type+'_confidence_thresholds.npy')*100)
    model_names = np.ravel(np.array(model_names))
    model_thresholds = np.ravel(np.array(model_thresholds))

    loaded_models = []
    loaded_weights = []
    for i, name in enumerate(model_names):
        with open(models_path+name+'.json', 'r') as json_file:
            loaded_models.append(tf.keras.models.model_from_json(json_file.read()))
            loaded_weights.append(loaded_models[i].load_weights(models_path+name+'.h5'))
    
    # Make the predictions for each model, and find the mean and standard deviation
    prediction_scores = np.zeros(n_models*3)
    for i,model in enumerate(loaded_models):
        prediction_scores[i] = 100*tf.squeeze(model(data), [-1]).numpy()[0]
    
    prediction_score_means = np.zeros(3)
    prediction_score_errors = np.zeros(3)
    for i, model_type in enumerate(model_types):
        prediction_score_means[i] = np.mean(prediction_scores[i*n_models:i*n_models+n_models])
        prediction_score_errors[i] = np.std(prediction_scores[i*n_models:i*n_models+n_models])
    
    # Find the hierarchical prediction classifications
    # (hier_order given as list of 3 integers signifying order, with 0=Glitch, 1=NS, 2=BBH)
    hier_order = [0,1,2]
    hierarchical_prediction_classes = []
    for i in range(n_models):
        cur_model_scores = prediction_scores[[i,i+n_models,i+n_models*2]]
        cur_model_threholds = model_thresholds[[i,i+n_models,i+n_models*2]]
        
        if cur_model_scores[hier_order[0]] >= cur_model_threholds[hier_order[0]]:
            hier_pred=model_types[hier_order[0]]

        elif cur_model_scores[hier_order[1]] >= cur_model_threholds[hier_order[1]]:
            hier_pred=model_types[hier_order[1]]

        elif cur_model_scores[hier_order[2]] >= cur_model_threholds[hier_order[2]]:
            hier_pred=model_types[hier_order[2]]

        else: 
            m = max((cur_model_scores[0]-cur_model_threholds[0]),  (cur_model_scores[1]-cur_model_threholds[1]), (cur_model_scores[2]-cur_model_threholds[2]))
            if m == (cur_model_scores[0]-cur_model_threholds[0]):
                hier_pred=model_types[0]
            elif m == (cur_model_scores[1]-cur_model_threholds[1]):
                hier_pred=model_types[1]
            elif m == (cur_model_scores[2]-cur_model_threholds[2]):
                hier_pred=model_types[2]
        hierarchical_prediction_classes.append(hier_pred)
    hierarchical_prediction_classes = np.array(hierarchical_prediction_classes)
    hierarchical_class_counts = np.zeros(3)
    for i, model_type in enumerate(model_types):
        hierarchical_class_counts[i] = np.count_nonzero(hierarchical_prediction_classes == model_type)
    final_hierarchical_class = model_types[np.argmax(hierarchical_class_counts)]
    
    return prediction_score_means, prediction_score_errors, hierarchical_class_counts


## PROCESS THE FILE TO PREPARE DATA, AND MAKE THE PREDICTIONS
print('Processing the data...')
data = prepare_data(skymap_file)
print('Making the predictions...')
prediction_results = make_predictions(data)
model_scores = prediction_results[0]
model_scores_err = prediction_results[1]
final_hierarchical_class = model_types[np.argmax(prediction_results[2])]

## SAVE THE PREDICTION RESULTS
# Save the predictions for this event to a text file
results_save_file = '{}/{}.txt'.format(output_path, event_name)
with open(results_save_file, 'w') as f:
    for i, model_type in enumerate(model_types):
        f.write('{}-vs-all score: {:.0f} +/- {:.0f}\n'.format(model_type, model_scores[i], model_scores_err[i]))
    f.write('Hierarchical classification: {}'.format(final_hierarchical_class))
print('Done! The prediction results can be found in {}'.format(results_save_file))
