# GWSkyNet-Multi
GWSkyNet-Multi is a multi-class CNN classifier for gravitational-wave candidate events published by LIGO-Virgo-KAGRA. This repository contains the published model together with instructions on how to use the classifier. The input is a sky map FITS file from BAYESTAR, which can be found on GraceDB for real gravitational-wave candidates.

The files and scripts in this directory allow the user to make GWSkyNet-Multi classification predictions for candidate LVK CBC events. The GWSkyNet-Multi machine learning model and its inner workings are described in Abbott et al. (2022) (https://iopscience.iop.org/article/10.3847/1538-4357/ac5019) and Raza et al. (2024) (https://iopscience.iop.org/article/10.3847/1538-4357/ad13ea)

The directory structure should not be modified.
The "models" folder contains the trained machine learning models for each of the Glitch-vs-all, NS-vs-all, and BBH-vs-all classifiers. There are trained 20 versions of each model.
The "training data" folder contains the confidence thresholds that were determined for each model for which the False Negative Rate = False Positive Rate. The folder also contains the metadata from training that is necessary for normalizations.
The "FITS_files" folder is where the scripts download and save the BAYESTAR sky map FITS files for each candidate event.
The "prediction_results" folder is where the results of the GWSkyNet-Multi predictions are saved. Individual text files for each event, and a combined csv table for all events.

The user should first set up the environment with the required packages using either the provided yml file (conda) or txt file (pip). The following commands should accomplish this:
$ conda env create --file gwskynet_multi_predictions.yml
$ python3 -m pip install -r requirements.txt

If the user has the candidate event name, then the script GWSkyNet_Multi_predict.py can be executed with the event name provided to download the FITS file from graceDB, make the predictions, and save the results. For example, for predicting on the first significant event in O4a (“S230524x”), the user would execute:
$ python3 GWSkyNet_Multi_predict.py S230524x
This will print the following lines to your terminal output, as the script completes each step:
"Downloading the FITS file for candidate event S230524x...
Processing the data...
Making the predictions...
Saving the results...
Done!"
And will save a file called S230524x.txt in the “prediction_results” folder, the contents of which are:
"Glitch-vs-all score: 87 +/- 14
NS-vs-all score: 71 +/- 17
BBH-vs-all score: 0 +/- 0
Hierarchical classification: Glitch"
In the prediction results that are saved, the output for each classifier is the mean score of the 20 trained models used, with the standard deviation of the 20 model scores quoted as the uncertainty. The given classification is based on applying score thresholds from training to the model results in a hierarchical order, and selecting the classification that occurs the most often (see Abbott et al. 2022). The script will also add these results to the “events_predictions_results.csv” file, which is a combined table for all events that the user has predicted on, in the “prediction_results” folder (the first time you run this script and make a prediction it will create the csv file, and subsequent times simply add to it).


If the user wishes to launch a listener that uses a GCN stream for LVK public alerts, and then make predictions once a candidate CBC event alert is issued, then the script LVK_alert_stream_and_predict.py can be executed. Note that the first two lines of the script must be modified to provide the user's GCN client ID and client secret for the stream. If these have not already been set up, see the "Account Creation and Credential Generation" section at https://emfollow.docs.ligo.org/userguide/tutorial/receiving/gcn.html. The third line of the script can also be modified to run in test mode, where it listens for hourly LVK mock alerts instead of real alerts.
