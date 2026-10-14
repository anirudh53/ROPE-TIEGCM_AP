Users are referred to "Probabilistic Solar Proxy Forecasting with Neural Network Ensembles" by Joshua D. Daniell and Piyush M. Mehta submitted to AGU Space Weather (2023). 
At the time of the establishment of this repository, the article is not yet published. However, an original and unrevised preprint is available in the meantime (https://arxiv.org/abs/2306.02169v1).

Author Contact Information
jddaniell@mix.wvu.edu

This repository contains Python scripts, directories, and data.
#########################################################################################################
Directories & Objects
1. Data/   
	   -> noaa_radio_flux_fullset.csv contains F10.7 observed values for 01/01/1948-12/10/2021
           -> Scaling_mean.npy NumPy object that stores the mean value used to originally scale training data (needs to be used to unscale predictions)
           -> Scaling_mean.npy NumPy object that stores the standard deviation value used to originally scale training data (needs to be used to unscale predictions)

2. Dynamic Lookback Ensemble/
	   -> Models/ is used to store tuned/trained dynamic lookback models
           -> Predictions/ is used to store any predictions made on the evaluation set in the form of .csv files
           -> True/ is used to store any true values of the desired set in the form of .csv files

3. Investigations/   
	   -> Back Average/ are files for a given year (or Sim1) with the PCA coefficients
           -> Lookback/ contains PCA coefficients and inputs for 2002-2008 with normalization parameters

4. Lookback Ensemble/   
	   -> Models/ is used to store tuned/trained multi-step lookback models
           -> Predictions/ is used to store any predictions made on the evaluation set in the form of .csv files
           -> True/ is used to store any true values of the evaluation set in the form of .csv files

5. NBEATS/   
	   -> Models/ is used to store trained NBEATS models when desired
           -> Predictions/ is used to store any predictions made on the desired evaluation set in the form of .csv files
           -> True/ is used to store any true values of the desired evaluation set in the form of .csv files

6. Persistence/   
           -> Predictions/ is used to store any predictions made on the desired evaluation set in the form of .csv files
           -> True/ is used to store any true values of the desired evaluation set in the form of .csv files

7. Plots/
	   -> EMPTY but used to store any desired plots for further analysis

8. Post Processing/   
	   -> Metrics/ used to store calculated metrics in the form of .csv files


9. SET/   
	   -> Predictions/ is used to store any predictions made on the desired evaluation set in the form of .csv files
           -> True/ is used to store any true values of the desired evaluation set in the form of .csv files

10. SWPC Compared/   
	   -> Contains forecast data from original work, to be used in direct comparison with SWPC. (Make updates to these .npy files if you seek to compare further results)
           -> Weekly/ is used to store NOAA SWPC forecast data that has been converted from PDF to csv
11. Trained Models/
           -> Dynamic Lookback Ensemble/ stores pre trained dynamic models from original work saved in .h5 format using Keras (for use in example script)
           -> Lookback Ensemble/stores pre trained multi-step models from original work saved in .h5 format using Keras

12. Uncertainty Work/
	   -> Used as a folder to store/organize uncertainty evaluation scripts

13. Example/
	   -> Directory used to store forecasts made by the 'MLE_Prediction_Example.py' script
	   -> Additionally, directory stores folders that store predictions separated by model type and file type. 
	      For example; Example/PredictionCSV/LSTM/ is the directory that will store all csvs created by the script if desired.
#########################################################################################################
Scripts:

MLE_Prediction_Example.py | Script used with pre-trained models to make forecasts on given dates. User specifies dates and desired model types for forecasting. 
			    This script should be primarily used for future forecast comparisons and is not easily post processed by other scripts in this repository.
			    Users will also need to specify how they desire forecasts to be saved (csv or npy)

1. --

2.1. LSTM_Tuner_Trainer.py | used for creating necessary dynamic forecasting LSTM models; additionally makes predictions on the given date range.
2.2. MLP_Tuner_Trainer.py  | used for creating necessary dynamic forecasting MLP models; additionally makes predictions on the given date range.

3.1 Investigations\Back Average\Backwards_Average_Investigation.py | Script used to investigate the performance of inclusion of backwards averages into input data; runs Keras Tuner and saves results as TunedBackAveResults.mat 
3.2 Investigations\Back Average\Backwards_Average_Investigation.py | Script used to post process data and save associated figures and data from the investigation.
(Predictions,Targets, and Tuner directories are populated when the Investigation is performed)

3.3 Investigations\Lookback\Lookback Investigation.py | Script used to investigate the performance of varied lookback; runs Keras Tuner and saves results as TunedLookbackResultsMLP.mat 
3.4 Investigations\Lookback\Lookback Post.py | Script used to post process data and save associated figures and data from the investigation.
(Predictions,Targets, and Tuner directories are populated when the Investigation is performed)

4.1. LSTM_Tuner_Trainer.py | used for creating necessary multi-step forecasting LSTM models; additionally makes predictions on the given date range.
4.2. MLP_Tuner_Trainer.py  | used for creating necessary multi-step forecasting MLP models; additionally makes predictions on the given date range.
(Models,Predictions, and True directories populate once scripts 4.1 and 4.2 are executed.)

5.1. NBEATS_Ensemble.py | Script used for creating, training, and creating predictions with various NBEATS models. 
(similar to the work done by Stenveson et al., A Deep Learning Approach to Solar Radio Flux Forecasting 2020. https://github.com/stardust-r/deep-learning-space-weather-forecasting)
(Models,Predictions, and True directories populate once 5.1 is executed.)

6.1 Persist.py | Script used for creating forecasts by way of the persistence model on desired dates to be specified in the script itself.
(Predictions & True directories populate once persist.py is executed.)

7. --

8.Post Process Global.py | Script used to post process all predictions made my all model types. Script will generate plots and output metrics to the /Metrics directory.

9.1 SET.py | Script used to generate forecasts and true values using the SET (TS_FCAST) method for the desired dates. Error statistics after binning are used in other scripts and are located in this directory.
9.2 TS_FCAST.py | Script function used by 9.1 to generate forecasts.

10.1 Weekly_MLE_Comparison.py | A script that loads NOAA SWPC weekly forecasts for use in comparison with forecasts made by multi-step and dynamic lookback models. 
The script saves a figure which plots comparison metrics. Further metrics can be located in the variable explorer in an IDE.

11. --

12.1 Uncertainty Calibration Reduced.py | A script that loads ensemble forecasting methods NBEATS and the best MLE and prepares the CES score and any figures associated with calibration curves and sigma scaling.
12.2 Uncertainty Calibration.py   	| A script that loads all ensemble forecasting methods and prepares the CES score and any figures associated with calibration curves and sigma scaling.


#########################################################################################################
