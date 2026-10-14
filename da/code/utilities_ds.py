import scipy as sp
import os 
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from itertools import combinations, product, combinations_with_replacement
import pickle
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


class SVD_manager:

    def __init__(self, train, n_components, norm_dict):
        print("PCA_Manager initialized")
        self._train = train.T
        self.n_components = n_components
        self.norm_dict = norm_dict

    def calculate_reconstruction_error(self, coeff, x_original_norm, method = 'mean'):
        if method == 'mean':
            ds_reconstructed = self.U @ coeff + self.norm_dict['x_mean']
            ds_reconstructed = 10**ds_reconstructed
            x_original_denormalized = x_original_norm  + self.norm_dict['x_mean']
            x_original_denormalized = 10**x_original_denormalized
            ae = ds_reconstructed - x_original_denormalized
            pae = np.abs(ae / x_original_denormalized)
            print(f'Reconstruction error on dataset is {100. * np.mean(pae):.2f}%')
        elif method == 'minmax':
            ds_reconstructed = self.U @ coeff * (self.norm_dict['x_max'] - self.norm_dict['x_min']) + self.norm_dict['x_min']
            ds_reconstructed = 10**ds_reconstructed
            x_original_denormalized = x_original_norm * (self.norm_dict['x_max'] - self.norm_dict['x_min']) + self.norm_dict['x_min']
            x_original_denormalized = 10**x_original_denormalized
            ae = ds_reconstructed - x_original_denormalized
            pae = np.abs(ae / x_original_denormalized)
            print(f'Reconstruction error on dataset is {100. * np.mean(pae):.2f}%')       
        elif method == 'std':
            ds_reconstructed = (self.U @ coeff) * self.norm_dict['x_std'] + self.norm_dict['x_mean']
            ds_reconstructed = 10**ds_reconstructed
            x_original_denormalized = x_original_norm * self.norm_dict['x_std'] + self.norm_dict['x_mean']
            x_original_denormalized = 10**x_original_denormalized
            ae = ds_reconstructed - x_original_denormalized
            pae = np.abs(ae / x_original_denormalized)
            print(f'Reconstruction error on dataset is {100. * np.mean(pae):.2f}%')       

    def fit_PCA(self) -> None:
        """fits PCA to the training data specified in __init__"""
        print("Fitting PCA")
        save_data = {}
        print(f"self._train.shape: {self._train.shape}")
        print(f"min: {np.min(self._train)}, max: {np.max(self._train)}")

        den = np.copy(self._train)
        save_data["mean_density"] = np.mean(den, axis = 0, keepdims = True)

        den -= save_data["mean_density"]
        pca = PCA(n_components=self.n_components)
        x = pca.fit(den)
        self.pca = pca

        save_data["UtX"] = x.fit_transform(den).T
        self.coeff = save_data["UtX"]
        save_data["U"] = x.components_.transpose()
        self.U = save_data["U"]
        save_data["S"] = x.singular_values_
        self.S = save_data["S"]
        self.sigma = (np.eye(self.S.shape[0])*self.S)
        self.sigma_inv = (np.eye(self.S.shape[0])*1/self.S)
        print("PCA fitting complete")
        self.data = save_data 

    def transform(self, x, norm_dict, method = 'mean'):
        if method == 'mean':
            z = self.U.T @ (x - norm_dict['x_mean'])
        return z

def shift_row(x):
    shifted = np.roll(x, -1)
    return shifted

def create_hankel(x, q, p):
    shifted_lst = []
    shifted_lst.append(x[0:(x.shape[0] - q)])
    shifted_0 = shift_row(x)
    shifted_lst.append(shifted_0[0:(shifted_0.shape[0] - q)])
    for k in range(q-2):
        shifted = shift_row(shifted_0)
        shifted_lst.append(shifted[0:(shifted_0.shape[0] - q)])

    hankel_x = np.concatenate(shifted_lst, axis=1)[0:p]
    return hankel_x

def calculate_reconstruction_error(USigma, Vt, original_data):
    ae = USigma @ Vt - original_data
    idx = original_data!=0.
    ae_perc = np.abs(ae[idx] / original_data[idx])
    print(f'Removed {100.*(np.prod(ae.shape) - np.prod(ae_perc.shape))/np.prod(ae.shape):.2f}% zero elements')
    print(f'Reconstruction error is {np.mean(ae_perc)*100.:.6f}%')

def predict_v0(Vt, X, A):
    Z1 = X@Vt.T
    Z2 = Z1@A
    return Z2

def get_svd_params(X, n_components = 10):
    #x_norm, norm_dict = normalize_array(np.copy(X))
    x_norm = (X - X.mean(axis = 1))/X.std(axis = 1)
    svd = TruncatedSVD(n_components = n_components, n_iter = 1)
    svd.fit(X)
    USigma = svd.transform(X)
    Vt = svd.components_
    sigma_values = svd.singular_values_
    sigma = (np.eye(sigma_values.shape[0])*sigma_values)
    sigma_inv = (np.eye(sigma_values.shape[0])*1/sigma_values)
    U = USigma @ sigma_inv
    svd_params_dict = {'USigma':USigma, 'U': U, 'Vt':Vt, 'sigma_values': sigma_values, 'sigma': sigma, 
        'sigma_inv': sigma_inv}#, 'norm_dict': norm_dict}
    return svd_params_dict


def create_library_functions(x, functions_dictionary, feature_names):
    function_names = functions_dictionary.keys()

    

    feature_names_lst = []
    for tuple in list(product(functions_dictionary, feature_names)):
        feature_names_lst.append(tuple[0] + '(' + tuple[1] + ')')

    feature_names_interactions_lst = []
    for tuple in list(combinations_with_replacement(feature_names_lst, 2)):
        feature_names_interactions_lst.append(tuple[0] + tuple[1] )

    all_feature_names = ['1'] + feature_names_lst + feature_names_interactions_lst 

    arr = np.concatenate([np.apply_along_axis(functions_dictionary[func_name], 0, x) for func_name in function_names], axis = 1)
    combs = list(combinations_with_replacement(range(arr.shape[1]), 2))
    
    left_combinations_index = [p[0] for p in combs]
    right_combinations_index = [p[1] for p in combs]
    
    theta = np.concatenate([np.ones(arr.shape[0]).reshape((-1, 1)), arr, arr[:, left_combinations_index] * arr[:, right_combinations_index]], axis = 1)
    library_dict = {'theta': theta, 'library_feature_names': all_feature_names}
    return library_dict


def reconstruct_state(x_norm, svd_dict, reg_norm_dict, pca_norm_dict, method = 'mean', drivers = True):
    if drivers is True:
        variables = -2
    else:
        variables = x_norm.shape[0]
    x_temp = np.copy(x_norm)
    if method == 'mean':
        x = x_temp + reg_norm_dict['x_mean'][0:variables, :]
    elif method == 'std':
        x = x_temp * reg_norm_dict['x_std'][0:variables, :] + reg_norm_dict['x_mean'][0:variables, :]
    elif method == 'minmax':
        x = x_temp * (reg_norm_dict['x_max'][0:variables, :] - reg_norm_dict['x_min'][0:variables, :]) + reg_norm_dict['x_min'][0:variables, :]
    elif method == 'none':
        x = x_temp
    x_out = svd_dict.U @ np.copy(x) + pca_norm_dict['x_mean']
    return 10**x_out

def normalize_with_dict(x, norm_dict, method):
    x_temp = np.copy(x)
    if method == 'mean':
        x_norm = x_temp - norm_dict['x_mean']
    elif method == 'std':
        x_norm = (x_temp - norm_dict['x_mean']) / norm_dict['x_std']
    elif method == 'minmax':
        x_norm = (x_temp - norm_dict['x_min']) / (norm_dict['x_max'] - norm_dict['x_min'])
    elif method == 'none':
        x_norm = x_temp
    return x_norm

def normalize_array(array, method = 'std'):

    x = np.copy(array)

    if method == 'mean':
        x_mean = np.mean(x, axis = 1, keepdims = True)
        x_norm = (x - x_mean)
        norm_dict = {'x_mean': x_mean}  

    elif method == 'std':
        x_mean = np.mean(x, axis = 1, keepdims = True)
        x_std = np.std(x, axis = 1, keepdims = True)
        x_norm = (x - x_mean) / x_std
        norm_dict = {'x_mean': x_mean, 'x_std': x_std}

    elif method == 'minmax':
        max_values = np.max(x, axis = 1, keepdims = True)
        min_values = np.min(x, axis = 1, keepdims = True)
        x_norm = (x - min_values) / (max_values - min_values)
        norm_dict = {'x_min': min_values, 'x_max': max_values}

    elif method == 'none':
        x_norm = x
        norm_dict = {}

    return x_norm, norm_dict



def stls_multi_output(X, Y, threshold, max_iter=100, tol=1e-5):
    """
    Sequential Thresholded Least Squares (STLS) for sparse multi-output regression.
    
    Parameters:
    X : numpy.ndarray
        Design matrix (input features) of shape (n_samples, n_features).
    Y : numpy.ndarray
        Target matrix (outputs) of shape (n_samples, n_outputs).
    threshold : float
        Threshold for setting small coefficients to zero.
    max_iter : int
        Maximum number of iterations for convergence.
    tol : float
        Tolerance for convergence. If change in coefficients between iterations
        is smaller than this value, stop the algorithm.
    
    Returns:
    W : numpy.ndarray
        Final weight (coefficient) matrix of shape (n_features, n_outputs).
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]

    # Initialize weights using least-squares solutions for each output
    W = np.linalg.lstsq(X, Y, rcond=None)[0]

    for iteration in range(max_iter):
        W_old = W.copy()

        # Apply the threshold: set coefficients to zero if they are smaller than the threshold
        W[np.abs(W) < threshold] = 0

        # Find the indices of non-zero coefficients
        non_zero_idx = np.where(np.abs(W).sum(axis=1) >= threshold)[0]
        if len(non_zero_idx) == 0:
            break  # If all coefficients are zero, exit

        # Recompute least squares on the reduced feature set
        X_reduced = X[:, non_zero_idx]

        # Recalculate the least squares solution for all outputs on the reduced set
        W_reduced = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]

        # Update the full weight matrix
        W[non_zero_idx, :] = W_reduced
        W[np.abs(W) < threshold] = 0
        # Check for convergence
        if np.linalg.norm(W - W_old) < tol:
            break
    
    return W



def reconstruct_state(x_norm, svd_obj, reg_norm_dict, pca_norm_dict, method = 'mean', drivers = True):
    if drivers is True:
        variables = -2
    else:
        variables = x_norm.shape[0]
    x_temp = np.copy(x_norm)
    if method == 'mean':
        x = x_temp + reg_norm_dict['x_mean'][0:variables, :]
    elif method == 'std':
        x = x_temp * reg_norm_dict['x_std'][0:variables, :] + reg_norm_dict['x_mean'][0:variables, :]
    elif method == 'minmax':
        x = x_temp * (reg_norm_dict['x_max'][0:variables, :] - reg_norm_dict['x_min'][0:variables, :]) + reg_norm_dict['x_min'][0:variables, :]
    elif method == 'none':
        x = x_temp
    x_out = svd_obj.U @ np.copy(x) + pca_norm_dict['x_mean']
    return np.float64(10**x_out)

def save_stats(data, file_name, path):
    # Directory where you want to save the file
    directory = f'data/{path}'
    file_name = f'{file_name}.h5'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full file path
    file_path = os.path.join(directory, file_name)

    # Save the data as .h5 file
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dataset_name', data=data)

    print(f"File saved at {file_path}")

def denormalize(x, norm_dict, method):
    x_temp = np.copy(x)
    if method == 'mean':
        x_orig = x_temp + norm_dict['x_mean']
    elif method == 'std':
        x_orig = x_temp * norm_dict['x_std'] + norm_dict['x_mean']
    elif method == 'minmax':
        x_orig = x_temp * (norm_dict['x_max'] - norm_dict['x_min']) + norm_dict['x_min']
    elif method == 'none':
        x_orig = x_temp
    return x_orig

def make_regression(X, Y, a_ridge = 0.0):
    B = Y @ X.T @ np.linalg.inv(X @ X.T + a_ridge * np.diag(np.ones(X.shape[0])))
    return B

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def predict(x1_test_centered, u_test, B, x_train_svd_obj, X_reg_norm_dict, normalization_method, model_params, input_features):
    z1_test = x_train_svd_obj.U.T @ x1_test_centered
    X_test = np.copy(np.concatenate([z1_test, u_test[3:, :]], axis = 0))    

    if model_params['model'] == 'sindy':  
        print(X_test.max(), X_test.min())
        library_dict = create_library_functions(np.copy(X_test.T), model_params['functions'], input_features)
        X_test = library_dict['theta'].T
        X_test_norm = normalize_with_dict(X_test[1:, :], X_reg_norm_dict, method = normalization_method)  
        X_test_norm = np.concatenate([X_test[0, :].reshape((1, -1)), X_test_norm])
        print(X_test_norm.max(), X_test_norm.min())
    elif model_params['model'] == 'dmd':
        X_test_norm = normalize_with_dict(X_test, X_reg_norm_dict, method = normalization_method)

    y_pred_test = B @ X_test_norm        

    return y_pred_test

def format_predictions(z2_train, z2_test, x2_train, x2_test, y_pred_train, y_pred_test, Y_reg_norm_dict, x_train_svd_obj, x_train_dict, normalization_method, mdl, reconstruction):
    if mdl == 'dmd':
        e_test = z2_test - y_pred_test
    elif mdl == 'sindy':
        e_test = z2_test - y_pred_test
        
    if reconstruction == True:
        y_actual_train = np.float64(10**x2_train)
        y_actual_test = np.float64(10**x2_test)
        y_pred_train = reconstruct_state(np.copy(y_pred_train), x_train_svd_obj, Y_reg_norm_dict, x_train_dict, method = normalization_method, drivers = False)
        y_pred_test = reconstruct_state(np.copy(y_pred_test), x_train_svd_obj, Y_reg_norm_dict, x_train_dict, method = normalization_method, drivers = False)
    else:
        y_actual_train = z2_train
        y_actual_test = z2_test
        y_pred_train = denormalize(y_pred_train, Y_reg_norm_dict, normalization_method)
        y_pred_test = denormalize(y_pred_test, Y_reg_norm_dict, normalization_method)

    return y_actual_train, y_actual_test, y_pred_train, y_pred_test, e_test

def calculate_errors(y_pred_train, y_actual_train, y_pred_test, y_actual_test, u_train, u_test, mdl_stats_dict, mdl_stats_lst, mdl, models, kp_idx = 4):
    aes_train = np.abs((y_pred_train - y_actual_train)/y_actual_train)
    mae_train_plain = 100.*np.mean(aes_train)
    aes_test = np.abs((y_pred_test - y_actual_test)/y_actual_test)
    mae_test_plain = 100.*np.mean(aes_test)
    mdl_stats_dict.update({f'{mdl}': {'mae_train': mae_train_plain, 'mae_test': mae_test_plain}})
    mdl_stats_lst.append(mdl_stats_dict)

    print(f'MAPE for {mdl}-c on train is {mae_train_plain:.2f}%')
    print('MAPE per year on train set')
    df_train = pd.DataFrame(np.concatenate([u_train[0, :].T.reshape((-1, 1)), u_train[kp_idx, :].T.reshape((-1, 1)), 100*np.mean(np.abs((y_pred_train - y_actual_train)/y_actual_train).T, axis = 1, keepdims=True)], axis = 1 ), \
        columns = ['year', 'kp', 'mape'])
    print(df_train.groupby(['year']).mean())

    print(f'MAPE for {mdl}-c on test is {mae_test_plain:.2f}%')

    print('MAPE per year on test set')
    df_test = pd.DataFrame(np.concatenate([u_test[0, :].T.reshape((-1, 1)), u_test[kp_idx, :].T.reshape((-1, 1)), 100*np.mean(np.abs((y_pred_test - y_actual_test)/y_actual_test).T, axis = 1, keepdims=True)], axis = 1 ), \
        columns = ['year', 'kp', 'mape'])
    print(df_test.groupby(['year']).mean())   

    return df_train, df_test, mdl_stats_dict, mdl_stats_lst


def predict_sindy_reduced(x1_test_centered, u_test, B, x_train_svd_obj, X_reg_norm_dict, normalization_method, model_params, input_features, pca_coupling = 2, f10_idx = 5):
    kp_idx = f10_idx + 1
    z1_test = x_train_svd_obj.U.T @ x1_test_centered

    if model_params['model'] == 'sindy':  
        X_for_sindy_test = np.copy(np.concatenate([z1_test[pca_coupling, :].reshape((1, -1)), u_test[f10_idx:, :]], axis = 0)) 
        print(X_for_sindy_test.max(), X_for_sindy_test.min())
        
        library_dict = create_library_functions(np.copy(X_for_sindy_test.T), model_params['functions'], input_features)
        theta_test, theta_features = library_dict['theta'].T, library_dict['library_feature_names']
        X_test = np.copy(np.concatenate([theta_test, np.delete(np.copy(z1_test), pca_coupling, axis = 0)], axis = 0)) 
        
        X_test_norm = normalize_with_dict(X_test[1:, :], X_reg_norm_dict, normalization_method)
        X_test_norm = np.concatenate([X_test[0, :].reshape((1, -1)), u_test[:f10_idx, :], X_test_norm])
        
        print(X_test_norm.max(), X_test_norm.min())
    elif model_params['model'] == 'dmd':
        X_test = np.copy(np.concatenate([z1_test, u_test[f10_idx:, :]], axis = 0))  
        X_test_norm = normalize_with_dict(X_test, X_reg_norm_dict, method = normalization_method)
        X_test_norm = np.concatenate([u_test[:f10_idx, :], X_test_norm])

    y_pred_test = B @ X_test_norm        

    return y_pred_test

def calculate_y_error_corr(y_pred, error):
    s_y = np.sqrt(np.sum((y_pred - np.mean(y_pred, axis = 1, keepdims = True))**2))
    s_e = np.sqrt(np.sum((error - np.mean(error, axis = 1, keepdims = True))**2))

    corr = np.sum((y_pred - np.mean(y_pred, axis = 1, keepdims = True)) * (error - np.mean(error, axis = 1, keepdims = True)), axis = 1 ) / (s_y * s_e) 
    
    return corr

def select_max_corr_coeff_idx(z_train, kp_train, n_components = 10):
    coeff_idx_lst = []
    for k in range(n_components):
        corr = np.abs(calculate_y_error_corr(z_train[k, :].reshape((1, -1)), kp_train))
        coeff_idx_lst.append(corr)
    corrs = np.concatenate(coeff_idx_lst)

    return corrs, np.where(corrs == np.max(corrs))[0]

def train_nlk_coefficients(X_train_original, Y_train_original, nl_idx, alpha_ridge = 0.5):
    X_nlk = np.copy(X_train_original[:, nl_idx])
    Y_nlk = np.copy(Y_train_original[:, nl_idx])
    B_nlk = Y_nlk @ X_nlk.T @ np.linalg.inv(X_nlk @ X_nlk.T + alpha_ridge * np.diag(np.ones(X_nlk.shape[0])))
    return B_nlk

def create_nl_indexes(u_train, kp_idx, window_size = 20):
    f10_idx = kp_idx - 1
    #
    cumsum = np.cumsum(np.copy(u_train[f10_idx, :]))
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    cumsum = cumsum / window_size
    delta = np.concatenate([np.zeros((1)), cumsum[1:] - cumsum[:-1]])


    nl80_idx = np.where((u_train[kp_idx, :] >= 8.) & (u_train[kp_idx, :] <= 9.) & (delta >= 0.))[0]
    nl70_idx = np.where((u_train[kp_idx, :] >= 7.) & (u_train[kp_idx, :] < 8.) & (delta >= 0.))[0]
    nl60_idx = np.where((u_train[kp_idx, :] >= 6.) & (u_train[kp_idx, :] < 7.) & (delta >= 0.))[0]
    nl50_idx = np.where((u_train[kp_idx, :] >= 5.) & (u_train[kp_idx, :] < 6.) & (delta >= 0.))[0]
    nl40_idx = np.where((u_train[kp_idx, :] >= 4.) & (u_train[kp_idx, :] < 5.) & (delta >= 0.))[0]
    nl30_idx = np.where((u_train[kp_idx, :] >= 3.) & (u_train[kp_idx, :] < 4.) & (delta >= 0.))[0]
    nl20_idx = np.where((u_train[kp_idx, :] >= 2.) & (u_train[kp_idx, :] < 3.) & (delta >= 0.))[0]
    nl10_idx = np.where((u_train[kp_idx, :] >= 1.) & (u_train[kp_idx, :] < 2.) & (delta >= 0.))[0]
    nl00_idx = np.where((u_train[kp_idx, :] < 1.) & (delta >= 0.))[0]

    nl81_idx = np.where((u_train[kp_idx, :] >= 8.) & (u_train[kp_idx, :] <= 9.) & (delta < 0.))[0]
    nl71_idx = np.where((u_train[kp_idx, :] >= 7.) & (u_train[kp_idx, :] < 8.) & (delta < 0.))[0]
    nl61_idx = np.where((u_train[kp_idx, :] >= 6.) & (u_train[kp_idx, :] < 7.) & (delta < 0.))[0]
    nl51_idx = np.where((u_train[kp_idx, :] >= 5.) & (u_train[kp_idx, :] < 6.) & (delta < 0.))[0]
    nl41_idx = np.where((u_train[kp_idx, :] >= 4.) & (u_train[kp_idx, :] < 5.) & (delta < 0.))[0]
    nl31_idx = np.where((u_train[kp_idx, :] >= 3.) & (u_train[kp_idx, :] < 4.) & (delta < 0.))[0]
    nl21_idx = np.where((u_train[kp_idx, :] >= 2.) & (u_train[kp_idx, :] < 3.) & (delta < 0.))[0]
    nl11_idx = np.where((u_train[kp_idx, :] >= 1.) & (u_train[kp_idx, :] < 2.) & (delta < 0.))[0]    
    nl01_idx = np.where((u_train[kp_idx, :] < 1.) & (delta < 0.))[0] 

    idx_len_dict={'nl00_idx': len(nl00_idx),\
                  'nl10_idx': len(nl10_idx),\
                  'nl20_idx': len(nl20_idx),\
                  'nl30_idx': len(nl30_idx),\
                  'nl40_idx': len(nl40_idx),\
                  'nl50_idx': len(nl50_idx),\
                  'nl60_idx': len(nl60_idx),\
                  'nl70_idx': len(nl70_idx),\
                  'nl80_idx': len(nl80_idx),\
                  'nl01_idx': len(nl01_idx),\
                  'nl11_idx': len(nl11_idx),\
                  'nl21_idx': len(nl21_idx),\
                  'nl31_idx': len(nl31_idx),\
                  'nl41_idx': len(nl41_idx),\
                  'nl51_idx': len(nl51_idx),\
                  'nl61_idx': len(nl61_idx),\
                  'nl71_idx': len(nl71_idx),\
                  'nl81_idx': len(nl81_idx)}
    idx_dict =   {'nl00_idx': nl00_idx,\
                  'nl10_idx': nl10_idx,\
                  'nl20_idx': nl20_idx,\
                  'nl30_idx': nl30_idx,\
                  'nl40_idx': nl40_idx,\
                  'nl50_idx': nl50_idx,\
                  'nl60_idx': nl60_idx,\
                  'nl70_idx': nl70_idx,\
                  'nl80_idx': nl80_idx,\
                  'nl01_idx': nl01_idx,\
                  'nl11_idx': nl11_idx,\
                  'nl21_idx': nl21_idx,\
                  'nl31_idx': nl31_idx,\
                  'nl41_idx': nl41_idx,\
                  'nl51_idx': nl51_idx,\
                  'nl61_idx': nl61_idx,\
                  'nl71_idx': nl71_idx,\
                  'nl81_idx': nl81_idx}
    return idx_len_dict, idx_dict, delta


def dynamic_prediction(X_k_norm, current_kp, B_nl0, B_nl1, B_nl2, B_nl3, B_nl4, B_nl5, B_nl6, B_nl7, B_nl8):
    Y_k = (B_nl0 * ((current_kp < 1.)) + B_nl1 * ((current_kp >= 1.) & (current_kp < 2.)) + B_nl2 * ((current_kp >= 2.) & (current_kp < 3.)) + \
        B_nl3 * ((current_kp >= 3.) & (current_kp < 4.)) + B_nl4 * ((current_kp >= 4.) & (current_kp < 5.)) + B_nl5 * ((current_kp >= 5.) & (current_kp < 6.)) + \
            B_nl6 * ((current_kp >= 6.) & (current_kp < 7.)) + B_nl7 * ((current_kp >= 7.) & (current_kp < 8.)) + B_nl8 * ((current_kp >= 8.)) ) @ X_k_norm
    return Y_k

def dynamic_prediction_dual_regimes(X_k_norm, current_kp, delta_f10, B_nl00, B_nl10, B_nl20, B_nl30, B_nl40, B_nl50, B_nl60, B_nl70, B_nl80, \
    B_nl01, B_nl11, B_nl21, B_nl31, B_nl41, B_nl51, B_nl61, B_nl71, B_nl81):
    Y_k = (delta_f10 >= 0) * (B_nl00 * ((current_kp < 1.)) + B_nl10 * ((current_kp >= 1.) & (current_kp < 2.)) + B_nl20 * ((current_kp >= 2.) & (current_kp < 3.)) + \
        B_nl30 * ((current_kp >= 3.) & (current_kp < 4.)) + B_nl40 * ((current_kp >= 4.) & (current_kp < 5.)) + B_nl50 * ((current_kp >= 5.) & (current_kp < 6.)) + \
            B_nl60 * ((current_kp >= 6.) & (current_kp < 7.)) + B_nl70 * ((current_kp >= 7.) & (current_kp < 8.)) + B_nl80 * ((current_kp >= 8.)) ) @ X_k_norm + \
                (delta_f10 < 0) * (B_nl01 * ((current_kp < 1.)) + B_nl11 * ((current_kp >= 1.) & (current_kp < 2.)) + B_nl21 * ((current_kp >= 2.) & (current_kp < 3.)) + \
                    B_nl31 * ((current_kp >= 3.) & (current_kp < 4.)) + B_nl41 * ((current_kp >= 4.) & (current_kp < 5.)) + B_nl51 * ((current_kp >= 5.) & (current_kp < 6.)) + \
                        B_nl61 * ((current_kp >= 6.) & (current_kp < 7.)) + B_nl71 * ((current_kp >= 7.) & (current_kp < 8.)) + B_nl81 * ((current_kp >= 8.)) ) @ X_k_norm
    return Y_k


def dynamic_prediction_v7(X_k_norm, current_kp, B_nl0, B_nl01, B_nl1, B_nl12, B_nl2, B_nl23, B_nl3, B_nl34, B_nl4, B_nl45, B_nl5, B_nl6, B_nl7, B_nl8):
    Y_k = (B_nl0 * ((current_kp < .5))+ B_nl01 * ((current_kp >= .5) & (current_kp < 1.)) + B_nl1 * ((current_kp >= 1.) & (current_kp < 1.5)) + B_nl12 * ((current_kp >= 1.5) & (current_kp < 2.)) + \
           B_nl2 * ((current_kp >= 2.) & (current_kp < 2.5)) + B_nl23 * ((current_kp >= 2.5) & (current_kp < 3.)) + \
        B_nl3 * ((current_kp >= 3.) & (current_kp < 3.5)) + B_nl34 * ((current_kp >= 3.5) & (current_kp < 4.)) + \
            B_nl4 * ((current_kp >= 4.) & (current_kp < 4.5)) + B_nl45 * ((current_kp >= 4.5) & (current_kp < 5.)) + B_nl5 * ((current_kp >= 5.) & (current_kp < 6.)) + \
            B_nl6 * ((current_kp >= 6.) & (current_kp < 7.)) + B_nl7 * ((current_kp >= 7.) & (current_kp < 8.)) + B_nl8 * ((current_kp >= 8.)) ) @ X_k_norm
    return Y_k



# def build_inputs(Y, z1, drivers, normalization_method, model_params, f10_idx, mdl = 'sindy', pca_couplings = 2, all_pca = False):
#     kp_idx = f10_idx + 1
#     if mdl == 'sindy':
#         Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
#         Y_reg_norm_dict_sindy = Y_reg_norm_dict
        
#         if all_pca == False:
#             X_for_sindy_train = np.copy(np.concatenate([z1[pca_couplings, :].reshape((1, -1)), drivers[f10_idx:, :]], axis = 0)) 
#         else:
#             X_for_sindy_train = np.copy(np.concatenate([z1[pca_couplings, :], drivers[f10_idx:, :]], axis = 0)) 
            
#         n_x = X_for_sindy_train.shape[0]
#         print(X_for_sindy_train.max(), X_for_sindy_train.min())
#         input_features = ['x_'+ str(k+1).zfill(2) for k in range(X_for_sindy_train.shape[0])]
#         library_dict = create_library_functions(np.copy(X_for_sindy_train.reshape((n_x, -1)).T), model_params['functions'], input_features)
#         theta_train, theta_features = library_dict['theta'].T, library_dict['library_feature_names']
#         if all_pca == False:
#             X_train = np.copy(np.concatenate([theta_train, np.delete(z1, pca_couplings, axis = 0)], axis = 0))
#         else:
#             X_train = np.copy(np.concatenate([theta_train, np.delete(z1, pca_couplings, axis = 0)], axis = 0))
        
#         X_train_norm, X_reg_norm_dict = normalize_array(X_train[1:, :], method = normalization_method)
#         X_reg_norm_dict_sindy = X_reg_norm_dict
#         X_train_norm = np.concatenate([X_train[0, :].reshape((1, -1)), drivers[1:f10_idx, :], X_train_norm])
        
#         print(X_train_norm.max(), X_train_norm.min())
#         return X_train_norm, X_reg_norm_dict_sindy, Y_train_norm, Y_reg_norm_dict_sindy, theta_features
    
#     elif mdl == 'dmd':
#         Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
#         Y_reg_norm_dict_dmd = Y_reg_norm_dict
#         X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis = 0)) 
        
#         X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
#         X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
#         X_reg_norm_dict_dmd = X_reg_norm_dict     
#         return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd
    
#     elif mdl == 'nl-dmd':
#         Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
#         Y_reg_norm_dict_dmd = Y_reg_norm_dict
#         X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), 
#             (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1))], axis = 0)) 
        
#         X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
#         X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
#         X_reg_norm_dict_dmd = X_reg_norm_dict     
#         return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd




def select_indices(row_vector, th1, th2, n_backward, n_forward):
    indices_above_threshold = np.where((row_vector >= th1) & (row_vector < th2))[1]
    extended_indices = set()
    data_length = row_vector.shape[1]
    for index in indices_above_threshold:
        start = max(index - n_backward, 0)
        end = min(index + n_forward, data_length - 1)
        valid_indices = range(start, end + 1)
        extended_indices.update(valid_indices)

    extended_indices_array = np.array(sorted(extended_indices))
    return extended_indices_array




basis_functions_latex_dict = {
    'plain': r"$\{F_{10}, K_p, F_{10}\cdot K_p, K_p^2\}$",
    'poly': r"$\{x\}$",
    'poly2': r"$\{x^2\}$",
    'poly3': r"$\{x^3\}$",
    'poly_all': r"$\{x, x^2, x^3\}$",
    'exp': r"$\{\exp(-x)\}$",
    'poly_exp': r"$\{x, \exp(-x)\}$",
    'poly_exp_2': r"$\{x, \exp(-x), \exp(x)\}$",
    'sincos': r"$\{\sin(x), \cos(x)\}$",
    'poly_sincos': r"$\{x, \sin(x), \cos(x)\}$",
    'poly_sincos2': r"$\{x, \sin\left(\frac{2\pi x}{9000}\right), \cos\left(\frac{2\pi x}{9000}\right)\}$",
    'sincos2': r"$\{\sin\left(\frac{2\pi x}{9000}\right), \cos\left(\frac{2\pi x}{9000}\right)\}$",
    'poly_sincos3': r"$\{x, \sin\left(\frac{2\pi x}{10.84}\right), \cos\left(\frac{2\pi x}{10.84}\right), \sin\left(\frac{2\pi x}{153.29}\right), \cos\left(\frac{2\pi x}{153.29}\right), \sin\left(\frac{2\pi x}{12.24}\right), \cos\left(\frac{2\pi x}{12.24}\right)\}$",
    'sincos3': r"$\{\sin\left(\frac{2\pi x}{10.84}\right), \cos\left(\frac{2\pi x}{10.84}\right), \sin\left(\frac{2\pi x}{153.29}\right), \cos\left(\frac{2\pi x}{153.29}\right), \sin\left(\frac{2\pi x}{12.24}\right), \cos\left(\frac{2\pi x}{12.24}\right)\}$",
    'all_1': r"$\{x, x^2, x^3, \exp(-x), \exp(x), \sin(x), \cos(x), \sin\left(\frac{2\pi x}{9000}\right), \cos\left(\frac{2\pi x}{9000}\right), \sin\left(\frac{2\pi x}{10.84}\right), \cos\left(\frac{2\pi x}{10.84}\right), \sin\left(\frac{2\pi x}{153.29}\right), \cos\left(\frac{2\pi x}{153.29}\right), \sin\left(\frac{2\pi x}{12.24}\right), \cos\left(\frac{2\pi x}{12.24}\right)\}$",
    'all_2': r"$\{x, x^2, x^3, \exp(-x), \sin(x), \cos(x), \sin\left(\frac{2\pi x}{9000}\right), \cos\left(\frac{2\pi x}{9000}\right), \sin\left(\frac{2\pi x}{10.84}\right), \cos\left(\frac{2\pi x}{10.84}\right), \sin\left(\frac{2\pi x}{153.29}\right), \cos\left(\frac{2\pi x}{153.29}\right), \sin\left(\frac{2\pi x}{12.24}\right), \cos\left(\frac{2\pi x}{12.24}\right)\}$"
}


def calculate_ensemble_stats(df_mae_all_train):
    df_mae_all_train_agg = df_mae_all_train.groupby(['mdl', 'alpha_ridge', 'basis_functions']).agg(mape = ('mape', 'mean')).reset_index()
    
    df_mae_all_train_agg['mape'] = df_mae_all_train_agg['mape'] * 100
    df_mae_all_train_agg.loc[df_mae_all_train_agg.mape > 20000, 'mape'] = np.inf
    
    df_mae_all_train_agg['mape'] = df_mae_all_train_agg['mape'].replace([float('inf'), -float('inf')], np.nan)
    df_mae_all_train_agg = df_mae_all_train_agg.dropna()
    
    
    df_min_mape_train = df_mae_all_train_agg.loc[
        df_mae_all_train_agg.groupby(['mdl', 'basis_functions'])['mape'].idxmin()
    ].copy()
    
    df_min_mape_train = df_min_mape_train.reset_index(drop=True)
    
    filtered_df = df_mae_all_train.merge(
        df_min_mape_train[['basis_functions', 'alpha_ridge']],
        on=['basis_functions', 'alpha_ridge']
    )
    
    stats = filtered_df.groupby('basis_functions')['mape'].agg(['mean', 'std']).reset_index()
    return stats

#Ensemble modeling
def calculate_ensemble_model(df_mae_all_test, ys_df_test, x_train_svd_obj, x_train_dict, x2_actual, drivers, idx_actual, doy):
    df_mae_all_test_agg = df_mae_all_test.groupby(['mdl', 'basis_functions', 'alpha_ridge']).agg(mape = ('mape', 'mean')).reset_index().copy()
    
    df_mae_all_test_agg['mape'] = df_mae_all_test_agg['mape'] * 100
    df_mae_all_test_agg.loc[df_mae_all_test_agg.mape > 20000, 'mape'] = np.inf
    # Replace infinite values in MAPE with 0
    df_mae_all_test_agg['mape'] = df_mae_all_test_agg['mape'].replace([float('inf'), -float('inf')], np.nan)
    df_mae_all_test_agg = df_mae_all_test_agg.dropna()
    
    
    df_min_mape = df_mae_all_test_agg.loc[
        df_mae_all_test_agg.groupby(['mdl', 'basis_functions'])['mape'].idxmin()
    ].copy()
    
    # Reset index for cleaner output
    df_min_mape = df_min_mape.reset_index(drop=True)
    
    selected_bf_dict = df_min_mape.set_index('basis_functions')['alpha_ridge'].to_dict()
    
    max_y = 1.5*ys_df_test.loc[(ys_df_test.model == 'sindy') & (ys_df_test.alpha_ridge == 0.5) & (ys_df_test.basis_functions == 'poly')][['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9']].values.max()   
    min_y = 1.5*ys_df_test.loc[(ys_df_test.model == 'sindy') & (ys_df_test.alpha_ridge == 0.5) & (ys_df_test.basis_functions == 'poly')][['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9']].values.min()   
    
    
    mdl_ensemble_lst = []
    for bf in selected_bf_dict.keys():
        a = selected_bf_dict[bf]
        mdl_array = ys_df_test.fillna(0).loc[(ys_df_test.model == 'sindy') & (ys_df_test.alpha_ridge == a) & (ys_df_test.basis_functions == bf)][['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9']].values
        #mdl_array = np.clip(mdl_array, min_y, max_y)
        mdl_ensemble_lst.append(mdl_array)
    
    mdl = np.zeros(mdl_ensemble_lst[0].shape)
    for mdl in mdl_ensemble_lst:
        mdl = mdl + mdl/len(mdl_ensemble_lst)
    
        
    predicted_ensemble_state = x_train_svd_obj.U @ mdl.T + x_train_dict['x_mean']
    rho_fcst_ensemble = np.float64(10**predicted_ensemble_state)
    rho_actual_test = np.float64(10**x2_actual[:, idx_actual])
    error = rho_fcst_ensemble - rho_actual_test
    pae = np.abs(error)/rho_actual_test
    instant_pae = np.mean(pae, axis = 0, keepdims = True)
    mape = np.mean(pae) 
    
    df_mae_ensemble = pd.DataFrame(np.concatenate([frcst_periods, \
        drivers[0, idx_actual].T.reshape((-1, 1)), doy[idx_actual].reshape((-1, 1)), \
            drivers[kp_idx, idx_actual].T.reshape((-1, 1)), drivers[f10_idx, idx_actual].T.reshape((-1, 1)), \
                np.sum(error, axis = 0, keepdims = True).T, np.sum(rho_actual_test, axis = 0, keepdims = True).T, instant_pae.T.reshape((-1, 1))], axis = 1 ), \
                    columns = ['period', 'year', 'doy', 'kp', \
                        'f107', 'error', 'actual', 'mape'])
    df_mae_ensemble['cycle'] = (df_mae_ensemble['period'] == 0).cumsum() - 1
    df_mae_ensemble['ones'] = 1
    df_mae_ensemble['mdl'] = model_name
    overall_stats_ensemble = df_mae_ensemble.loc[(df_mae_ensemble.kp >= th1) & (df_mae_ensemble.kp < th2)][['kp', 'f107', 'mape']].mean()
    print('Ensemble model')
    print(f'{n_days} days dynamic forecast on test dataset for {model_name}, with {instant_pae.shape} samples')
    print(f'{model_name} overall MAPE:{100. * overall_stats_ensemble["mape"]:.2f}%')
    mape_year_ensemble = df_mae_ensemble.loc[(df_mae_ensemble.kp >= th1) & (df_mae_ensemble.kp < th2)].groupby(['year']).agg({'period':"mean" , 'kp':"mean" , \
                        'f107':"mean" , 'error':"mean" , 'actual':"mean" , 'mape':"mean"}).reset_index()
    mape_year_ensemble.mape = 100. * mape_year_ensemble.mape
    print(f'Ensemble MAPE (%) per year:\n')
    print(mape_year_ensemble.drop(columns = 'period').round(2))
    dual_output = mdl 
    return df_mae_ensemble, dual_output, predicted_ensemble_state

def escape_latex(text):
    return text.replace('_', r' ').replace('$', r'\$')

def generate_latex_table_ensemble(df, a):
    """
    Generates LaTeX code for a table with the averages of Model, Basis Functions, and MAPE.
    """
    
    # Multiply MAPE by 100 to convert to percentage

    # Grouping the DataFrame and calculating average MAPE
    grouped = df.groupby(['mdl', 'year', 'basis_functions', 'alpha_ridge']).agg(
        avg_mape=('mape', 'mean')
    ).reset_index()

    grouped['avg_mape'] = grouped['avg_mape'] * 100
    grouped.loc[grouped.avg_mape > 20000] = np.inf
    
    # Replace infinite values in MAPE with 0
    grouped['avg_mape'] = grouped['avg_mape'].replace([float('inf'), -float('inf')], np.nan)
    grouped = grouped.dropna()
    
    # Filter rows where avg_mape is within a valid range
    grouped = grouped.loc[grouped['avg_mape'] <= 100.]
    
    
    # Begin LaTeX table structure
    latex_code = rf"""
\begin{{table}}[H]
    \centering
    \caption{{Ridge parameter: {a}, Averages of Model, Basis Functions, and MAPE. Train dataset}}
    \begin{{tabular}}{{|c|c|p{{4cm}}|c|}}
        \hline
        \textbf{{Model}} & \textbf{{Year}} & \textbf{{Basis Functions}} & \textbf{{Average MAPE (\%)}} \\ \hline
    """
    
    # Add rows from the grouped DataFrame
    for _, row in grouped.iterrows():
        bf_label_latex = row['basis_functions'].replace('_', r'\_')
        latex_code += (
            f"{row['mdl']} & {row['year']} & {bf_label_latex} & {row['avg_mape']:.2f} \\\\ \\hline\n"
        )
    
    # Close the LaTeX table structure
    latex_code += rf"""
        \hline
    \end{{tabular}}
    \label{{table:model_averages_mape_train}}
\end{{table}}
    """
    
    print(latex_code)

def generate_latex_table(df, a):
    """
    Generates LaTeX code for a table with the averages of Model, Basis Functions, and MAPE.
    """
    
    # Multiply MAPE by 100 to convert to percentage

    # Grouping the DataFrame and calculating average MAPE
    grouped = df.groupby(['mdl', 'basis_functions', 'alpha_ridge']).agg(
        avg_mape=('mape', 'mean')
    ).reset_index()

    grouped['avg_mape'] = grouped['avg_mape'] * 100
    grouped.loc[grouped.avg_mape > 20000] = np.inf
    
    # Replace infinite values in MAPE with 0
    grouped['avg_mape'] = grouped['avg_mape'].replace([float('inf'), -float('inf')], np.nan)
    grouped = grouped.dropna()
    
    # Filter rows where avg_mape is within a valid range
    grouped = grouped.loc[grouped['avg_mape'] <= 100.]
    
    
    # Begin LaTeX table structure
    latex_code = rf"""
\begin{{table}}[H]
    \centering
    \caption{{Ridge parameter: {a}, Averages of Model, Basis Functions, and MAPE. Train dataset}}
    \begin{{tabular}}{{|c|p{{4cm}}|c|}}
        \hline
        \textbf{{Model}} & \textbf{{Basis Functions}} & \textbf{{Average MAPE (\%)}} \\ \hline
    """
    
    # Add rows from the grouped DataFrame
    for _, row in grouped.iterrows():
        bf_label_latex = row['basis_functions'].replace('_', r'\_')
        latex_code += (
            f"{row['mdl']} & {bf_label_latex} & {row['avg_mape']:.2f} \\\\ \\hline\n"
        )
    
    # Close the LaTeX table structure
    latex_code += rf"""
        \hline
    \end{{tabular}}
    \label{{table:model_averages_mape_train}}
\end{{table}}
    """
    
    print(latex_code)

def get_size(obj, seen=None):
    """Recursively calculate the size of objects."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Mark the object as seen
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif isinstance(obj, (list, set, tuple)):
        size += sum(get_size(i, seen) for i in obj)
    return size

    

def gather_data_ext(train_years = [2000, 2001, 2002], test_years = [2003], is_holdout = True, folder = './data/ext_inputs/tiegcm_ext_yearly/'):
    if is_holdout:
        print(f'Downloading {len(train_years)} years')
        state_data, drivers_data = preprocess_from_h5_ext(train_years, folder)
        # Print memory usage of each variable
        print(f'---------state_data:{ get_size(state_data, seen=None)}---------drivers_data:{ get_size(drivers_data, seen=None)}--------------------------------------')

        dataset_dict = {'state_data':state_data, 'drivers_data':drivers_data}
        
        return dataset_dict        
    else:
        print(f'Downloading {len(train_years)} years for train')
        state_data_train, drivers_data_train = preprocess_from_h5(train_years)
        print(f'Downloading {len(test_years)} years for test')
        state_data_test, drivers_data_test = preprocess_from_h5(test_years)
        dataset_dict = {'state_train':state_data_train, 'drivers_train':drivers_data_train, 'state_test':state_data_test, 'drivers_test':drivers_data_test}
        return dataset_dict


# def build_inputs(Y, z1, drivers, normalization_method, model_params, f10_idx, mdl = 'sindy', pca_couplings = 2, more_sindy_vars = False):
#     kp_idx = f10_idx + 1
#     if mdl == 'sindy':
#         Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
#         Y_reg_norm_dict_sindy = Y_reg_norm_dict
        
#         if more_sindy_vars == False:
#             X_for_sindy_train = np.copy(np.concatenate([z1[pca_couplings, :].reshape((1, -1)), drivers[f10_idx:, :]], axis = 0)) 
#         else:
#             X_for_sindy_train = np.copy(np.concatenate([z1[pca_couplings, :], drivers[f10_idx:, :]], axis = 0)) 
            
#         n_x = X_for_sindy_train.shape[0]
#         print(X_for_sindy_train.max(), X_for_sindy_train.min())
#         if more_sindy_vars == False:
#             input_features = [f'x_{pca_couplings[0]}'] + ['f107', 'kp']
#         else:
#             input_features = [f'x_{pc}' for pc in pca_couplings] + ['f107', 'kp']
#         library_dict = create_library_functions(np.copy(X_for_sindy_train.reshape((n_x, -1)).T), model_params['functions'], input_features)
#         theta_train, theta_features = library_dict['theta'].T, library_dict['library_feature_names']
#         if more_sindy_vars == False:
#             X_train = np.copy(np.concatenate([theta_train, np.delete(z1, pca_couplings, axis = 0)], axis = 0))
#         else:
#             X_train = np.copy(np.concatenate([theta_train, np.delete(z1, pca_couplings, axis = 0)], axis = 0))
        
#         X_train_norm, X_reg_norm_dict = normalize_array(X_train[1:, :], method = normalization_method)
#         X_reg_norm_dict_sindy = X_reg_norm_dict
#         X_train_norm = np.concatenate([X_train[0, :].reshape((1, -1)), drivers[1:f10_idx, :], X_train_norm])
        
#         print(X_train_norm.max(), X_train_norm.min())
#         return X_train_norm, X_reg_norm_dict_sindy, Y_train_norm, Y_reg_norm_dict_sindy, theta_features
    
#     elif mdl == 'dmd':
#         Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
#         Y_reg_norm_dict_dmd = Y_reg_norm_dict
#         X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis = 0)) 
        
#         X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
#         X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
#         X_reg_norm_dict_dmd = X_reg_norm_dict     
#         return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd
    
#     elif mdl == 'nl-dmd':
#         Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
#         Y_reg_norm_dict_dmd = Y_reg_norm_dict
#         X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), 
#             (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1))], axis = 0)) 
        
#         X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
#         X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
#         X_reg_norm_dict_dmd = X_reg_norm_dict     
#         return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd



def build_inputs(Y, z1, drivers, normalization_method, model_params, f10_idx, mdl = 'sindy', pca_couplings = 2, all_pca = False):
    kp_idx = f10_idx + 1
    if mdl == 'sindy':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_sindy = Y_reg_norm_dict
        
        if all_pca == False:
            X_for_sindy_train = np.copy(np.concatenate([z1[pca_couplings, :].reshape((1, -1)), drivers[f10_idx:, :]], axis = 0)) 
        else:
            X_for_sindy_train = np.copy(np.concatenate([z1[pca_couplings, :], drivers[f10_idx:, :]], axis = 0)) 
            
        n_x = X_for_sindy_train.shape[0]
        print('Minmax before pre sindy normalization')
        print(X_for_sindy_train.max(), X_for_sindy_train.min())
        X_library_matrix_inputs_norm, X_library_matrix_inputs_norm_dict = normalize_array(X_for_sindy_train, method = normalization_method)
        print('Minmax after pre sindy normalization')
        print(X_library_matrix_inputs_norm.max(), X_library_matrix_inputs_norm.min())
        
        if all_pca == False:
            input_features = [f'x_{pca_couplings[0]}'] + ['f107', 'kp']
        else:
            input_features = [f'x_{pc}' for pc in pca_couplings] + ['f107', 'kp']
        library_dict = create_library_functions(np.copy(X_library_matrix_inputs_norm.reshape((n_x, -1)).T), model_params['functions'], input_features)
        theta_train, theta_features = library_dict['theta'].T, library_dict['library_feature_names']
        if all_pca == False:
            X_train = np.copy(np.concatenate([theta_train, np.delete(z1, pca_couplings, axis = 0)], axis = 0))
        else:
            X_train = np.copy(np.concatenate([theta_train, np.delete(z1, pca_couplings, axis = 0)], axis = 0))
        
        X_train_norm, X_reg_norm_dict = normalize_array(X_train[1:, :], method = normalization_method)
        X_reg_norm_dict_sindy = X_reg_norm_dict
        X_train_norm = np.concatenate([X_train[0, :].reshape((1, -1)), drivers[1:f10_idx, :], X_train_norm])
        
        print(X_train_norm.max(), X_train_norm.min())
        return X_train_norm, X_reg_norm_dict_sindy, Y_train_norm, Y_reg_norm_dict_sindy, theta_features, X_library_matrix_inputs_norm_dict
    
    elif mdl == 'dmd':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis = 0)) 
        
        X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
        X_reg_norm_dict_dmd = X_reg_norm_dict     
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd
    
    elif mdl == 'nl-dmd':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), 
            (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1))], axis = 0)) 
        
        X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
        X_reg_norm_dict_dmd = X_reg_norm_dict     
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd


def move_column(array, from_col, to_col):
    return np.insert(np.delete(array, from_col, axis=1), to_col, array[:, from_col], axis=1)

def plotCMap(data, title, cbar_ticks = 20, x_labels=['all', 'moderate', 'quiet', 'strong'], y_labels=['all', 'moderate', 'quiet', 'strong']):
    # Define a diverging color map with gray at zero
    cmap = plt.cm.get_cmap('coolwarm', 10)  # 10 color gradations
    norm = TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=np.max(data))  # Center at zero

    fig = plt.figure(figsize=(10, 7))
    plt.subplot(1, 1, 1)
    im = plt.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    
    # Add colorbar with ticks for each gradation
    cbar = plt.colorbar(im, extend='both', ticks=np.linspace(np.min(data), np.max(data), cbar_ticks))
    cbar.set_label('')
    cbar.ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(np.min(data), np.max(data), cbar_ticks)])
    
    # Set x and y axis labels
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=60)
    plt.yticks(ticks=range(len(y_labels)), labels=y_labels, rotation=0)
    
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    plt.show()
