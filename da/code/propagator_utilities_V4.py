import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.linalg import logm
import utilities_ds as u
import glob
import datetime as dt
import os
from ftplib import FTP
from io import StringIO, BytesIO
import re
import zipfile
import os
from pymsis import msis
from scipy.interpolate import RegularGridInterpolator as rgi
from itertools import product, combinations_with_replacement
from pathlib import Path
import warnings
from scipy.interpolate import interp1d


def matlab_datenum_to_py_datetime(dn):
    dn = np.asarray(dn, dtype=float)
    out = []
    for x in dn.flatten():
        day = int(x)
        frac = x - day
        date = dt.date.fromordinal(day - 366)
        seconds = frac * 86400.0
        out.append(dt.datetime.combine(date, dt.time()) +
                   dt.timedelta(seconds=seconds))
    return np.array(out, dtype=object).reshape(dn.shape)


def add_lags(data, lags=[1, 2, 3]):
    "Constructs the lagged version of the input vector, with lags the input lags list. If, for instance,"
    "there is a pair of rows with shapes [2, m_times], then the output will have shape [2 + 2*len(lags), m_times]"
    if lags[0] == 0:
        return data.T

    n_samples, n_features = data.T.shape
    lagged_data = []

    for lag in lags:
        padded = np.zeros((n_samples, n_features))
        padded[lag:] = data.T[:-lag]
        lagged_data.append(padded)

    return np.hstack([data.T] + lagged_data).T


def load_mat_any(path):
    # MATLAB v7.3 files are HDF5 and start with the HDF5 signature: b"\x89HDF\r\n\x1a\n"
    with open(path, "rb") as f:
        sig = f.read(8)

    is_hdf5 = (sig == b"\x89HDF\r\n\x1a\n")

    if is_hdf5:
        # v7.3: use h5py
        f = h5py.File(path, "r")
        return f, "h5py"  # remember to close f when done
    else:
        # pre-v7.3: use scipy
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        return data, "loadmat"


def build_ar_inputs(u_train, z1_train, z2_train, u_test, ar_lags, n_components, f10_idx, kp_idx, ar_drivers_flag, ar_state_flag):
    if ar_drivers_flag is False:
        u_train_ar = np.copy(u_train)
        u_test_ar = u_test

    else:
        u_train_ar = np.concatenate([np.copy(u_train),
                                     add_lags(np.copy(u_train)[[f10_idx, f10_idx + 1, kp_idx], :], lags=ar_lags)[3:, :]], axis=0)

        u_test_ar = np.concatenate([np.copy(u_test),
                                    add_lags(np.copy(u_test)[[f10_idx, f10_idx + 1, kp_idx], :], lags=ar_lags)[3:, :]], axis=0)

    z1_train_ar = add_lags(z1_train, lags=ar_lags)[n_components:, :]
    z2_train_ar = add_lags(z2_train, lags=ar_lags)[n_components:, :]

    if ar_state_flag is False:
        print('Not AR states in the run.')

    if u_train_ar.shape[1] == z1_train_ar.shape[1]:
        u_train_ar = np.concatenate([u_train_ar,
                                     z1_train_ar], axis=0)
    else:
        u_train_ar = np.concatenate([u_train_ar,
                                     np.zeros((int(n_components*max(ar_lags)), u_train_ar.shape[1]))], axis=0)

    return u_train_ar, z1_train_ar, z2_train_ar, u_test_ar


def interpolate_matrix_rows(matrix, time_conversion_ratio):
    # time_conversion_ratio is the number of new intervals between the current
    n, m = matrix.shape
    converted_length = int(m * time_conversion_ratio)
    result = np.zeros((n, converted_length))

    for i in range(n):
        t_original = np.arange(m)
        t_interpolated = np.linspace(0, m - 1, converted_length)

        result[i, :] = scipy.interpolate.CubicSpline(
            t_original, matrix[i, :])(t_interpolated)

    return result


def interpolate_matrix_rows_linear_fast(matrix: np.ndarray, time_conversion_ratio: float) -> np.ndarray:
    """
    Linear interpolation along axis=1 (columns) for each row.

    time_conversion_ratio: factor for new number of samples.
      new_m = round(m * time_conversion_ratio)
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D (n, m)")

    n, m = matrix.shape
    new_m = int(np.round(m * time_conversion_ratio))
    if new_m <= 1 or m <= 1:
        # nothing meaningful to interpolate
        return matrix[:, :max(new_m, 0)].copy()

    # New sample positions in "old index" coordinates: [0, m-1]
    x_new = np.linspace(0.0, m - 1.0, new_m, dtype=np.float64)

    # Neighbor indices
    x0 = np.floor(x_new).astype(np.int64)
    x1 = np.minimum(x0 + 1, m - 1)

    # Linear weights
    w = (x_new - x0).astype(matrix.dtype, copy=False)  # shape (new_m,)

    # Gather and blend (broadcast over rows)
    y0 = matrix[:, x0]          # (n, new_m)
    y1 = matrix[:, x1]          # (n, new_m)
    return y0 + (y1 - y0) * w   # w broadcasts over axis 0


def discrete_to_continuous(A_d, B_d, delta_t):
    A_c = (1 / delta_t) * logm(A_d)
    B_c = A_c @ (np.linalg.inv(A_d - np.eye(A_d.shape[0])) @ B_d)
    return A_c, B_c


def change_resolution(A_d1, B_d1, T1, T2):
    A_d2 = scipy.linalg.fractional_matrix_power(A_d1, T2 / T1)
    A_d2 = np.real_if_close(A_d2, tol=1000)
    B_d2 = (
        A_d2 - np.eye(A_d2.shape[0])) @ np.linalg.inv(A_d1 - np.eye(A_d1.shape[0])) @ B_d1
    return A_d2, B_d2


def move_column(array, from_col, to_col):
    return np.insert(np.delete(array, from_col, axis=1), to_col, array[:, from_col], axis=1)


def normalize_array(array, method='std'):

    x = np.copy(array)

    if method == 'mean':
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_norm = (x - x_mean)
        norm_dict = {'x_mean': x_mean}

    elif method == 'std':
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        x_norm = (x - x_mean) / x_std
        norm_dict = {'x_mean': x_mean, 'x_std': x_std}

    elif method == 'minmax':
        max_values = np.max(x, axis=1, keepdims=True)
        min_values = np.min(x, axis=1, keepdims=True)
        x_norm = (x - min_values) / (max_values - min_values)
        norm_dict = {'x_min': min_values, 'x_max': max_values}

    elif method == 'none':
        x_norm = x
        norm_dict = {}

    return x_norm, norm_dict


def normalize_with_dict(x, norm_dict, method):
    x_temp = np.copy(x)
    if method == 'mean':
        x_norm = x_temp - norm_dict['x_mean']
    elif method == 'std':
        x_norm = (x_temp - norm_dict['x_mean']) / norm_dict['x_std']
    elif method == 'minmax':
        x_norm = (x_temp - norm_dict['x_min']) / \
            (norm_dict['x_max'] - norm_dict['x_min'])
    elif method == 'none':
        x_norm = x_temp
    return x_norm


def denormalize(x, norm_dict, method):
    x_temp = np.copy(x)
    if method == 'mean':
        x_orig = x_temp + norm_dict['x_mean']
    elif method == 'std':
        x_orig = x_temp * norm_dict['x_std'] + norm_dict['x_mean']
    elif method == 'minmax':
        x_orig = x_temp * (norm_dict['x_max'] -
                           norm_dict['x_min']) + norm_dict['x_min']
    elif method == 'none':
        x_orig = x_temp
    return x_orig


def build_regression_inputs(Y, z1, all_drivers, normalization_method, model_params, f10_idx, mdl='sindy', pca_couplings=2, all_pca=False):
    kp_idx = f10_idx + 3
    f10a_idx = f10_idx + 2
    drivers = all_drivers[:(kp_idx + 1), :]
    print(drivers.shape)
    ar_drivers = all_drivers[(kp_idx + 1):, :]
    print(ar_drivers.shape)

    if mdl == 'sindy':
        Y_train_norm, Y_reg_norm_dict = normalize_array(
            Y, method=normalization_method)
        Y_reg_norm_dict_sindy = Y_reg_norm_dict

        if all_pca == False:
            X_for_sindy_train = np.copy(
                np.concatenate([z1, drivers[1:, :]], axis=0))
        else:
            X_for_sindy_train = np.copy(
                np.concatenate([z1, drivers[1:, :]], axis=0))

        n_x = X_for_sindy_train.shape[0]
        print('Minmax before pre sindy normalization')
        print(X_for_sindy_train.max(), X_for_sindy_train.min())

        X_library_matrix_inputs_norm_dict = None

        if all_pca == False:
            # + ['t1', 't2', 't3', 't4', 'f107a', 'f107', 'kp']
            input_features = [f'x_{k}' for k in range(
                z1.shape[0])] + ['t1', 't2', 't3', 't4', 'f107', 'f107a', 'kp']
        else:
            # + ['t1', 't2', 't3', 't4', 'f107a', 'f107', 'kp']
            input_features = [f'x_{k}' for k in range(
                z1.shape[0])] + ['t1', 't2', 't3', 't4', 'f107', 'f107a', 'kp']

        library_dict = create_library_functions(np.copy(X_for_sindy_train.reshape(
            (n_x, -1)).T), model_params['functions'], input_features)
        theta_train, theta_features = library_dict['theta'].T, library_dict['library_feature_names']
        print("Theta features ", theta_features)
        if all_pca == False:
            X_train = np.copy(np.concatenate(
                [theta_train, ar_drivers, z1], axis=0))
        else:
            X_train = np.copy(np.concatenate(
                [theta_train, ar_drivers, z1], axis=0))

        X_train = np.concatenate([np.copy(X_train[0, :]).reshape(
            (1, -1)), np.copy(X_train[15:, :])], axis=0)
        # np.copy(X_train[11:, :]) since X_train includes the first row which is constant

        X_train_norm, X_reg_norm_dict = normalize_array(
            X_train[1:, :], method='std')  # <------
        X_reg_norm_dict_sindy = X_reg_norm_dict
        X_train_norm = np.concatenate([X_train[0, :].reshape(
            (1, -1)), drivers[1:f10_idx, :], X_train_norm])
        # X_train_norm = np.concatenate([X_train[0, :].reshape((1, -1)), X_train_norm])
        # print(X_train_norm[:, :10])
        print(X_train_norm.max(), X_train_norm.min())
        return X_train_norm, X_reg_norm_dict_sindy, Y_train_norm, Y_reg_norm_dict_sindy, \
            theta_features, X_library_matrix_inputs_norm_dict

    elif mdl == 'dmd':
        Y_train_norm, Y_reg_norm_dict = normalize_array(
            Y, method=normalization_method)
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis=0))

        X_train_norm, X_reg_norm_dict = normalize_array(
            X_train, method=normalization_method)
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm])
        X_reg_norm_dict_dmd = X_reg_norm_dict
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd

    elif mdl == 'dmd-c':
        Y_train_norm, Y_reg_norm_dict = normalize_array(
            Y, method=normalization_method)
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)),
                                          (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), z1], axis=0))
        X_train_norm, X_reg_norm_dict = normalize_array(
            X_train, method=normalization_method)
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm])
        X_reg_norm_dict_dmd = X_reg_norm_dict
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd

    elif mdl == 'nl-ddm':
        Y_train_norm, Y_reg_norm_dict = normalize_array(
            Y, method=normalization_method)
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)),
                                          (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), z1], axis=0))
        X_train_norm, X_reg_norm_dict = normalize_array(
            X_train, method=normalization_method)
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm])
        X_reg_norm_dict_dmd = X_reg_norm_dict
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd


def import_satellite_data(satellite_name):
    columns = [
        'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
        'accelerometer_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'
    ]

    file_paths = sorted(
        glob.glob(f'../../dmd_sindy/local_archive/{satellite_name}/*.txt'))
    df_list = []

    for file_path in file_paths:
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            comment='#',
            names=columns
        )
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date', 'time', 'GPS'], inplace=True)

        df_list.append(df)

    champ_all = pd.concat(df_list)
    champ_all.sort_index(inplace=True)
    champ_hourly = champ_all.loc[(champ_all.index.minute == 0) &
                                 (champ_all.index.second == 0)].copy()

    return champ_all, champ_hourly


def mat_to_lambda_dict(obj):
    temp_dict = {}

    for basis_code in obj._fieldnames:
        if basis_code.count('p') >= 1:
            powers = re.findall(r'(?:\d+\.\d+|\d+|\.\d+)', basis_code)

            if len(powers) != 1:
                raise ValueError(f"Ambiguous power in {basis_code}: {powers}")

            power = float(powers[0])

            # bind power at definition time
            temp_dict[basis_code] = (lambda x, p=power: x**p)

    return temp_dict


def create_library_functions(x, functions_dictionary, feature_names):
    function_names = functions_dictionary.keys()

    feature_names_lst = []
    for tuple in list(product(functions_dictionary, feature_names)):
        feature_names_lst.append(tuple[0] + '(' + tuple[1] + ')')

    feature_names_interactions_lst = []
    for tuple in list(combinations_with_replacement(feature_names_lst, 2)):
        feature_names_interactions_lst.append(tuple[0] + tuple[1])

    all_feature_names = ['1'] + feature_names_lst + \
        feature_names_interactions_lst

    arr = np.concatenate([np.apply_along_axis(
        functions_dictionary[func_name], 0, x) for func_name in function_names], axis=1)
    combs = list(combinations_with_replacement(range(arr.shape[1]), 2))

    left_combinations_index = [p[0] for p in combs]
    right_combinations_index = [p[1] for p in combs]

    theta = np.concatenate([np.ones(arr.shape[0]).reshape(
        (-1, 1)), arr, arr[:, left_combinations_index] * arr[:, right_combinations_index]], axis=1)
    library_dict = {'theta': theta, 'library_feature_names': all_feature_names}
    return library_dict


def create_library_functions_optimized(x, functions_dictionary, feature_names, return_names=True, poly_order=2):

    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    func_names = list(functions_dictionary.keys())
    n_samples, n_feat = x.shape
    n_funcs = len(func_names)

    if feature_names is None:
        feature_names_fixed = [f"x{i}" for i in range(n_feat)]
    else:
        feature_names = list(feature_names)
        if len(feature_names) >= n_feat:
            feature_names_fixed = feature_names[:n_feat]
        else:
            feature_names_fixed = feature_names + \
                [f"x{i}" for i in range(len(feature_names), n_feat)]

    unary_blocks = []
    for fn in func_names:
        y = functions_dictionary[fn](x)
        y = np.asarray(y)

        if y.shape != x.shape:
            if y.shape == (n_samples,):
                y = y.reshape(n_samples, 1).repeat(n_feat, axis=1)
            elif y.shape == (n_samples, 1):
                y = y.repeat(n_feat, axis=1)
            else:
                raise ValueError(
                    f"Function '{fn}' returned shape {y.shape}, expected {x.shape} (or broadcastable).")

        unary_blocks.append(y)

    arr = np.concatenate(unary_blocks, axis=1)
    m = arr.shape[1]

    # Build theta with polynomial interactions up to poly_order
    theta_blocks = [np.ones((n_samples, 1), dtype=arr.dtype)]  # constant
    theta_blocks.append(arr)  # unary terms
    all_feature_names = ["1"] + [f"{fn}({feat})" for fn in func_names for feat in feature_names_fixed]
    if len(all_feature_names) - 1 != m:
        all_feature_names = ["1"] + [f"u{i}" for i in range(m)]

    # Build interaction terms for poly_order >= 2
    if poly_order >= 2:
        from itertools import combinations_with_replacement
        interaction_blocks = []
        for order in range(2, poly_order + 1):
            indices_list = list(combinations_with_replacement(range(m), order))
            if len(indices_list) > 0:
                # Compute products for this order
                order_terms = np.ones((n_samples, len(indices_list)), dtype=arr.dtype)
                for col_idx, idx_tuple in enumerate(indices_list):
                    for idx in idx_tuple:
                        order_terms[:, col_idx] *= arr[:, idx]
                interaction_blocks.append(order_terms)
                # Add feature names for this order
                if poly_order == 2:
                    all_feature_names.extend([f"{all_feature_names[i+1]}{all_feature_names[j+1]}"
                                              for i, j in indices_list])
                else:
                    all_feature_names.extend([f"{'*'.join(str(all_feature_names[idx+1]) for idx in idx_tuple)}"
                                              for idx_tuple in indices_list])

        if interaction_blocks:
            theta_blocks.extend(interaction_blocks)

    theta = np.concatenate(theta_blocks, axis=1)

    if not return_names:
        return {"theta": theta}

    return {"theta": theta, "library_feature_names": all_feature_names}


def find_closest_match(df, f10_target, kp_target):
    df['f10_diff'] = np.abs(df['f10'] - f10_target)
    df['kp_diff'] = np.abs(df['kp'] - kp_target)

    closest_row = df.loc[(df['f10_diff'] + df['kp_diff']).idxmin()]
    closest_row = closest_row.drop(columns=['f10_diff', 'kp_diff'])
    df = df.drop(columns=['f10_diff', 'kp_diff'])

    return closest_row


def get_propagator_datetimes(datetimes_ds):
    if isinstance(datetimes_ds, np.ndarray) and np.issubdtype(datetimes_ds.dtype, np.number):
        datetimes_ds = matlab_datenum_to_py_datetime(datetimes_ds)
        return datetimes_ds
    elif np.isscalar(datetimes_ds) and isinstance(datetimes_ds, (float, int)):
        datetimes_ds = matlab_datenum_to_py_datetime([datetimes_ds])
        return datetimes_ds


def build_dyn_sindy_inputs(z1_k_denormalized, all_drivers, X_reg_norm_dict_sindy, f10_idx, model_params,
                           normalization_method, input_features, time):
    kp_idx = f10_idx + 3
    f10a_idx = kp_idx + 1
    drivers = np.copy(all_drivers[:(kp_idx + 1), time])
    ar_drivers = np.copy(all_drivers[(kp_idx + 1):, time])
    X_k_for_sindy = np.concatenate(
        [z1_k_denormalized.reshape((-1, 1)), drivers[f10_idx:].reshape((-1, 1))])

    library_dict = create_library_functions(
        np.copy(X_k_for_sindy.T), model_params['functions'], input_features)
    theta_k = library_dict['theta'].T
    X_k = np.concatenate([np.ones((1, 1)), theta_k[11:, 0].reshape(
        (-1, 1)), ar_drivers.reshape((-1, 1)), z1_k_denormalized], axis=0)

    X_k_norm = normalize_with_dict(
        X_k[1:], X_reg_norm_dict_sindy, method=normalization_method)
    X_k_norm = np.concatenate([X_k[0, 0].reshape((1, 1)), drivers[1:f10_idx].reshape(
        (-1, 1)), X_k_norm], axis=0)  # adding the constant column to the inputs

    return X_k_norm


def build_dyn_sindy_inputs_optimized(
    z1_k_denormalized,
    all_drivers,
    X_reg_norm_dict_sindy,
    f10_idx,
    model_params,
    normalization_method,
    input_features,
    time,
    kp_idx=None,
):
    if kp_idx is None:
        kp_idx = f10_idx + 3

    dcol = all_drivers[:, time]
    drivers = dcol[:kp_idx + 1]
    ar_drivers = dcol[kp_idx + 1:]

    zcol = z1_k_denormalized.reshape(-1, 1)
    drv_for_sindy = drivers[1:].reshape(-1, 1)
    # Here I add the part to consider also time inputs

    n_z = zcol.shape[0]
    n_ud = drv_for_sindy.shape[0]  # t1 to t4, f10, [kpstd], kp
    n_t = 4
    n_d = n_ud - n_t
    X_k_for_sindy = np.empty((n_z + n_ud, 1), dtype=zcol.dtype)
    X_k_for_sindy[:n_z, 0] = zcol[:, 0]
    X_k_for_sindy[n_z:, 0] = drv_for_sindy[:, 0]

    library_dict = create_library_functions_optimized(
        X_k_for_sindy.T, model_params["functions"], input_features, return_names=False,
        poly_order=model_params.get("poly_order", 2)
    )
    theta = library_dict["theta"]

    theta_row = theta[0] if theta.ndim == 2 else theta
    # Detect whether the library includes standalone linear z-state terms.
    # Full quadratic poly in (n_z + n_ud) variables has C(n+2,2) = (n+1)(n+2)/2 terms.
    # Old library: includes z linear -> theta.size == full poly count
    # New library: z states only in quadratics -> theta.size < full poly count
    n_total = n_z + n_ud
    full_poly_size = (n_total + 1) * (n_total + 2) // 2
    if theta_row.size == full_poly_size:
        # Old-style library: [1, z_lin(n_z), t_lin(n_t), driver_lin, quadratics]
        theta_start = 1 + n_z + n_t
    else:
        # New-style library: [1, t_lin(n_t), driver_lin, quadratics] (no standalone z linear)
        theta_start = 1 + n_t
    theta_d_nl = np.asarray(theta_row[theta_start:], dtype=X_k_for_sindy.dtype).reshape(-1, 1)

    n_theta_d_nl = theta_d_nl.shape[0]

    # Only the theta features are used downstream (sindy_loop_optimized takes
    # sindy_inputs_norm[:ar_drivers_start] which covers [1, t_vars, theta_norm]).
    # AR drivers and z are handled separately via B_ar_drivers / B_ar_state, so
    # we must NOT include them here — they would cause a shape mismatch against
    # X_reg_norm_dict (built at training time with a possibly different n_ar).
    theta_norm_dict = {
        'x_mean': X_reg_norm_dict_sindy['x_mean'][:n_theta_d_nl],
        'x_std':  X_reg_norm_dict_sindy['x_std'][:n_theta_d_nl],
    }
    theta_d_nl_norm = normalize_with_dict(
        theta_d_nl, theta_norm_dict, method=normalization_method)

    t_variables = drivers[1:f10_idx].reshape(-1, 1)

    out = np.empty(
        (1 + t_variables.shape[0] + n_theta_d_nl, 1), dtype=X_k_for_sindy.dtype)
    j = 0
    out[j, 0] = 1.0
    j += 1

    if t_variables.size:
        out[j:j+t_variables.shape[0], 0] = t_variables[:, 0]
        j += t_variables.shape[0]

    out[j:, 0] = theta_d_nl_norm[:, 0]

    return out


def sindy_loop(u_prop_interpolated, X_reg_norm_dict, Y_reg_norm_dict, A_d2, B_ar_drivers_lst, B_ar_state_lst, B_nonlinear,
               stored_frcst, ar_drivers_start, t0, tf, ar_lags, time_conversion_factor, f10_idx, normalization_method, model_params, input_features):
    kp_idx = f10_idx + 3
    delta_idx = 5
    X_ar_drivers_norm_dict = {'x_mean': X_reg_norm_dict['x_mean'][(ar_drivers_start - delta_idx):(ar_drivers_start + 2*max(ar_lags) - delta_idx), :].reshape((-1, 1)),
                              'x_std': X_reg_norm_dict['x_std'][(ar_drivers_start - delta_idx):(ar_drivers_start + 2*max(ar_lags) - delta_idx), :].reshape((-1, 1))}

    for k in range(int(t0), int(tf)):

        # Getting the different input variables blocks
        # stored_frcst goes into the future as the index advances through columns
        z_ar_np_interpolated = np.fliplr(
            stored_frcst[:, -int((max(ar_lags) + 1) * time_conversion_factor):-int(time_conversion_factor)])

        # Latent space componbents need to be interpolated only once before the beginning of the loop, as well
        # as with AR drtivers. Then, SINDY variables can be constructed within the loop each time.
        # AR drivers can be interpolated once, and the picked at each iteration.
        # SINDY inputs need to be built from scratch and then be normalized.
        # Calculation of drivers and AR dtrivers variables.
        L = len(ar_lags)
        ar_drivers_prop_interpolated = u_prop_interpolated[(
            # This goes into the past
            kp_idx + 1): ((kp_idx + 1) + 2 * L), k].reshape((-1, 1))

        ar_drivers_prop_interpolated = normalize_with_dict(np.copy(ar_drivers_prop_interpolated),
                                                           X_ar_drivers_norm_dict, method=normalization_method)
        # ar_drivers_prop_interpolated = ar_drivers_prop_interpolated * ar_drivers_weights

        ar_drivers_prop_interpolated = ar_drivers_prop_interpolated.reshape(
            L, 2).T

        z_k_norm = np.copy(stored_frcst[:, -1]).reshape((-1, 1))

        z_k_denormalized = denormalize(
            np.copy(z_k_norm), norm_dict=Y_reg_norm_dict, method=normalization_method)
        sindy_inputs_norm = build_dyn_sindy_inputs(z_k_denormalized, u_prop_interpolated,
                                                   X_reg_norm_dict, f10_idx, model_params,
                                                   normalization_method, input_features, time=k)
        # SINDY variables ()
        x_nonlinear = np.copy(
            sindy_inputs_norm[:ar_drivers_start].reshape((-1, 1)))

        # Forecasting next step
        ar_state_contribution = np.sum(
            [B_ar_state_lst[i] @ z_ar_np_interpolated[:, i].reshape((-1, 1)) for i in range(len(ar_lags))], axis=0)
        ar_drivers_contribution = np.sum(
            [B_ar_drivers_lst[i] @ ar_drivers_prop_interpolated[:, i].reshape((-1, 1)) for i in range(len(ar_lags))], axis=0)

        z_frcst = A_d2 @ z_k_norm + ar_state_contribution + \
            ar_drivers_contribution + B_nonlinear @ x_nonlinear
        # if max(z_frcst) >= 10.:
        #     stored_frcst = np.concatenate([stored_frcst, z_frcst], axis = 1)
        #     return stored_frcst
        stored_frcst = np.concatenate([stored_frcst, z_frcst], axis=1)

    return stored_frcst


def sindy_loop_optimized(
    u_prop_interpolated,
    X_reg_norm_dict,
    Y_reg_norm_dict,
    A_d2,
    B_ar_drivers_lst,
    B_ar_state_lst,
    B_nonlinear,
    stored_frcst,
    ar_drivers_start,
    t0,
    tf,
    ar_lags,
    time_conversion_factor,
    f10_idx,
    normalization_method,
    model_params,
    input_features,
    kp_idx=None,
):
    if kp_idx is None:
        kp_idx = f10_idx + 3
    delta_idx = 5

    ar_lags = np.asarray(ar_lags, dtype=int)
    L = ar_lags.size
    tc = int(time_conversion_factor)
    n_state, n_init = stored_frcst.shape
    n_steps = int(tf) - int(t0)
    if n_steps <= 0:
        return stored_frcst

    sl0 = ar_drivers_start - delta_idx
    sl1 = ar_drivers_start + 3 * int(ar_lags.max()) - delta_idx
    X_ar_drivers_norm_dict = {
        "x_mean": X_reg_norm_dict["x_mean"][sl0:sl1, :].reshape((-1, 1)),
        "x_std":  X_reg_norm_dict["x_std"][sl0:sl1, :].reshape((-1, 1)),
    }

    B_state = np.stack(B_ar_state_lst, axis=0)

    B_drv = np.stack(B_ar_drivers_lst, axis=0)

    out = np.empty((n_state, n_init + n_steps), dtype=stored_frcst.dtype)
    out[:, :n_init] = stored_frcst

    lag_offsets = ar_lags * tc

    cur = n_init - 1

    for k in range(int(t0), int(tf)):
        z_k_norm = out[:, cur:cur+1]

        cols = cur - lag_offsets
        z_ar = out[:, cols]

        ar_drv = u_prop_interpolated[(
            kp_idx + 1):(kp_idx + 1 + 3 * L), k].reshape((-1, 1))
        ar_drv = normalize_with_dict(
            ar_drv, X_ar_drivers_norm_dict, method=normalization_method)
        ar_drv = ar_drv.reshape(L, 3).T

        z_k_denorm = denormalize(
            z_k_norm, norm_dict=Y_reg_norm_dict, method=normalization_method)
        sindy_inputs_norm = build_dyn_sindy_inputs_optimized(
            z_k_denorm,
            u_prop_interpolated,
            X_reg_norm_dict,
            f10_idx,
            model_params,
            normalization_method,
            input_features,
            time=k,
            kp_idx=kp_idx,
        )

        x_nonlinear = sindy_inputs_norm[:ar_drivers_start].reshape((-1, 1))

        ar_state_contribution = np.einsum(
            "lij,jl->i", B_state, z_ar).reshape((-1, 1))

        ar_drivers_contribution = np.einsum(
            "lik,kl->i", B_drv, ar_drv).reshape((-1, 1))

        z_next = (A_d2 @ z_k_norm
                  + ar_state_contribution
                  + ar_drivers_contribution
                  + (B_nonlinear @ x_nonlinear))

        out[:, cur + 1:cur + 2] = z_next
        cur += 1

    return out


def dmd_loop(u_ds_interpolated, A_d2, B_nonlinear, stored_frcst, t0, tf, X_reg_norm_dict,
             Y_reg_norm_dict, f10_idx, kp_idx, normalization_method, n_components):
    x_nonlinear = np.concatenate([u_ds_interpolated[f10_idx:, :],
                                  (u_ds_interpolated[f10_idx, :] *
                                   u_ds_interpolated[kp_idx, :]).reshape((1, -1)),
                                  (u_ds_interpolated[kp_idx, :] * u_ds_interpolated[kp_idx, :]).reshape((1, -1))], axis=0)
    for k in range(int(t0), int(tf)):
        # Getting the different input variables blocks
        z_k = stored_frcst[:, -1].reshape((-1, 1))
        x_nonlinear_k = np.concatenate(
            [np.copy(x_nonlinear[:, k]).reshape((-1, 1)), z_k.reshape((-1, 1))], axis=0)

        x_nonlinear_norm = normalize_with_dict(
            x_nonlinear_k, X_reg_norm_dict, normalization_method)
        x_nonlinear_norm = np.concatenate([np.copy(
            u_ds_interpolated[1:f10_idx, k]).reshape((-1, 1)), x_nonlinear_norm], axis=0)

        z_frcst = A_d2 @ z_k.reshape((-1, 1)) + \
            B_nonlinear @ x_nonlinear_norm[:-n_components].reshape((-1, 1))

        stored_frcst = np.concatenate([stored_frcst, z_frcst], axis=1)

    return stored_frcst


def propagate(start_date, stop_date, model_object, ar_lags, T1, T2, time_conversion_factor, n_components, f10_idx=5,
              model_name='sindy-c', B_sindy=None, drivers='tiegcm2', kp_idx=None):
    if kp_idx is None:
        kp_idx = f10_idx + 3
    if model_name == 'sindy-c':
        B = model_object['B_sindy']
        X_reg_norm_dict = {'x_mean': np.copy(model_object['X_sindy_norm_dict'].x_mean).reshape(
            (-1, 1)), 'x_std': np.copy(model_object['X_sindy_norm_dict'].x_std.reshape((-1, 1)))}
        Y_reg_norm_dict = {'x_mean': np.copy(model_object['Y_sindy_norm_dict'].x_mean).reshape(
            (-1, 1)), 'x_std': np.copy(model_object['Y_sindy_norm_dict'].x_std.reshape((-1, 1)))}
    elif model_name == 'dmd-c':
        B = model_object['B_dmd']
        u_ar_ds = model_object['u_ar_ds']
        X_reg_norm_dict = {'x_mean': np.copy(model_object['X_dmd_ds_norm_dict'].x_mean).reshape(
            (-1, 1)), 'x_std': np.copy(model_object['X_dmd_ds_norm_dict'].x_std.reshape((-1, 1)))}
        Y_reg_norm_dict = {'x_mean': np.copy(model_object['Y_dmd_ds_norm_dict'].x_mean).reshape(
            (-1, 1)), 'x_std': np.copy(model_object['Y_dmd_ds_norm_dict'].x_std.reshape((-1, 1)))}
    elif model_name == 'nl-ddm':
        B = model_object['B_nl_ddm']
        u_ar_ds = model_object['X_nl_ddm_ds_norm']
        Y_reg_norm_dict = {'x_mean': np.copy(model_object['Y_nl_ddm_ds_norm_dict'].x_mean).reshape(
            (-1, 1)), 'x_std': np.copy(model_object['Y_nl_ddm_ds_norm_dict'].x_std.reshape((-1, 1)))}
    else:
        raise ValueError("Invalid model name")
    if drivers == 'tiegcm2':
        u_ar_ds = model_object['u_ar_ds_tiegcm2']
        datetimes_ds = model_object['datetimes_ds_tiegcm2']
    else:
        u_ar_ds = model_object['u_ar_ds']
        datetimes_ds = model_object['datetimes_ds']
    normalization_method = model_object['normalization_method']
    model_params_mat = model_object['model_params']

    model_params = {'normalization_method': model_params_mat.normalization_method, 'shift': model_params_mat.shift, 'dof': model_params_mat.dof,
                    'functions': mat_to_lambda_dict(model_params_mat.functions), 'model': model_params_mat.model, 'control': model_params_mat.control,
                    'poly_order': int(getattr(model_params_mat, 'poly_order', 2))}
    input_features = model_object['input_features']

    # z_ds = model_object["z_ds"]
    ic_array_data = model_object['ic_array']
    if hasattr(ic_array_data, 'ic_array'):
        ic_array_data = ic_array_data.ic_array
    ic_df = pd.DataFrame(ic_array_data, columns=[
                         'f10', 'kp'] + [f'z_{str(k+1).zfill(2)}' for k in range(n_components)])
    datetime_series = pd.date_range(
        start=start_date, end=stop_date, freq=f'{int(T2)}s')

    datetimes_ds = get_propagator_datetimes(datetimes_ds)
    propagation_idx = np.array(
        [(start_date <= date_time <= stop_date) for date_time in datetimes_ds])

    n_hours = (stop_date - start_date).total_seconds()/3600
    t0 = 0 * time_conversion_factor
    tf = np.floor((n_hours)) * time_conversion_factor
    # print(z_ds.shape, propagation_idx.shape)
    # z_ds_prop = np.copy(z_ds[:, propagation_idx])
    u_ar_ds_prop = np.copy(u_ar_ds[:, propagation_idx])
    u_ds_prop = np.copy(u_ar_ds_prop[: (kp_idx + 1), :])

    if model_name == 'sindy-c':
        if B_sindy is not None:
            B = B_sindy
        A_d = np.copy(B[:, -n_components:])
        B_d = np.copy(B[:, :-n_components])

        A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)

        ar_drivers_start = B.shape[1] - n_components - \
            max(ar_lags) * n_components - 3 * max(ar_lags)
        ar_state_start = B.shape[1] - n_components - len(ar_lags)*n_components

        B_nonlinear = np.copy(B_d2[:, :ar_drivers_start])
        B_ar_drivers_lst = [np.copy(B_d2[:, slice(ar_drivers_start + k * 3, ar_drivers_start + (k + 1) * 3)])
                            # The more idx advances, the more goes into the past
                            for k in range(len(ar_lags))]
        B_ar_state_lst = [np.copy(B_d2[:, slice(ar_state_start + k * n_components, ar_state_start + (k + 1) * n_components)])
                          # The more idx advances, the more goes into the past
                          for k in range(len(ar_lags))]

        ti_lst = ar_lags
        # z_i_lst = [np.copy(z_ds_prop[:, k].reshape((-1, 1))) for k in range(max(ti_lst) + 1)]#this is going into the future as k increases
        z_i_lst = [find_closest_match(ic_df, u_ds_prop[f10_idx, k], u_ds_prop[kp_idx, k]).values[2:2+n_components].reshape(
            (-1, 1)) for k in range(max(ti_lst) + 1)]
        # print(z_i_lst)
        # print(z_i_lst2)
        # aaa
        x_drivers_prop = np.copy(u_ar_ds_prop)
        x_drivers_prop_interpolated = interpolate_matrix_rows(
            x_drivers_prop, time_conversion_factor)
        # This is going into the future as the column index increases
        stored_frcst = np.concatenate(z_i_lst, axis=1)
        stored_frcst = normalize_with_dict(
            stored_frcst, Y_reg_norm_dict, method=normalization_method)
        stored_frcst = interpolate_matrix_rows(
            stored_frcst, time_conversion_factor)

        stored_frcst = sindy_loop_optimized(x_drivers_prop_interpolated, X_reg_norm_dict, Y_reg_norm_dict, A_d2, B_ar_drivers_lst,
                                            B_ar_state_lst, B_nonlinear, stored_frcst, ar_drivers_start, t0, tf, ar_lags,
                                            time_conversion_factor, f10_idx, normalization_method, model_params, input_features,
                                            kp_idx=kp_idx)[:, int((max(ar_lags) - 2) * time_conversion_factor):-2]

        # np.copy(stored_frcst) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        z_frcst_np = denormalize(
            stored_frcst, norm_dict=Y_reg_norm_dict, method=normalization_method)
        z_frcst_np_hourly = np.copy(stored_frcst[:, ::int(
            time_conversion_factor)]) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        return z_frcst_np[:, :datetime_series.shape[0]], z_frcst_np_hourly, x_drivers_prop[:(kp_idx+1), :], datetime_series

    elif model_name == 'dmd-c':
        A_d = np.copy(B[:, -n_components:])
        B_d = np.copy(B[:, :-n_components])

        A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)

        ti_lst = [ar_lags[0]]
        # z_i_lst = [np.copy(z_ds_prop[:, k].reshape((-1, 1))) for k in ti_lst]#this is going into the future as k increases
        z_i_lst = [find_closest_match(
            ic_df, u_ds_prop[f10_idx, k], u_ds_prop[kp_idx, k]).values[2:2+n_components].reshape((-1, 1)) for k in ti_lst]
        # print(z_i_lst)
        # print(z_i_lst2)
        # aaa
        u_ds_prop_interpolated = interpolate_matrix_rows(
            u_ds_prop, time_conversion_factor)
        stored_frcst = np.array(z_i_lst).reshape((-1, 1))
        stored_frcst = normalize_with_dict(
            stored_frcst, Y_reg_norm_dict, method=normalization_method)
        stored_frcst = dmd_loop(u_ds_prop_interpolated, A_d2, B_d2, stored_frcst, t0, tf, X_reg_norm_dict,
                                Y_reg_norm_dict, f10_idx, kp_idx, normalization_method, n_components)

        z_frcst_np = np.copy(stored_frcst) * \
            Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        z_frcst_np_hourly = np.copy(stored_frcst[:, ::int(
            time_conversion_factor)]) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        return z_frcst_np[:, :datetime_series.shape[0]], z_frcst_np_hourly, u_ds_prop, datetime_series

    elif model_name == 'nl-ddm':
        A_d = np.copy(B[:, -n_components:])
        B_d = np.copy(B[:, :-n_components])

        A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)

        ti_lst = [ar_lags[0]]
        z_i_lst = [np.copy(u_ar_ds_prop[-n_components:, k].reshape((-1, 1)))
                   # this is going into the future as k increases
                   for k in ti_lst]
        x_prop_norm_interpolated = interpolate_matrix_rows(
            u_ar_ds_prop, time_conversion_factor)

        stored_frcst = np.array(z_i_lst).reshape((-1, 1))
        stored_frcst = dmd_loop(x_prop_norm_interpolated,
                                A_d2, B_d2, stored_frcst, t0, tf, n_components)

        # np.copy(stored_frcst) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        z_frcst_np = denormalize(
            stored_frcst, norm_dict=Y_reg_norm_dict, method=normalization_method)
        z_frcst_np_hourly = np.copy(stored_frcst[:, ::int(
            time_conversion_factor)]) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        return z_frcst_np[:, :datetime_series.shape[0]], z_frcst_np_hourly, x_prop_norm_interpolated[:(kp_idx+1), int((max(ar_lags) - 2) * time_conversion_factor):-2], datetime_series


def interpolate(output_datetimes: np.array, timestamps: np.array, lla: np.array, model_object: object, z_dict: dict) -> tuple[float, float]:
    warnings.filterwarnings("ignore")

    lt_axis = np.linspace(0, 23.66666667, 72)
    lat_axis = np.linspace(-87.5, 87.5, 36)
    alt_axis = np.linspace(100, 980, 45)  # km

    U0 = model_object['U'].reshape((72, 36, 45, 10), order='C')
    mu0 = model_object['mu'].reshape((72, 36, 45), order='C')

    timestamps = pd.to_datetime(timestamps)

    original_time_variations = (
        output_datetimes - output_datetimes[0]).total_seconds()
    time_variations = (timestamps - output_datetimes[0]).total_seconds()
    # print(np.max(T), t.min(), t.max())

    # (LAT, LT, ALT) ---> (LT, LAT, ALT)
    points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])

    # Prepare interpolation variables
    # Define grid for interpolation

    lt = np.linspace(lt_axis[0], lt_axis[-1], U0.shape[0])
    lat = np.linspace(lat_axis[0], lat_axis[-1], U0.shape[1])
    alt = np.linspace(alt_axis[0], alt_axis[-1], U0.shape[2])

    my_interpolating_U0 = rgi(
        (lt, lat, alt), U0, bounds_error=False, fill_value=None)
    my_interpolating_mu0 = rgi(
        (lt, lat, alt), mu0, bounds_error=False, fill_value=None)

    # Perform interpolation
    U0_interp = my_interpolating_U0(points)
    mu0_interp = my_interpolating_mu0(points)

    sindy_interpolators_lst = []
    for sindy_model in list(z_dict.keys()):
        sindy_interpolators_lst.append(interp1d(original_time_variations, z_dict[sindy_model],
                                                kind='linear', axis=1, fill_value="extrapolate"))

    z_uncertainties_values_lst = [10 ** (np.sum(U0_interp * interpolator(time_variations).T, axis=1).reshape((-1, 1)) +
                                         mu0_interp.T.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
    interpolated_models = np.stack(z_uncertainties_values_lst).squeeze(-1).T

    density_std = np.nanstd(interpolated_models, axis=1)
    density = np.nanmean(interpolated_models, axis=1)

    return density, density_std

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SN additions:


def mape_density(rho_hat, rho, epsilon=1e-12):
    """
    Compute MAPE along a trajectory (1D time series).

    Parameters
    ----------
    rho_hat : np.ndarray
        Forecast/assimilated density, shape (m,)
    rho : np.ndarray
        Measured density, shape (m,)
    epsilon : float
        Small value to avoid division by zero in rho

    Returns
    -------
    mape : float
        Mean Absolute Percentage Error in percent
    """
    # Avoid division by zero
    rho_safe = np.where(rho == 0, epsilon, rho)

    # Number of time steps
    m = rho.shape[0]

    # Sum over time
    sum_error = np.sum(np.abs(rho_hat - rho) / rho_safe)

    # MAPE. Divide by m, matching the definition given in the manuscript. This used to
    # divide by m - 1, which inflated every reported value by m / (m - 1); at the ~17,000
    # samples of a 12-day 60 s track that is 0.006% relative, so no published number moves
    # visibly, but the archived code did not implement the published equation.
    mape = 100 * sum_error / m

    return mape


class MeasurementModel:
    def __init__(self, H_U0: np.ndarray, H_mu0: np.ndarray):
        """
        Measurement model for density assimilation.

        Parameters
        ----------
        H_U0 : np.ndarray
            Measurement basis matrix.
        H_mu0 : np.ndarray
            Measurement bias vector.
        """
        self.H_U0 = H_U0
        self.H_mu0 = H_mu0

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Nonlinear measurement function in linear density space.
        h(x) = 10^(H_U0 x + H_mu0)
        """
        return 10 ** (self.H_U0 @ x + self.H_mu0)

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of nonlinear measurement h(x) w.r.t state x.
        """
        h_x = self.h(x)
        return np.log(10) * np.diag(h_x.flatten()) @ self.H_U0

    def h_log(self, x: np.ndarray) -> np.ndarray:
        """
        Linear measurement function in log10 space.
        h_log(x) = H_U0 x + H_mu0
        """
        return self.H_U0 @ x + self.H_mu0

    def H_jacobian_log(self) -> np.ndarray:
        """
        Jacobian of linear log-space measurement function.
        Constant matrix H_U0.
        """
        return self.H_U0


class RopeMeasurements:
    COLUMNS = ['date', 'time', 'GPS', 'alt', 'lon',
               'lat', 'lst', 'arglat', 'accelerometer_density']

    def __init__(self, dataset_name, msis_version="2.1", version="v01"):
        self.dataset_name = dataset_name.lower()
        self.msis_version = msis_version
        self.version = version
        self.data_cache = {}  # monthly DataFrame cache
        self.cached_df = pd.DataFrame()
        self.base_path = Path(__file__).parent.parent / "hrd_20250608" / "data"

    def _get_folder(self):
        folder_map = {
            'champ': 'CHAMP_data',
            'grace': 'GRACE_data',
            'goce': 'GOCE_data',
            'swarm_a': 'Swarm_data',
            'swarm_b': 'Swarm_data',
            'swarm_c': 'Swarm_data',
            'grace-fo': 'GRACE-FO_data'
        }
        return os.path.join(self.base_path, self.version, folder_map[self.dataset_name])

    def _get_txt_filename(self, year, month):
        prefix_map = {
            "champ": "CH_DNS_ACC",
            "grace": "GA_DNS_ACC",
            "goce": "GO_DNS_ACC",
            "swarm_a": "SA_DNS_ACC",
            "swarm_b": "SC_DNS_ACC",
            "swarm_c": "SC_DNS_ACC"
        }
        return f"{prefix_map[self.dataset_name]}_{year}_{month:02d}_v01.txt"

    def _get_zip_filename(self, year, month):
        """
        Return a glob pattern for v02 ZIP files.
        Matches v02, v02b, v02c, ...
        """
        prefix_map = { "champ": "CH_DNS_ACC", 
                      "grace": "GA_DNS_ACC", 
                      "goce": "GO_DNS_ACC", 
                      "swarm_a": "SA_DNS_POD", 
                      "swarm_b": "SB_DNS_POD", 
                      "swarm_c": "SC_DNS_POD", 
                      "grace-fo": "GC_DNS_ACC"
                        }

        sep = "-" if self.dataset_name == "champ" else "_"

        return (
            f"{prefix_map[self.dataset_name]}_"
            f"{year}{sep}{month:02d}_v02*.zip"
        )

    def _parse_txt(self, raw_text):
        fixed_lines = []
        current_date = None
        for line in raw_text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
                current_date = parts[0]
                fixed_lines.append(" ".join(parts))
            else:
                if current_date:
                    fixed_lines.append(current_date + " " + " ".join(parts))

        df = pd.read_csv(
            StringIO("\n".join(fixed_lines)),
            sep=r"\s+",
            names=cols,
            engine="python"   
        )
        # tz-naive datetime (old behavior)
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str),
                                        errors='coerce')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date', 'time', 'GPS'], inplace=True)
        print(
            f"[{self.dataset_name}] Parsed TXT: {df.index.min()} -> {df.index.max()}, rows={len(df)}")
        return df.sort_index()

    def _parse_zip_txt(self, raw_text, year=None, month=None):
        """
        Parse ZIP-extracted text for all satellites. Handles GOCE separately.
        year/month only needed for consistency, but not actually used for GOCE
        because the datetime is fully present in the file.
        """
        fixed_lines = []
        for line in raw_text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            fixed_lines.append(line.strip())

        if self.dataset_name == "goce":
            cols = [
                'time_raw', 'time_system', 'alt', 'lon', 'lat', 'lst', 'arglat',
                'goce_density', 'dens_mean', 'flag_dens', 'flag_dens_mean', 'flag_thruster'
            ]
            df = pd.read_csv(
                StringIO("\n".join(fixed_lines)),
                sep=r"\s+",
                names=cols,
                engine="python"  
            )

            if year is None or month is None:
                raise ValueError("GOCE parsing requires year and month")

            start_date = pd.Timestamp(year=year, month=month, day=1)
            df['datetime'] = pd.to_datetime(
                start_date.strftime("%Y-%m-%d") + " " +
                df['time_raw'].astype(str),
                errors='coerce'
            )

            df['date_offset'] = (
                df['datetime'].diff().dt.total_seconds() < 0).cumsum()
            df['datetime'] = df['datetime'] + \
                pd.to_timedelta(df['date_offset'], unit="D")

            df.set_index('datetime', inplace=True)
            df.drop(columns=['time_raw', 'time_system',
                    'date_offset'], inplace=True)

        else:
            col_map = {
                'champ': ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                          'champ_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'],
                'grace': ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                          'grace_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'],
                'swarm_a': ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                            'swarm_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'],
                'swarm_b': ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                            'swarm_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'],
                'swarm_c': ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                            'swarm_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'],
                'grace-fo': ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                             'grace_fo_density', 'dens_mean', 'flag_dens', 'flag_dens_mean']
            }
            cols = col_map[self.dataset_name]
            df = pd.read_csv(
                StringIO("\n".join(fixed_lines)),
                sep=r"\s+",
                names=cols,
                engine="python"   
            )
            df['datetime'] = pd.to_datetime(
                df['date'].astype(str) + ' ' + df['time'].astype(str),
                errors='coerce'
            )
            df.set_index('datetime', inplace=True)
            df.drop(columns=['date', 'time', 'GPS'], inplace=True)

        # Sort and debug info
        df = df.sort_index()
        print(
            f"[{self.dataset_name}] Parsed ZIP: {df.index.min()} -> {df.index.max()}, rows={len(df)}")

        return df

    # --- rest of class remains unchanged ---

    def _load_month_txt(self, year, month):
        folder = self._get_folder()
        file = os.path.join(folder, self._get_txt_filename(year, month))
        if os.path.isfile(file):
            try:
                with open(file, 'r') as f:
                    raw = f.read()
                df = self._parse_txt(raw)
                df_month = df[(df.index.year == year) &
                              (df.index.month == month)]
                if not df_month.empty:
                    return df_month.sort_index()
            except Exception as e:
                print(f"[WARN] Failed to load {file}: {e}")
        else:
            print(f"[INFO] File not found: {file}")
        return None

    def _load_month_zip(self, year, month):
        """
        Load v02 ZIP file (handles v02, v02b, v02c, ...).
        Always selects the latest available version.
        """
        pattern = self._get_zip_filename(year, month)
        folder = self._get_folder()

        matches = sorted(
            glob.glob(os.path.join(folder, pattern))
        )

        if not matches:
            return None

        # Use the newest re-release (v02c > v02b > v02)
        path = matches[-1]

        try:
            with zipfile.ZipFile(path, 'r') as z:
                name = z.namelist()[0]
                raw = z.read(name).decode("utf-8")

            return self._parse_zip_txt(raw, year=year, month=month)

        except Exception as e:
            print(f"[ZIP ERROR] {os.path.basename(path)}: {e}")
            return None


    def _load_month_remote(self, year, month):
        filename = self._get_zip_filename(year, month)
        folder_map = {
            'champ': 'CHAMP_data',
            'grace': 'GRACE_data',
            'goce': 'GOCE_data',
            'swarm_a': 'Swarm_data',
            'swarm_b': 'Swarm_data',
            'swarm_c': 'Swarm_data',
            'grace-fo': 'GRACE-FO_data'
        }
        try:
            ftp = FTP("thermosphere.tudelft.nl")
            ftp.login()
            ftp.cwd(f"/version_02/{folder_map[self.dataset_name]}")
            buffer = BytesIO()
            ftp.retrbinary(f"RETR {filename}", buffer.write)
            ftp.quit()
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as z:
                name = z.namelist()[0]
                raw = z.read(name).decode("utf-8")
            return self._parse_zip_txt(raw)
        except Exception as e:
            print(f"[FTP ERROR] {filename}: {e}")
            return None

    def _load_month(self, year, month, prefer_remote=False):
        key = f"{year}-{month:02d}"
        if key in self.data_cache:
            return self.data_cache[key]

        df = None
        if self.version == "v01":
            df = self._load_month_txt(year, month)
        elif self.version == "v02":
            df = self._load_month_zip(year, month)
            if df is None and prefer_remote:
                df = self._load_month_remote(year, month)

        if df is not None and not df.empty:
            self.data_cache[key] = df
            print(f"[LOADED] {key} with {len(df)} rows")
        else:
            print(f"[NO DATA] {key}")

        return df
    
    def _safe_msis(self, timestamps, lon, lat, alt_km):
        """
        Safely evaluate MSIS. Returns NaNs if MSIS fails.
        """
        n = len(timestamps)

        try:
            out = msis.calculate(
                timestamps,
                lon,
                lat,
                alt_km,
                geomagnetic_activity=-1,
                version=self.msis_version
            )

            # Sanity check (MSIS sometimes returns garbage silently)
            if out is None or out.shape[0] != n:
                raise ValueError("Invalid MSIS output shape")

            return out[:, 0]

        except Exception as e:
            print(f"[MSIS WARNING] MSIS failed ({type(e).__name__}): {e}")
            return np.full(n, np.nan)

    def _build_meas_df(self, df, use_msis=True):
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'accelerometer_density', 'msis_rho',
                'latitude', 'local_time', 'longitude', 'altitude'
            ])

        if self.version == "v01":
            density_col = 'accelerometer_density'
        else:
            density_col_map = {
                'champ': 'champ_density',
                'grace': 'grace_density',
                'goce': 'goce_density',
                'swarm_a': 'swarm_density',
                'swarm_b': 'swarm_density',
                'swarm_c': 'swarm_density',
                'grace-fo': 'grace_fo_density'
            }
            density_col = density_col_map.get(self.dataset_name)
            if density_col not in df.columns:
                raise KeyError(
                    f"Expected density column '{density_col}' not found")

        timestamps = df.index.values

        if use_msis:
            msis_rho = self._safe_msis(
                timestamps,
                df['lon'].values,
                df['lat'].values,
                df['alt'].values / 1000.0
            )
        else:
            msis_rho = np.full(len(df), np.nan)

        return pd.DataFrame({
            'timestamp': timestamps,
            'accelerometer_density': df[density_col].values,
            'msis_rho': msis_rho,
            'latitude': df['lat'].values,
            'local_time': df['lst'].values,
            'longitude': df['lon'].values,
            'altitude': df['alt'].values / 1000.0
        })

    def build_meas_data(self, start_date, end_date, prefer_remote=False, use_msis=True):
        """
        Build a combined measurement DataFrame between start_date and end_date.
        Handles tz-naive timestamps to avoid KeyError during slicing.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Determine which months to load
        months = pd.date_range(start.replace(
            day=1), end.replace(day=1), freq='MS')
        dfs = [self._load_month(dt.year, dt.month, prefer_remote)
               for dt in months]
        dfs_valid = [df for df in dfs if df is not None and not df.empty]

        if not dfs_valid:
            print(
                f"No valid measurement data loaded between {start_date} and {end_date}")
            return pd.DataFrame()

        # Concatenate monthly DataFrames
        combined = pd.concat(dfs_valid)

        # Ensure index is tz-naive for consistent slicing
        if isinstance(combined.index, pd.DatetimeIndex):
            if combined.index.tz is not None:
                combined.index = combined.index.tz_localize(None)
        else:
            combined = combined.set_index(pd.to_datetime(
                combined['timestamp'], errors='coerce'))
            combined.dropna(subset=['timestamp'], inplace=True)
            combined.index = combined.index.tz_localize(None)

        # Slice DataFrame safely
        mask = (combined.index >= start) & (combined.index <= end)
        sliced = combined.loc[mask]

        if sliced.empty:
            print(
                f"No measurement data found after slicing between {start_date} and {end_date}")

        # Build final measurement DataFrame
        return self._build_meas_df(sliced, use_msis=use_msis)

    def cache_measurements(self, start, end, prefer_remote=False, use_msis=True):
        self.cached_df = self.build_meas_data(start, end, prefer_remote, use_msis)
        return self.cached_df

    def get_measurement_slice(self, start, end):
        if self.cached_df.empty:
            return pd.DataFrame()
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        return self.cached_df[
            (self.cached_df['timestamp'] >= start) & (
                self.cached_df['timestamp'] <= end)
        ].copy()


def decimate_to_filter_grid(meas_df, start_date, T2):
    """
    Keep the first measurement falling in each filter step, so exactly one observation
    per satellite is assimilated per step.

    ROPE density files are sampled at 10 s while the filter runs on a T2 = 60 s grid, so
    six rows land in every step. Assimilating all six as independent observations
    deflates the effective measurement variance by ~6x per satellite per step; this
    selects the same row the scalar filter picked with rows.iloc[0]. Apply it per
    satellite *before* the variances are computed so the two stay aligned row-for-row.

    Parameters
    ----------
    meas_df : pd.DataFrame
        Measurements with a 'timestamp' column, sorted ascending.
    start_date : datetime
        Filter epoch, i.e. the time of step k = 0.
    T2 : float
        Filter time step in seconds.

    Returns
    -------
    pd.DataFrame
        Copy holding one row per occupied filter step, with a fresh RangeIndex.
    """
    df = meas_df.copy()
    times = df["timestamp"].values.astype("datetime64[ns]")
    k = ((times - np.datetime64(start_date)) /
         np.timedelta64(int(T2), "s")).astype(int)
    df["_step"] = k
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates("_step", keep="first").drop(columns="_step")
    return df.reset_index(drop=True)


def _as_variance_vector(R, name):
    """
    Normalise a measurement-noise argument to a 1-D vector of variances.

    Accepts either a 1-D vector or a 2-D diagonal covariance matrix, so callers can pass
    log_var_df['log_variance'].values directly instead of materialising np.diag(...),
    which at raw measurement cadence is a ~120k x 120k array.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return np.diag(R)
    if R.ndim == 1:
        return R
    raise ValueError(f"R for '{name}' must be 1-D or 2-D, got {R.shape}")


def _check_meas_variance_lengths(entries):
    """
    Assert each satellite's measurement frame and variance vector line up row-for-row.

    entries : sequence of (name, meas_df, R_vector)

    A mismatch means the variances were computed on a differently-sliced or
    differently-decimated frame than the one being assimilated, which silently
    mis-assigns noise to measurements. Raise instead of clipping.
    """
    for name, df, R in entries:
        if len(df) != len(R):
            raise ValueError(
                f"'{name}': {len(df)} measurements but {len(R)} variances. "
                "Compute the variances from the same (decimated) frame that is "
                "passed to the filter -- see decimate_to_filter_grid()."
            )


def monte_carlo_log_variance(meas_df, rel_error=0.05, n_samples=100, scaling_factor=1.0,
                             seed=0, return_matrix=True):
    """
    Estimate measurement noise variance in log space using Monte Carlo sampling.

    Parameters
    ----------
    meas_df : pd.DataFrame
        Measurement dataframe with 'accelerometer_density' and 'timestamp' columns.
    rel_error : float
        Assumed relative error in measurements (default 5%).
    n_samples : int
        Number of Monte Carlo samples.
    scaling_factor : float
        Multiplier on the log-space variance. Defaults to 1.0, i.e. R is exactly the
        variance implied by rel_error, with no undisclosed weighting. The former default
        of 1e-4 was hand-tuned to compensate for three defects in this function (natural
        log against a log10 innovation, an epsilon floor above typical density, and
        over-counted observations upstream); with those fixed, model error belongs in Q,
        not in a fudge factor on R. See tune_scaling_factor.py.
    seed : int or None
        Seed for the Monte-Carlo draw. Fixed by default so a published notebook
        re-runs identically; pass None for a fresh draw each call.
    return_matrix : bool
        Whether to build the dense (N, N) diagonal matrix. It is N^2 in the number of
        *measurements*, so on a 14-day GRACE-FO track (N = 120961) it is 117 GB and the
        process is OOM-killed before the filter starts; on a 1-day track it is 0.6 GB and
        merely wasteful. Every caller in this repo takes only the DataFrame -- the filter
        wants R_df['log_variance'].values -- so pass False unless the matrix is genuinely
        used. Default stays True so existing callers are unaffected.

    Returns
    -------
    log_var_df : pd.DataFrame
        DataFrame with 'timestamp' as index and estimated log-space variance,
        in log10^2 units to match the log10 innovation.
    R_diag : np.ndarray
        Diagonal covariance matrix (shape: [N, N]) for Kalman filter use.
    """
    rho = meas_df['accelerometer_density'].values
    timestamps = pd.to_datetime(meas_df['timestamp'])

    # Floor guarding log(0) only. It must stay far below any physical thermospheric
    # density -- ~1.3e-13 kg/m^3 at 400 km. The previous value of 1e-12 was ABOVE the
    # data, so both rho and delta_rho were pinned to the floor: the assumed error became
    # 100% instead of rel_error, and identical for every observation, which made R a
    # constant and the per-observation Monte Carlo below a no-op.
    epsilon = 1e-30

    rho = np.clip(rho, epsilon, None)
    # Derived from the floored rho, so it is positive without an absolute floor of its own.
    delta_rho = rel_error * rho

    # Monte Carlo sampling in linear space
    rng = np.random.default_rng(seed)
    samples = rho[:, None] + delta_rho[:, None] * \
        rng.standard_normal((len(rho), n_samples))
    samples = np.clip(samples, epsilon, None)

    # Convert to log space. Base 10, matching log_density = np.log10(rho) and the
    # log10 ROM basis, so R is in the same units as the innovation. Using natural log
    # here makes R too large by ln(10)^2 ~ 5.30.
    log_samples = np.log10(samples)

    # Compute sample-wise variance in log space
    log_var = np.var(log_samples, axis=1)  # shape: (N,)

    # Apply scaling factor to reduce measurement noise
    log_var_scaled = log_var * scaling_factor

    # Return as DataFrame and diagonal matrix
    log_var_df = pd.DataFrame({
        'timestamp': timestamps,
        'log_variance': log_var_scaled
    }).set_index('timestamp')

    R_diag = np.diag(log_var_scaled) if return_matrix else None

    return log_var_df, R_diag


class RopeDataInterpolator:
    """
    Interpolates PCA measurement operators (U0, mu0) at spacecraft locations.
    Optimized for sequential Kalman filtering: no time axis needed.
    """

    def __init__(self, U0, mu0,
                 lt_low=0, lt_high=23.6667,
                 lat_low=-87.5, lat_high=87.5,
                 alt_low=100, alt_high=980):
        """
        U0: PCA basis, shape (n_lt, n_lat, n_alt, n_components)
        mu0: mean density offset, shape (n_lt, n_lat, n_alt)
        """
        self.U0 = U0
        self.mu0 = mu0

        # Grids
        self.lt_grid = np.linspace(lt_low, lt_high, U0.shape[0])
        self.lat_grid = np.linspace(lat_low, lat_high, U0.shape[1])
        self.alt_grid = np.linspace(alt_low, alt_high, U0.shape[2])

        # Build interpolators once for speed
        self._interp_U0 = rgi(
            (self.lt_grid, self.lat_grid, self.alt_grid),
            U0, bounds_error=False, fill_value=None
        )
        self._interp_mu0 = rgi(
            (self.lt_grid, self.lat_grid, self.alt_grid),
            mu0, bounds_error=False, fill_value=None
        )

    def get_H(self, lla):
        """
        Return measurement operator H_U0 and H_mu0 at spacecraft locations.

        lla: array-like, shape (n_samples, 3)
             Columns: [latitude, local_time, altitude]

        Returns:
        H_U0: (n_samples, n_components)
        H_mu0: (n_samples, 1)
        """
        lla = np.atleast_2d(lla)

        # Build points for interpolation: (lt, lat, alt)
        points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])

        H_U0 = self._interp_U0(points)
        H_mu0 = self._interp_mu0(points).reshape(-1, 1)

        return H_U0, H_mu0

    def h_and_H(self, z, lla):
        """
        Measurement prediction and Jacobian in log10 density space.

        Same interface as coae_decoder.CAEMeasurementModel.h_and_H, so the Kalman filter
        can drive either reduced-order model through one code path. For this affine
        decoder H does not actually depend on z, but it is still recomputed per call --
        the cost is a grid interpolation, not a network evaluation.

        Returns (h (n_samples,), H (n_samples, n_components)).
        """
        H_U0, H_mu0 = self.get_H(lla)
        z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
        h = (H_U0 @ z + H_mu0).ravel()
        return h, H_U0

    def h_batch(self, z_seq, lla_seq, batch_size=None):
        """
        log10 density for a sequence of (state, location) pairs -- one location per state.

        z_seq:   (n_components, T)
        lla_seq: (T, 3), columns [latitude, local_time, altitude]
        Returns: (T,)

        Counterpart to coae_decoder.CAEMeasurementModel.h_batch, so post-processing along
        a satellite track is written once for both reduced-order models. `batch_size` is
        accepted and ignored -- this operator is already vectorised.
        """
        z_seq = np.atleast_2d(np.asarray(z_seq, dtype=np.float64))
        H_U0, H_mu0 = self.get_H(lla_seq)
        if H_U0.shape[0] != z_seq.shape[1]:
            raise ValueError(f'{z_seq.shape[1]} states but {H_U0.shape[0]} locations')
        return np.einsum('tc,ct->t', H_U0, z_seq) + H_mu0.ravel()

    def h_batch_grouped(self, z_unique, lla_seq, group, batch_size=None):
        """
        Counterpart to coae_decoder.CAEMeasurementModel.h_batch_grouped, so the
        along-track post-processing is written once for both reduced-order models.

        z_unique: (n_components, G) one state per distinct filter step
        lla_seq:  (T, 3);  group: (T,) index into z_unique
        There is nothing to save here -- this operator interpolates the basis rather than
        decoding a field -- so it simply expands and defers to h_batch.
        """
        z_unique = np.atleast_2d(np.asarray(z_unique, dtype=np.float64))
        return self.h_batch(z_unique[:, np.asarray(group, dtype=np.int64)], lla_seq)

    def h_only(self, z, lla):
        """log10 density at `lla` for a single state (the Jacobian is free here anyway)."""
        return self.h_and_H(z, lla)[0]

    def interpolate_density(self, z, lla):
        """
        Compute physical density at given spacecraft locations from state z.
        z: (n_components, 1)
        lla: (n_samples, 3)
        Returns: rho (n_samples, 1)
        """
        H_U0, H_mu0 = self.get_H(lla)
        z = z.reshape(-1, 1)
        log_rho = H_U0 @ z + H_mu0
        return 10.0 ** log_rho


def sindy_jacobian_cache_build(
    model_params, f10_idx, kp_idx, n_components,
    ar_drivers_start, X_reg_norm_dict, Y_reg_norm_dict,
):
    """
    Precompute static quantities for sindy_process_jacobian that don't change across KF steps.
    Call once before the main KF loop, pass the returned dict to sindy_process_jacobian each step.
    """
    n_t     = f10_idx - 1
    n_z     = n_components
    n_drv   = kp_idx
    n_total = n_z + n_drv
    poly_order = model_params.get('poly_order', 2) if isinstance(model_params, dict) else getattr(model_params, 'poly_order', 2)

    # --- extract powers from model_params ---
    try:
        functions_obj = model_params['functions']
    except (TypeError, KeyError):
        functions_obj = model_params.functions
    if hasattr(functions_obj, 'keys'):
        func_names = list(functions_obj.keys())
    else:
        func_names = [k for k in dir(functions_obj) if not k.startswith('_')]

    powers = []
    for fn in func_names:
        pw = re.findall(r'(?:\d+\.\d+|\d+|\.\d+)', fn)
        if pw:
            powers.append(float(pw[0]))
    powers_arr = np.array(powers, dtype=np.float64)
    n_funcs = len(powers_arr)
    m = n_funcs * n_total

    # --- pairwise indices (currently only supports poly_order=2) ---
    if poly_order == 2:
        iu, ju = np.triu_indices(m)
        n_theta_full = 1 + m + iu.size
    else:
        # For poly_order != 2, compute the full library size differently
        # NOTE: Higher-order support requires more complex derivative computation in sindy_process_jacobian
        iu, ju = np.triu_indices(m)
        from itertools import combinations_with_replacement
        n_interactions = sum(len(list(combinations_with_replacement(range(m), order)))
                           for order in range(2, poly_order + 1))
        n_theta_full = 1 + m + n_interactions

    full_poly_size = (n_total + 1) * (n_total + 2) // 2
    theta_start = (1 + n_z + n_t) if (n_funcs == 1 and n_theta_full == full_poly_size) else (1 + n_t)
    n_theta_d_nl = n_theta_full - theta_start

    # --- unary / pairwise split boundary ---
    unary_end = max(0, min(1 + m - theta_start, n_theta_d_nl))

    # --- pairwise index arrays (precomputed once) ---
    if n_theta_d_nl > unary_end and poly_order >= 2:
        c_pair   = np.arange(unary_end, n_theta_d_nl)
        p_c_arr  = (theta_start + c_pair) - (1 + m)
        a_arr    = iu[p_c_arr]
        b_arr    = ju[p_c_arr]
    else:
        a_arr = b_arr = np.empty(0, dtype=np.int64)

    # --- normalization arrays (flattened once) ---
    y_std  = np.asarray(Y_reg_norm_dict['x_std']).flatten()
    y_mean = np.asarray(Y_reg_norm_dict['x_mean']).flatten()
    x_std_theta = np.asarray(X_reg_norm_dict['x_std']).flatten()[:n_theta_d_nl]

    # --- darr_dz row/col index vectors (non-zero positions only) ---
    b_idx   = np.repeat(np.arange(n_funcs), n_z)
    j_idx   = np.tile(np.arange(n_z), n_funcs)
    row_idx = b_idx * n_total + j_idx
    p_rep   = np.repeat(powers_arr, n_z)

    # --- phi params ---
    n_phi      = ar_drivers_start
    phi_offset = 1 + n_t
    n_fill     = min(n_theta_d_nl, n_phi - phi_offset)
    src_start  = theta_start - 1

    return dict(
        n_t=n_t, n_z=n_z, n_total=n_total, n_funcs=n_funcs,
        powers_arr=powers_arr, m=m, iu=iu, ju=ju,
        theta_start=theta_start, n_theta_d_nl=n_theta_d_nl, unary_end=unary_end,
        a_arr=a_arr, b_arr=b_arr,
        y_std=y_std, y_mean=y_mean, x_std_theta=x_std_theta,
        row_idx=row_idx, j_idx=j_idx, p_rep=p_rep,
        n_phi=n_phi, phi_offset=phi_offset, n_fill=n_fill, src_start=src_start,
        poly_order=poly_order,
    )


def sindy_process_jacobian(
    z_k_norm,
    x_drivers_interp,
    interp_idx,
    A_aug,
    B_nonlinear,
    ar_drivers_start,
    n_components,
    f10_idx,
    kp_idx,
    model_params,
    X_reg_norm_dict,
    Y_reg_norm_dict,
    normalization_method,
    jac_cache=None,
):
    """
    Compute the full augmented-state Jacobian F_jac = dF/dX_aug for SINDy nonlinear process.

    Only the top-left (n_components × n_components) block differs from A_aug:
    F_jac = A_aug  +  delta_block
    where delta_block = B_nonlinear @ d(phi)/d(z_k_norm)

    Parameters
    ----------
    jac_cache : dict, optional
        Precomputed static quantities from sindy_jacobian_cache_build().
        If provided, skips recomputation of powers, indices, normalization arrays, etc.

    Notes
    -----
    Library column structure verified against mat_inputs/sindy_matrix_legend.mat.
    phi (non-AR) = 180 cols: [1 | t1..t4 (4) | f107..kp (4) | quadratic (171)].
    Quadratic ordering matches np.triu_indices(18) = combinations_with_replacement(range(18), 2)
    on the full input [x_0..x_9, t1..kp]: xx and xu interleaved by first x-index, then uu.
    uu terms have zero Jacobian w.r.t. z; xu contributes only through the x-side (driver rows
    of darr_dz are zero). x-state linear terms (p1(x_i)) are absent from the training library
    and are correctly skipped via theta_start = 1 + n_z + n_t.
    """
    n_z = n_components

    # Use cache if provided; otherwise compute on-the-fly (backward compatibility)
    if jac_cache is not None:
        n_t = jac_cache['n_t']
        n_total = jac_cache['n_total']
        n_funcs = jac_cache['n_funcs']
        m = jac_cache['m']
        powers_arr = jac_cache['powers_arr']
        iu, ju = jac_cache['iu'], jac_cache['ju']
        theta_start = jac_cache['theta_start']
        n_theta_d_nl = jac_cache['n_theta_d_nl']
        unary_end = jac_cache['unary_end']
        a_arr, b_arr = jac_cache['a_arr'], jac_cache['b_arr']
        y_std = jac_cache['y_std']
        y_mean = jac_cache['y_mean']
        x_std_theta = jac_cache['x_std_theta']
        row_idx = jac_cache['row_idx']
        j_idx = jac_cache['j_idx']
        p_rep = jac_cache['p_rep']
        n_phi = jac_cache['n_phi']
        phi_offset = jac_cache['phi_offset']
        n_fill = jac_cache['n_fill']
        src_start = jac_cache['src_start']
        poly_order = jac_cache.get('poly_order', 2)
    else:
        # Backward compatibility: compute on-the-fly
        n_t = f10_idx - 1
        n_drv = kp_idx
        n_total = n_z + n_drv
        poly_order = model_params.get('poly_order', 2) if isinstance(model_params, dict) else getattr(model_params, 'poly_order', 2)

        try:
            functions_obj = model_params['functions']
        except (TypeError, KeyError):
            functions_obj = model_params.functions
        if hasattr(functions_obj, 'keys'):
            func_names = list(functions_obj.keys())
        else:
            func_names = [k for k in dir(functions_obj) if not k.startswith('_')]

        powers = []
        for fn in func_names:
            pw = re.findall(r'(?:\d+\.\d+|\d+|\.\d+)', fn)
            if pw:
                powers.append(float(pw[0]))
        powers_arr = np.array(powers, dtype=np.float64)
        n_funcs = len(powers_arr)
        m = n_funcs * n_total

        if n_funcs == 0:
            return A_aug.copy()

        iu, ju = np.triu_indices(m)
        n_theta_full = 1 + m + iu.size
        full_poly_size = (n_total + 1) * (n_total + 2) // 2
        theta_start = (1 + n_z + n_t) if (n_funcs == 1 and n_theta_full == full_poly_size) else (1 + n_t)
        n_theta_d_nl = n_theta_full - theta_start

        unary_end = max(0, min(1 + m - theta_start, n_theta_d_nl))
        if n_theta_d_nl > unary_end:
            c_pair = np.arange(unary_end, n_theta_d_nl)
            p_c_arr = (theta_start + c_pair) - (1 + m)
            a_arr = iu[p_c_arr]
            b_arr = ju[p_c_arr]
        else:
            a_arr = b_arr = np.empty(0, dtype=np.int64)

        y_std = np.asarray(Y_reg_norm_dict['x_std']).flatten()
        y_mean = np.asarray(Y_reg_norm_dict['x_mean']).flatten()
        x_std_theta = np.asarray(X_reg_norm_dict['x_std']).flatten()[:n_theta_d_nl]

        b_idx = np.repeat(np.arange(n_funcs), n_z)
        j_idx = np.tile(np.arange(n_z), n_funcs)
        row_idx = b_idx * n_total + j_idx
        p_rep = np.repeat(powers_arr, n_z)

        n_phi = ar_drivers_start
        phi_offset = 1 + n_t
        n_fill = min(n_theta_d_nl, n_phi - phi_offset)
        src_start = theta_start - 1

    if n_funcs == 0:
        return A_aug.copy()

    # 1. Denormalize state
    z_k_denorm = np.asarray(z_k_norm).flatten() * y_std + y_mean

    # 2. Build X_k_for_sindy
    interp_idx_safe = min(interp_idx, x_drivers_interp.shape[1] - 1)
    dcol = np.asarray(x_drivers_interp[:, interp_idx_safe]).flatten()
    drv_for_sindy = dcol[1:kp_idx + 1]
    X_k = np.concatenate([z_k_denorm, drv_for_sindy])

    # 3. Build arr
    arr_blocks = [X_k ** p for p in powers_arr]
    arr = np.concatenate(arr_blocks)

    # 4. Vectorized darr_dz construction (replaces nested for loop)
    darr_dz = np.zeros((m, n_z))
    safe_exp = np.where(p_rep == 1.0, 1.0, p_rep - 1.0)
    vals = np.where(p_rep == 1.0, 1.0, p_rep * X_k[j_idx] ** safe_exp)
    darr_dz[row_idx, j_idx] = vals

    # 5. Vectorized dtheta_dz construction (replaces Python for loop)
    dtheta_dz = np.zeros((n_theta_d_nl, n_z))

    # Unary slice
    if unary_end > 0:
        dtheta_dz[:unary_end] = darr_dz[src_start : src_start + unary_end]

    # Pairwise slice
    if len(a_arr) > 0:
        dtheta_dz[unary_end:] = (arr[b_arr, np.newaxis] * darr_dz[a_arr]
                                + arr[a_arr, np.newaxis] * darr_dz[b_arr])

    # 6. Chain rule: w.r.t. normalized z, then w.r.t. normalized theta
    dtheta_dznorm = dtheta_dz * y_std[np.newaxis, :]
    dtheta_norm_dznorm = dtheta_dznorm / x_std_theta[:, np.newaxis]

    # 7. Build dphi_dz
    dphi_dz = np.zeros((n_phi, n_z))
    if n_fill > 0:
        dphi_dz[phi_offset:phi_offset + n_fill] = dtheta_norm_dznorm[:n_fill]

    # 8. Assemble F_jac
    F_jac = A_aug.copy()
    delta = B_nonlinear @ dphi_dz
    F_jac[:n_z, :n_z] += delta

    return np.asarray(F_jac, dtype=np.float64)


def sindy_step(
    z_hist,                    # normalized, shape (n, max_lag+1)
    x_drivers_interp,          # interpolated drivers (ALL rows)
    A_d2,
    B_ar_state_lst,
    B_ar_drivers_lst,
    B_nonlinear,
    ar_lags,
    interp_idx,
    ar_drivers_start,
    f10_idx,
    normalization_method,
    X_reg_norm_dict,
    Y_reg_norm_dict,
    input_features,
    model_params,
    kp_idx=None
):
    n_lags = len(ar_lags)
    if kp_idx is None:
        kp_idx = f10_idx + 3

    # ---- AR state history (past -> present)
    z_ar = np.fliplr(z_hist[:, -n_lags:])

    # Clamp interp_idx to valid range
    interp_idx = min(interp_idx, x_drivers_interp.shape[1] - 1)

    # ---- AR drivers: F10, F10A, Kp (3 drivers — explicitly indexed to skip kpstd)
    ar_driver_rows = [f10_idx, f10_idx + 1, kp_idx]
    ar_drivers = []
    for i in range(n_lags):
        idx = max(interp_idx - i, 0)
        ar_drivers.append(
            x_drivers_interp[ar_driver_rows, idx].reshape(-1, 1)
        )
    ar_drivers = np.concatenate(ar_drivers, axis=1)  # shape: (3, n_lags)

    # Reshape for normalization: stack vertically
    ar_drivers_flat = ar_drivers.T.reshape(-1, 1)  # shape: (3*n_lags, 1)

    # normalize AR drivers - now 3*n_lags instead of 2*n_lags
    X_ar_norm = {
        "x_mean": X_reg_norm_dict["x_mean"][
            ar_drivers_start: ar_drivers_start+3*n_lags
        ],
        "x_std": X_reg_norm_dict["x_std"][
            ar_drivers_start: ar_drivers_start+3*n_lags
        ],
    }
    ar_drivers_norm = u.normalize_with_dict(
        ar_drivers_flat, X_ar_norm, normalization_method)

    # Reshape back to (3, n_lags) for multiplication
    ar_drivers_norm = ar_drivers_norm.reshape(
        n_lags, 3).T  # shape: (3, n_lags)

    # ---- nonlinear inputs
    z_phys = u.denormalize(
        z_hist[:, -1:], Y_reg_norm_dict, normalization_method)
    sindy_inputs = build_dyn_sindy_inputs_optimized(
        z_phys,
        x_drivers_interp,
        X_reg_norm_dict,
        f10_idx,
        model_params,
        normalization_method,
        input_features,
        interp_idx,
        kp_idx=kp_idx,
    )
    x_nl = sindy_inputs[:ar_drivers_start]

    # ---- contributions
    ar_state = sum(
        B_ar_state_lst[i] @ z_ar[:, i:i+1]
        for i in range(n_lags)
    )
    ar_drv = sum(
        B_ar_drivers_lst[i] @ ar_drivers_norm[:, i:i+1]
        for i in range(n_lags)
    )

    z_next = (
        A_d2 @ z_hist[:, -1:]
        + ar_state
        + ar_drv
        + B_nonlinear @ x_nl
    )

    return z_next


def dmd_step(
    z_hist,
    x_drivers_interp,
    A_d2,
    B_d2,
    interp_idx,
    f10_idx,
    kp_idx,
    X_reg_norm_dict=None,
    normalization_method="standard",
    n_components=10,
):
    z_k = z_hist[:, -1:]
    
    interp_idx = min(interp_idx, x_drivers_interp.shape[1] - 1)
    u_k = x_drivers_interp[:kp_idx+1, interp_idx:interp_idx+1]
    x_nonlinear_k = np.concatenate([
        u_k[f10_idx:, :],  # Should be 3 features
        (u_k[f10_idx, :] * u_k[kp_idx, :]).reshape((1, -1)),  # F10 * Kp
        (u_k[kp_idx, :] * u_k[kp_idx, :]).reshape((1, -1)),   # Kp²
        z_k  # 10 state components
    ], axis=0)
       
    if X_reg_norm_dict is not None:
        x_nonlinear_norm = u.normalize_with_dict(x_nonlinear_k, X_reg_norm_dict, normalization_method)
    else:
        x_nonlinear_norm = x_nonlinear_k
    
    unnorm_part = u_k[1:f10_idx, :]
    
    x_full = np.concatenate([
        unnorm_part,
        x_nonlinear_norm
    ], axis=0)
    
    x_input = x_full[:-n_components]
    

    return A_d2 @ z_k + B_d2 @ x_input


def kalman_filter_logspace_ar(
    start_date,
    stop_date,
    model_object,
    interpolator,
    P0,
    Q,
    ar_lags,
    T1,
    T2,
    time_conversion_factor,
    n_components,
    meas_df,
    R_matrix,
    model_name="sindy-c",
    drivers="cls",
    f10_idx=5,
    kp_idx=None,
    nonlinear_meas=None,
    checkpoint_path=None,
    checkpoint_every=1440,
):
    """
    Log-space EKF with SINDy or DMD propagation.
    Fully robust to 1D/2D drivers and short driver histories.
    Updated to handle 3 AR drivers (F10, F10A, Kp) instead of 2.

    interpolator / nonlinear_meas: the measurement operator. Either may be given; both
    must expose

        h_and_H(z, lla) -> (h (n,), H (n, n_components))

    `interpolator` is the affine PCA operator (RopeDataInterpolator); `nonlinear_meas`
    is a learned decoder such as coae_decoder.CAEMeasurementModel, which relinearises
    about the current state at every measurement epoch. `nonlinear_meas` wins if both
    are supplied. Select between them from a config with
    rom_config.build_measurement_model().

    Note the filter state is the *raw* reduced coefficient, on the same scale as the
    model's z1_ds -- not the regression-normalised one -- so it can be handed to either
    measurement operator directly.

    checkpoint_path / checkpoint_every: if a path is given, the filtered and open-loop
    states filled so far are dumped there every `checkpoint_every` steps. Off by default,
    so the PCA path and the V4 notebooks are unaffected. It exists for the COAE path,
    where H is recomputed by autodiff at every measurement epoch and a full window is
    hours of work -- a kill mid-loop used to cost the entire run. `n_done` in the file
    says how many columns of `z_filt` are valid.
    """
    meas_model = nonlinear_meas if nonlinear_meas is not None else interpolator
    if meas_model is None:
        raise ValueError("provide either `interpolator` or `nonlinear_meas`")
    if kp_idx is None:
        kp_idx = f10_idx + 3

    # -------------------------------
    # 1) Open-loop forecast (independent, external)
    # -------------------------------
    z_openloop, _, _, _ = propagate(
        start_date=start_date,
        stop_date=stop_date,
        model_object=model_object,
        ar_lags=ar_lags,
        T1=T1,
        T2=T2,
        time_conversion_factor=time_conversion_factor,
        n_components=n_components,
        f10_idx=f10_idx,
        model_name=model_name,
        drivers=drivers,
        kp_idx=kp_idx
    )

    # -------------------------------
    # 2) Load model + drivers
    # -------------------------------
    if model_name == "sindy-c":
        B_full = model_object["B_sindy"]
        X_reg_norm_dict = {
            "x_mean": model_object["X_sindy_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["X_sindy_norm_dict"].x_std.reshape(-1, 1)
        }
        Y_reg_norm_dict = {
            "x_mean": model_object["Y_sindy_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["Y_sindy_norm_dict"].x_std.reshape(-1, 1)
        }
    elif model_name == "dmd-c":
        B_full = model_object["B_dmd"]
        X_reg_norm_dict = {
            "x_mean": model_object["X_dmd_ds_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["X_dmd_ds_norm_dict"].x_std.reshape(-1, 1)
        }
        Y_reg_norm_dict = {
            "x_mean": model_object["Y_dmd_ds_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["Y_dmd_ds_norm_dict"].x_std.reshape(-1, 1)
        }
    else:
        raise ValueError(f"Invalid model_name {model_name}")

    if drivers == "tiegcm2":
        u_ar_ds = model_object["u_ar_ds_tiegcm2"]
        datetimes_ds = model_object["datetimes_ds_tiegcm2"]
    else:
        u_ar_ds = model_object["u_ar_ds"]
        datetimes_ds = model_object["datetimes_ds"]

    # -------------------------------
    # 3) Convert drivers to 2D safely
    # -------------------------------
    u_ar_ds = np.array(u_ar_ds)

    if u_ar_ds.ndim == 1:
        u_ar_ds = u_ar_ds.reshape(1, -1)
    elif u_ar_ds.ndim == 2:
        n_rows, n_cols = u_ar_ds.shape
        if n_rows <= kp_idx:
            u_ar_ds = u_ar_ds.T
    else:
        raise ValueError(f"u_ar_ds must be 1D or 2D, got {u_ar_ds.shape}")

    if u_ar_ds.shape[0] <= kp_idx:
        raise ValueError(
            f"u_ar_ds must have at least {kp_idx+1} drivers (rows), got {u_ar_ds.shape}"
        )

    print(f"[DEBUG] u_ar_ds final shape: {u_ar_ds.shape}")

    # -------------------------------
    # 4) Time slicing & interpolation
    # -------------------------------
    datetimes_ds = get_propagator_datetimes(datetimes_ds)
    mask = np.array([(start_date <= t <= stop_date) for t in datetimes_ds])
    u_ar_ds = u_ar_ds[:, mask]

    x_drivers_interp = interpolate_matrix_rows(u_ar_ds, time_conversion_factor)
    x_drivers_interp = np.atleast_2d(x_drivers_interp)

    # -------------------------------
    # 5) Discrete-time matrices
    # -------------------------------
    A_d = B_full[:, -n_components:]
    B_d = B_full[:, :-n_components]
    A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)


    if model_name == "sindy-c":
        n_lags = len(ar_lags)
        # Updated: Now 3*max(ar_lags) instead of 2*max(ar_lags) for AR drivers
        ar_drivers_start = B_full.shape[1] - n_components - \
            max(ar_lags)*n_components - 3*max(ar_lags)
        ar_state_start = B_full.shape[1] - n_components - n_lags*n_components

        B_nonlinear = B_d2[:, :ar_drivers_start]
        # Updated: Now expecting 3 drivers per lag instead of 2
        B_ar_drivers_lst = [
            B_d2[:, ar_drivers_start+3*i: ar_drivers_start+3*(i+1)]
            for i in range(n_lags)
        ]
        B_ar_state_lst = [
            B_d2[:, ar_state_start+i *
                 n_components: ar_state_start+(i+1)*n_components]
            for i in range(n_lags)
        ]
    else:
        B_nonlinear = B_d2
        B_ar_drivers_lst = None
        B_ar_state_lst = None
        ar_drivers_start = None

    # -------------------------------
    # 6) Initial condition history
    # -------------------------------
    ic_array_data = model_object["ic_array"]
    if hasattr(ic_array_data, "ic_array"):
        ic_array_data = ic_array_data.ic_array
    ic_df = pd.DataFrame(
        ic_array_data,
        columns=["f10", "kp"] + [f"z_{k+1:02d}" for k in range(n_components)]
    )

    # **Clip n_ic_steps to driver array length**
    n_ic_steps = min(max(ar_lags)+1, u_ar_ds.shape[1])

    z_init = []
    for k in range(n_ic_steps):
        f10_val = u_ar_ds[f10_idx, k]
        kp_val = u_ar_ds[kp_idx, k]
        match = find_closest_match(ic_df, f10_val, kp_val)
        if match.empty:
            row = ic_df.iloc[0, 2:2+n_components].values.reshape(-1, 1)
        else:
            # Handle both DataFrame and Series returns
            if isinstance(match, pd.Series):
                row = match.iloc[2:2+n_components].values.reshape(-1, 1)
            else:
                row = match.iloc[0, 2:2+n_components].values.reshape(-1, 1)
        z_init.append(row)

    z_hist = np.concatenate(z_init, axis=1)
    z_hist = u.normalize_with_dict(
        z_hist, Y_reg_norm_dict, model_object["normalization_method"])
    # z_hist stays on the HOURLY grid: B_sindy's x_ar block was fit on hourly lags, so the
    # filter's AR taps must be an hour apart. (It used to be upsampled to the T2 grid by
    # interpolate_matrix_rows, which made the five "lags" 1-5 minutes apart while the
    # coefficients expected 1-5 hours.) The taps rotate once per hour in the loop below.

    # -------------------------------
    # 7) KF setup & augmentation
    # -------------------------------
    # State is augmented to [z_k; z_{k-1}; ...; z_{k-p+1}] where p = max(ar_lags)
    p_lags = max(ar_lags) if model_name == "sindy-c" else 1
    n_aug  = n_components * p_lags

    N = int((stop_date - start_date).total_seconds() / T2)
    z_filt = np.zeros((n_components, N))
    z_prior = np.zeros((n_components, N))

    # Per-step filter diagnostics, mirroring the joint filters and the windowed Nsat
    # driver so all three paths report consistency the same way. NIS = innov^2 / S for
    # this scalar update; 1.0 is a consistent filter, >1 means R too small or model
    # error, <1 means R too large.
    sl_mode      = np.zeros(N, dtype=np.uint8)
    sl_n_meas    = np.zeros(N, dtype=np.int16)
    sl_trace_P   = np.full(N, np.nan)
    sl_diag_P    = np.full((N, n_components), np.nan)
    sl_innov_rms = np.full(N, np.nan)
    sl_NIS       = np.full(N, np.nan)

    # Jacobian scale factor taking d/dz_norm to d/dz_raw for the measurement operator,
    # and the same factor used to report the filter output in raw coefficients. Only the
    # 'std' normalisation rescales; 'mean' and 'none' shift or do nothing.
    if model_object["normalization_method"] == "std":
        y_scale = Y_reg_norm_dict["x_std"].reshape(-1)[:n_components]
    else:
        y_scale = np.ones(n_components)

    # P0 and Q are given in *raw* coefficient units, so they mean the same thing across
    # reduced-order models whose latent scales differ by orders of magnitude (PCA sigma
    # is energy-ordered 142 -> 3.2; COAE is flat at ~0.05). The loop runs in normalised
    # space, so convert once here.
    _sc = np.outer(y_scale, y_scale)
    P0_norm = np.asarray(P0, dtype=float) / _sc
    Q_norm = np.asarray(Q, dtype=float) / _sc

    # Filter steps per hour, i.e. how often the hourly AR taps rotate.
    steps_per_hour = max(int(round(time_conversion_factor)), 1)

    # Augmented state and covariance
    X_aug  = np.zeros((n_aug, 1))
    P_aug  = np.zeros((n_aug, n_aug))
    # Initialize blocks with P0
    for i in range(p_lags):
        P_aug[i*n_components:(i+1)*n_components, i*n_components:(i+1)*n_components] = P0_norm.copy()

    Q_aug  = np.zeros((n_aug, n_aug))
    Q_aug[:n_components, :n_components] = Q_norm

    # Companion matrix A_aug
    A_aug = np.zeros((n_aug, n_aug))
    if model_name == "sindy-c":
        A_aug[:n_components, :n_components] = A_d2 + B_ar_state_lst[0]
        for i in range(1, p_lags):
            A_aug[:n_components, i*n_components:(i+1)*n_components] = B_ar_state_lst[i]
        for i in range(1, p_lags):
            A_aug[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = np.eye(n_components)
    else:
        A_aug[:n_components, :n_components] = A_d2
        if p_lags > 1:
            for i in range(1, p_lags):
                A_aug[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = np.eye(n_components)

    # Companion variant for steps that are not on an hour boundary. The AR taps are
    # hourly, so between boundaries they hold their own value instead of shifting down
    # the chain -- the covariance has to follow the same bookkeeping as the mean.
    A_hold = A_aug.copy()
    for i in range(1, p_lags):
        A_hold[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = 0.0
        A_hold[i*n_components:(i+1)*n_components, i*n_components:(i+1)*n_components] = np.eye(n_components)

    A_aug_T = A_aug.T
    I_aug   = np.eye(n_aug)
    P_filt  = []

    # Initialize augmented state from history
    for i in range(p_lags):
        # z_hist is (n, steps), newest is at -1
        X_aug[i*n_components:(i+1)*n_components, 0] = z_hist[:, -(i+1)]

    meas_times = meas_df["timestamp"].values.astype("datetime64[ns]")
    meas_k = ((meas_times - np.datetime64(start_date)) /
              np.timedelta64(int(T2), "s")).astype(int)
    mask = (meas_k >= 0) & (meas_k < N)
    meas_df = meas_df[mask].copy()
    meas_df["k"] = meas_k[mask]

    # Precompute Jacobian cache for sindy-c (called once before KF loop)
    if model_name == "sindy-c":
        jac_cache = sindy_jacobian_cache_build(
            model_params=model_object["model_params"],
            f10_idx=f10_idx,
            kp_idx=kp_idx,
            n_components=n_components,
            ar_drivers_start=ar_drivers_start,
            X_reg_norm_dict=X_reg_norm_dict,
            Y_reg_norm_dict=Y_reg_norm_dict,
        )
    else:
        jac_cache = None

    # -------------------------------
    # 8) KF loop
    # -------------------------------
    for k in range(N):
        if k % 1440 == 0:
            print(f"[ KF step {k+1}/{N} ]", flush=True)
        if checkpoint_path and k > 0 and k % checkpoint_every == 0:
            # Partial dump. Columns >= k are still zero-filled; n_done marks the boundary.
            np.savez(checkpoint_path, z_filt=z_filt[:, :k], z_openloop=z_openloop[:, :k],
                     n_done=k, n_total=N)
            print(f"    [checkpoint] {k}/{N} steps -> "
                  f"{os.path.basename(checkpoint_path)}", flush=True)
        # x_drivers_interp was upsampled from hourly to the T2 grid by
        # interpolate_matrix_rows(u_ar_ds, time_conversion_factor), so its columns are
        # already one per filter step: the driver index is k, not k*time_conversion_factor.
        # The old expression overshot by that factor and was silently pinned to the last
        # column by the clamp inside sindy_step, which froze the drivers a few minutes
        # into every run. sindy_loop_optimized does not clamp, so it surfaces instead.
        # `propagate` uses the same convention -- it passes fine-grid step indices
        # straight through as t0/tf.
        interp_idx = k

        # The AR taps live on the hourly grid; steps_per_hour filter steps make one hour.
        hour_boundary = (k > 0) and (k % steps_per_hour == 0)

        # ---- measurement update check
        rows = meas_df[meas_df.k == k]
        has_measurement = not rows.empty

        # Reconstruct chronological history for the model step [oldest -> newest]
        z_hist_for_step = np.fliplr(X_aug.reshape(p_lags, n_components).T)

        # ---- prediction (unconditional — always compute)
        if model_name == "sindy-c":
            # Use the same stepper `propagate` uses, rather than sindy_step. On identical
            # inputs sindy_step's one-step error is 10-5000x larger, from two defects it
            # does not share with sindy_loop_optimized:
            #   1. X_sindy_norm_dict holds 250 entries against 255 library features (the
            #      bias and 4 time rows are not normalised). sindy_loop_optimized offsets
            #      its slice by delta_idx=5 to account for that; sindy_step does not, so
            #      it normalises the AR drivers with the wrong 15 entries.
            #   2. Rows kp_idx+1 .. kp_idx+3*L of u_ar_ds already hold the AR-lagged
            #      [f107, f107a, kp] stack exactly. sindy_loop_optimized reads them;
            #      sindy_step rebuilds them from rows [f10_idx, f10_idx+1, kp_idx] using
            #      interp_idx shifts taken at the filter's T2 resolution rather than the
            #      model's hourly lag resolution.
            #
            # time_conversion_factor is passed as 1 so lag_offsets stay at consecutive
            # columns, matching the augmented state's existing one-step-shift structure
            # (X_aug_pred[n_components:] = X_aug[:-n_components]). z_hist is used rather
            # than z_hist_for_step because it carries max(ar_lags)+1 columns; the 5-column
            # z_hist_for_step would make the longest lag index wrap to -1.
            z_pred = sindy_loop_optimized(
                x_drivers_interp,
                X_reg_norm_dict,
                Y_reg_norm_dict,
                A_d2,
                B_ar_drivers_lst,
                B_ar_state_lst,
                B_nonlinear,
                z_hist,
                ar_drivers_start,
                interp_idx,
                interp_idx + 1,
                ar_lags,
                1,
                f10_idx,
                model_object["normalization_method"],
                {
                    "functions": mat_to_lambda_dict(model_object["model_params"].functions),
                    "shift": model_object["model_params"].shift,
                    "dof": model_object["model_params"].dof
                },
                model_object["input_features"],
                kp_idx=kp_idx,
            )[:, -1:]
        else:
            z_pred = dmd_step(
                z_hist_for_step,
                x_drivers_interp,
                A_d2,
                B_nonlinear,
                interp_idx,
                f10_idx,
                kp_idx,
                X_reg_norm_dict,
                model_object["normalization_method"],
                n_components,
            )

        # Augmented prediction. The lag taps are hourly, so they rotate only when a whole
        # hour of filter steps has elapsed; in between the older taps hold and only the
        # leading block advances.
        X_aug_pred = np.zeros_like(X_aug)
        X_aug_pred[:n_components, 0] = z_pred.flatten()
        if hour_boundary:
            X_aug_pred[n_components:, 0] = X_aug[:-n_components, 0]
        else:
            X_aug_pred[n_components:, 0] = X_aug[n_components:, 0]

        # Propagate covariance through the companion matrix -- the shifting variant on an
        # hour boundary, the holding variant otherwise, matching the mean update above.
        A_step = A_aug if hour_boundary else A_hold
        if model_name == "sindy-c":
            # EKF: linearize F_k at the prior mean ẑ_{k|k}, i.e. the input state to the
            # dynamics step, not the propagated state ẑ_{k+1|k}.
            F_jac = sindy_process_jacobian(
                z_k_norm=X_aug[:n_components, 0:1],
                x_drivers_interp=x_drivers_interp,
                interp_idx=interp_idx,
                A_aug=A_step,
                B_nonlinear=B_nonlinear,
                ar_drivers_start=ar_drivers_start,
                n_components=n_components,
                f10_idx=f10_idx,
                kp_idx=kp_idx,
                model_params=model_object["model_params"],
                X_reg_norm_dict=X_reg_norm_dict,
                Y_reg_norm_dict=Y_reg_norm_dict,
                normalization_method=model_object["normalization_method"],
                jac_cache=jac_cache,
            )
            F_jac_T = F_jac.T
            P_aug = F_jac @ P_aug @ F_jac_T + Q_aug
        else:
            # dmd-c: nonlinear input excludes z_k, so A_step is already the exact Jacobian
            P_aug = A_step @ P_aug @ A_step.T + Q_aug

        if has_measurement:
            # ---- measurement update
            r = rows.iloc[0]
            lla = np.array([[r.latitude, r.local_time, r.altitude]])

            # One path for both reduced-order models. For the affine PCA decoder H is
            # constant in z; for the COAE decoder this relinearises about the predicted
            # state.
            #
            # The filter state is normalised (that is the space B_sindy was fit in and
            # the space sindy_loop_optimized steps in), but both measurement operators
            # expect the raw coefficient. Denormalise before calling, then chain-rule the
            # Jacobian back: z_raw = z_norm * x_std + x_mean, so
            # dh/dz_norm = dh/dz_raw * x_std.
            z_raw = u.denormalize(X_aug_pred[:n_components, :1],
                                  Y_reg_norm_dict,
                                  model_object["normalization_method"]).ravel()
            h_val, H_raw = meas_model.h_and_H(z_raw, lla)
            H_small = H_raw * y_scale[None, :]

            H_aug = np.zeros((1, n_aug))
            H_aug[0, :n_components] = H_small.flatten()

            innov = r.log_density - h_val[0]

            S = H_aug @ P_aug @ H_aug.T + R_matrix[r.name]
            K = P_aug @ H_aug.T / S
            X_aug = X_aug_pred + K * innov

            # Joseph form
            ImKH = I_aug - K @ H_aug
            P_aug = ImKH @ P_aug @ ImKH.T + K @ np.array([[R_matrix[r.name]]]) @ K.T
            P_aug = 0.5 * (P_aug + P_aug.T)

            # ---- innovation diagnostics for this step (scalar update: n_meas = 1)
            sl_mode[k]      = 1
            sl_n_meas[k]    = 1
            sl_innov_rms[k] = abs(float(innov))
            sl_NIS[k]       = float(innov) ** 2 / float(np.squeeze(S))

        else:
            # No measurement: use prediction as state
            X_aug = X_aug_pred

        # ---- store prior and update AR history
        # The loop carries the normalised state; outputs are reported in raw
        # coefficients so they match `propagate`'s z_openloop and can be fed straight to
        # the measurement operator (this is what the downstream notebooks assume).
        z_curr = X_aug[:n_components, :1]
        # z_hist is the hourly AR buffer feeding the next prediction, so append one column
        # per hour rather than one per filter step. In between, its newest column tracks
        # the current estimate.
        if hour_boundary:
            z_hist = np.concatenate([z_hist, z_curr], axis=1)[:, -n_ic_steps:]
        else:
            z_hist[:, -1:] = z_curr

        norm_method = model_object["normalization_method"]
        z_prior[:, k] = u.denormalize(
            z_pred.reshape(-1, 1), Y_reg_norm_dict, norm_method).ravel()
        z_filt[:, k] = u.denormalize(
            z_curr, Y_reg_norm_dict, norm_method).ravel()

        # Covariance likewise: P_raw = diag(x_std) P_norm diag(x_std). trace_P and diag_P
        # are reported raw too, matching the block-update filters and the windowed Nsat
        # driver, so the released steplog is in the same units as z_filt whatever the
        # satellite count. This used to take the diagonal BEFORE the rescaling, so a
        # single-satellite window reported trace_P in normalised coordinates while a
        # multi-satellite one reported it raw -- y_scale[0] is 141.9, so the two differed
        # by ~1e4 on the leading mode.
        P_z = P_aug[:n_components, :n_components] * np.outer(y_scale, y_scale)
        P_filt.append(P_z.copy())

        _dp = np.diag(P_z)
        sl_trace_P[k] = float(np.sum(_dp))
        sl_diag_P[k]  = _dp

    step_arrays = {
        "mode":       sl_mode,
        "n_meas":     sl_n_meas,
        "trace_P":    sl_trace_P,
        "diag_P":     sl_diag_P,
        "innov_rms":  sl_innov_rms,
        "NIS":        sl_NIS,
    }
    return {"openloop": z_openloop, "filtered": z_filt, "P_filt": P_filt,
            "step_arrays": step_arrays}

def block_measurement(meas_model, X_aug_pred, n_components, llas, log_density,
                      Y_reg_norm_dict, normalization_method, y_scale):
    """
    Build one block measurement update from a NORMALISED predicted state.

    Shared by the three block-update filters (2-satellite, 3-satellite, N-satellite) so
    the raw/normalised convention is stated once instead of three times -- getting it
    wrong in one of them is exactly the defect this function was written to close; see the
    NORMALISATION note on kalman_filter_logspace_ar_joint_blockupdate.

    The loop carries a normalised state because that is the space sindy_step works in;
    both measurement operators expect the RAW coefficient. So denormalise, evaluate, then
    chain-rule the Jacobian back: z_raw = z_norm * x_std + x_mean, hence
    dh/dz_norm = dh/dz_raw * x_std.

    meas_model is either the affine PCA operator (get_H -> constant H and offset mu) or a
    learned decoder (h_and_H -> h and its Jacobian, relinearised about z_raw). Both give
    the same two quantities, so the caller's Kalman algebra is identical.

    Returns (H_block_small (n_meas, n_components) in normalised coordinates,
             innov (n_meas, 1)).
    """
    n_meas = len(llas)
    z_raw = denormalize(X_aug_pred[:n_components, :1],
                        Y_reg_norm_dict, normalization_method).ravel()

    if hasattr(meas_model, "h_and_H"):
        h_vals, H_raw = meas_model.h_and_H(z_raw, llas)
        H_raw = np.asarray(H_raw).reshape(n_meas, n_components)
        h_vals = np.asarray(h_vals).ravel()
    else:
        # Batched get_H where the operator supports it, row loop otherwise.
        try:
            H_raw, mu_vec = meas_model.get_H(llas)
            H_raw = np.asarray(H_raw).reshape(n_meas, n_components)
            mu_vec = np.asarray(mu_vec).ravel()
        except Exception:                                          # noqa: BLE001
            H_raw = np.empty((n_meas, n_components))
            mu_vec = np.empty(n_meas)
            for i in range(n_meas):
                H_i, mu_i = meas_model.get_H(llas[i:i + 1])
                H_raw[i, :] = np.asarray(H_i).ravel()
                # get_H returns mu with shape (n, 1); squeeze so this stays a scalar
                # assignment. NumPy >= 1.25 raises instead of silently unwrapping.
                mu_vec[i] = float(np.squeeze(mu_i))
        h_vals = H_raw @ z_raw + mu_vec

    innov = (np.asarray(log_density, dtype=float).ravel() - h_vals).reshape(-1, 1)
    return H_raw * y_scale[None, :], innov


def kalman_filter_logspace_ar_joint_blockupdate(
    start_date,
    stop_date,
    model_object,
    interpolator,
    P0,
    Q,
    ar_lags,
    T1,
    T2,
    time_conversion_factor,
    n_components,
    meas_df_assim_1,
    meas_df_assim_2,
    R_assim_1,
    R_assim_2,
    model_name="sindy-c",
    drivers="cls",
    f10_idx=5,
    kp_idx=None,
    nonlinear_meas=None,
):
    """
    Log-space EKF with SINDy or DMD propagation.
    Joint assimilation: CHAMP + GRACE (single block update)
    Fully aligned with kalman_filter_logspace_ar (single satellite).

    interpolator / nonlinear_meas: the measurement operator, exactly as in
    kalman_filter_logspace_ar. `interpolator` is the affine PCA operator
    (RopeDataInterpolator); `nonlinear_meas` is a learned decoder such as
    coae_decoder.CAEMeasurementModel, which relinearises about the current state at every
    measurement epoch. `nonlinear_meas` wins if both are supplied.

    NORMALISATION -- CHANGED 2026-08-31, THE PUBLISHED MULTI-SATELLITE NUMBERS MOVE
    ------------------------------------------------------------------------------
    sindy_step consumes and returns a NORMALISED z (`z_hist` is normalised at init and
    `z_phys = denormalize(z_hist[:, -1:])` inside it), and sindy_process_jacobian is called
    with `z_k_norm`. This filter previously formed its innovation as
    `y - H_raw @ X_aug_pred` with H_raw the RAW-coefficient basis U0, so the measurement
    update dragged X_aug toward the raw scale while the prediction step kept consuming it
    as normalised. Probed over a 1-day CHAMP+GRACE window, the first component ran
    -1.39 (normalised) -> -27 -> -39 -> -46 -> -53 -> -56 across the first six steps, while
    the single-satellite filter held O(1) throughout. Y_sindy_norm_dict.x_std[0] is 134.9
    for the PCA model, so the two scales are not interchangeable, and the run survived only
    because measurements are dense enough for the update to re-dominate every step.

    The loop is now self-consistent for BOTH operators, mirroring
    kalman_filter_logspace_ar:
      * P0 and Q are read as RAW-coefficient covariances and converted once into the
        normalised space the loop runs in (divide by outer(y_scale, y_scale)), so they mean
        the same thing across models whose latent scales differ by orders of magnitude
        (PCA sigma is energy-ordered 142 -> 3.2; COAE is flat at ~0.05);
      * the state is denormalised before get_H / h_and_H and the Jacobian chain-ruled back
        by y_scale;
      * z_filt, z_prior and P_filt are denormalised on output, so this still RETURNS raw
        coefficients, which is what reconstruction (rho = 10**(H z + mu)) expects.

    This was not a COAE-only concern -- a decoder at x_std ~ 0.061 would have returned
    garbage rather than a degraded answer -- but it changes the PCA multi-satellite rows of
    the published table and the released 2000-2025 dataset, both of which must be
    regenerated. The single-satellite filter was already correct and its rows do not move.
    """
    if kp_idx is None:
        kp_idx = f10_idx + 3

    # -------------------------------
    # 1) Open-loop forecast (independent, external)
    # -------------------------------
    z_openloop, _, _, _ = propagate(
        start_date=start_date,
        stop_date=stop_date,
        model_object=model_object,
        ar_lags=ar_lags,
        T1=T1,
        T2=T2,
        time_conversion_factor=time_conversion_factor,
        n_components=n_components,
        f10_idx=f10_idx,
        model_name=model_name,
        drivers=drivers,
        kp_idx=kp_idx
    )

    # -------------------------------
    # 2) Load model + drivers
    # -------------------------------
    if model_name == "sindy-c":
        B_full = model_object["B_sindy"]
        X_reg_norm_dict = {
            "x_mean": model_object["X_sindy_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["X_sindy_norm_dict"].x_std.reshape(-1, 1)
        }
        Y_reg_norm_dict = {
            "x_mean": model_object["Y_sindy_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["Y_sindy_norm_dict"].x_std.reshape(-1, 1)
        }
    elif model_name == "dmd-c":
        B_full = model_object["B_dmd"]
        X_reg_norm_dict = {
            "x_mean": model_object["X_dmd_ds_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["X_dmd_ds_norm_dict"].x_std.reshape(-1, 1)
        }
        Y_reg_norm_dict = {
            "x_mean": model_object["Y_dmd_ds_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["Y_dmd_ds_norm_dict"].x_std.reshape(-1, 1)
        }
    else:
        raise ValueError(f"Invalid model_name {model_name}")

    if drivers == "tiegcm2":
        u_ar_ds = model_object["u_ar_ds_tiegcm2"]
        datetimes_ds = model_object["datetimes_ds_tiegcm2"]
    else:
        u_ar_ds = model_object["u_ar_ds"]
        datetimes_ds = model_object["datetimes_ds"]

    # -------------------------------
    # 3) Convert drivers to 2D safely
    # -------------------------------
    u_ar_ds = np.asarray(u_ar_ds)

    if u_ar_ds.ndim == 1:
        u_ar_ds = u_ar_ds.reshape(1, -1)
    elif u_ar_ds.ndim == 2 and u_ar_ds.shape[0] <= kp_idx:
        u_ar_ds = u_ar_ds.T

    if u_ar_ds.shape[0] <= kp_idx:
        raise ValueError(
            f"u_ar_ds must have at least {kp_idx+1} drivers, got {u_ar_ds.shape}"
        )

    # -------------------------------
    # 4) Time slicing & interpolation
    # -------------------------------
    datetimes_ds = get_propagator_datetimes(datetimes_ds)
    mask = np.array([(start_date <= t <= stop_date) for t in datetimes_ds])
    u_ar_ds = u_ar_ds[:, mask]

    x_drivers_interp = interpolate_matrix_rows(
        u_ar_ds, time_conversion_factor)
    x_drivers_interp = np.atleast_2d(x_drivers_interp)

    # -------------------------------
    # 5) Discrete-time matrices
    # -------------------------------
    A_d = B_full[:, -n_components:]
    B_d = B_full[:, :-n_components]
    A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)
    A_d2_T = A_d2.T

    if model_name == "sindy-c":
        n_lags = len(ar_lags)

        # UPDATED: 3 drivers per lag
        ar_drivers_start = (
            B_full.shape[1]
            - n_components
            - max(ar_lags) * n_components
            - 3 * max(ar_lags)
        )

        ar_state_start = (
            B_full.shape[1]
            - n_components
            - n_lags * n_components
        )

        B_nonlinear = B_d2[:, :ar_drivers_start]

        B_ar_drivers_lst = [
            B_d2[:, ar_drivers_start + 3*i : ar_drivers_start + 3*(i+1)]
            for i in range(n_lags)
        ]

        B_ar_state_lst = [
            B_d2[:, ar_state_start + i*n_components :
                     ar_state_start + (i+1)*n_components]
            for i in range(n_lags)
        ]
    else:
        B_nonlinear = B_d2
        B_ar_drivers_lst = None
        B_ar_state_lst = None
        ar_drivers_start = None

    # -------------------------------
    # 6) Initial condition history
    # -------------------------------
    ic_array_data = model_object["ic_array"]
    if hasattr(ic_array_data, "ic_array"):
        ic_array_data = ic_array_data.ic_array
    ic_df = pd.DataFrame(
        ic_array_data,
        columns=["f10", "kp"] +
        [f"z_{k+1:02d}" for k in range(n_components)]
    )

    n_ic_steps = min(max(ar_lags) + 1, u_ar_ds.shape[1])

    z_init = []
    for k in range(n_ic_steps):
        f10_val = u_ar_ds[f10_idx, k]
        kp_val = u_ar_ds[kp_idx, k]
        match = find_closest_match(ic_df, f10_val, kp_val)

        if match.empty:
            row = ic_df.iloc[0, 2:2+n_components].values.reshape(-1, 1)
        else:
            row = (
                match.iloc[2:2+n_components]
                if isinstance(match, pd.Series)
                else match.iloc[0, 2:2+n_components]
            ).values.reshape(-1, 1)

        z_init.append(row)

    z_hist = np.concatenate(z_init, axis=1)
    z_hist = u.normalize_with_dict(
        z_hist, Y_reg_norm_dict,
        model_object["normalization_method"]
    )
    # z_hist stays on the HOURLY grid, matching kalman_filter_logspace_ar. B_sindy's x_ar
    # block was fit on hourly lags, so the filter's AR taps must be an hour apart. This used
    # to be upsampled to the T2 grid by interpolate_matrix_rows, which made the five "lags"
    # 1-5 minutes apart while the coefficients expected 1-5 hours -- and, because the buffer
    # holds n_ic_steps columns and gains one per hour, left it contaminated for the first
    # ~5 hours of every window. On the Halloween IC the true hourly taps of mode 0 span
    # 173.1 / 203.9 / 183.5 / 180.6 / 205.6 / 173.6; the upsampled ones span
    # 179.9 / 178.7 / 177.4 / 176.2 / 174.9 / 173.6 -- the lag structure was flattened out.
    # Commit 90cf023 removed the upsampling from the single-satellite filter but not here.
    # The taps rotate once per hour in the loop below.

    normalization_method = model_object["normalization_method"]

    # Jacobian scale factor taking d/dz_norm to d/dz_raw for the measurement operator,
    # and the same factor used to report the filter output in raw coefficients. Only the
    # 'std' normalisation rescales; 'mean' and 'none' shift or do nothing.
    if normalization_method == "std":
        y_scale = Y_reg_norm_dict["x_std"].reshape(-1)[:n_components]
    else:
        y_scale = np.ones(n_components)

    # P0 and Q arrive in *raw* coefficient units; the loop runs normalised. See the
    # NORMALISATION note in the docstring.
    _sc = np.outer(y_scale, y_scale)
    P0 = np.asarray(P0, dtype=float) / _sc
    Q = np.asarray(Q, dtype=float) / _sc

    meas_model = nonlinear_meas if nonlinear_meas is not None else interpolator
    if meas_model is None:
        raise ValueError("provide either `interpolator` or `nonlinear_meas`")

    # -------------------------------
    # 7) KF setup & augmentation
    # -------------------------------
    # State is augmented to [z_k; z_{k-1}; ...; z_{k-p+1}] where p = max(ar_lags)
    p_lags = max(ar_lags) if model_name == "sindy-c" else 1
    n_aug  = n_components * p_lags

    N = int((stop_date - start_date).total_seconds() / T2)
    z_filt = np.zeros((n_components, N))
    z_prior = np.zeros((n_components, N))

    # Per-step filter diagnostics, mirroring ark_filter_windowed_Nsat so both paths
    # report consistency the same way. NIS = innov^T S^-1 innov / n_meas; 1.0 is a
    # consistent filter, >1 means R too small or model error, <1 means R too large.
    sl_mode      = np.zeros(N, dtype=np.uint8)
    sl_n_meas    = np.zeros(N, dtype=np.int16)
    sl_trace_P   = np.full(N, np.nan)
    sl_diag_P    = np.full((N, n_components), np.nan)
    sl_innov_rms = np.full(N, np.nan)
    sl_NIS       = np.full(N, np.nan)

    # Augmented state and covariance
    X_aug  = np.zeros((n_aug, 1))
    P_aug  = np.zeros((n_aug, n_aug))
    # Initialize blocks with P0
    for i in range(p_lags):
        P_aug[i*n_components:(i+1)*n_components, i*n_components:(i+1)*n_components] = P0.copy()

    Q_aug  = np.zeros((n_aug, n_aug))
    Q_aug[:n_components, :n_components] = Q

    # Companion matrix A_aug
    A_aug = np.zeros((n_aug, n_aug))
    if model_name == "sindy-c":
        A_aug[:n_components, :n_components] = A_d2 + B_ar_state_lst[0]
        for i in range(1, p_lags):
            A_aug[:n_components, i*n_components:(i+1)*n_components] = B_ar_state_lst[i]
        for i in range(1, p_lags):
            A_aug[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = np.eye(n_components)
    else:
        A_aug[:n_components, :n_components] = A_d2
        if p_lags > 1:
            for i in range(1, p_lags):
                A_aug[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = np.eye(n_components)

    # Companion variant for steps that are not on an hour boundary. The AR taps are
    # hourly, so between boundaries they hold their own value instead of shifting down
    # the chain -- the covariance has to follow the same bookkeeping as the mean.
    A_hold = A_aug.copy()
    for i in range(1, p_lags):
        A_hold[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = 0.0
        A_hold[i*n_components:(i+1)*n_components, i*n_components:(i+1)*n_components] = np.eye(n_components)

    # Filter steps per hour, i.e. how often the hourly AR taps rotate.
    steps_per_hour = max(int(round(time_conversion_factor)), 1)

    A_aug_T = A_aug.T
    I_aug   = np.eye(n_aug)
    P_filt  = []

    # Initialize augmented state from history
    for i in range(p_lags):
        X_aug[i*n_components:(i+1)*n_components, 0] = z_hist[:, -(i+1)]

    # -------------------------------
    # 8) Prepare joint measurements
    # -------------------------------
    # Combine CHAMP and GRACE into single dataframe
    meas_df_assim_1 = meas_df_assim_1.copy()
    meas_df_assim_2 = meas_df_assim_2.copy()

    # Accept R as a 1-D vector of variances or a 2-D diagonal covariance matrix.
    # The 1-D form is preferred: at raw cadence the dense matrix is ~120k x 120k.
    R_assim_1 = _as_variance_vector(R_assim_1, 'champ')
    R_assim_2 = _as_variance_vector(R_assim_2, 'grace')

    # Measurements and variances must line up row-for-row. Guaranteed when
    # decimate_to_filter_grid() is applied per satellite before the variances are
    # computed; fail loudly rather than silently reusing the last variance.
    _check_meas_variance_lengths(
        [('champ', meas_df_assim_1, R_assim_1),
         ('grace', meas_df_assim_2, R_assim_2)])

    # Add original row numbers BEFORE any filtering
    meas_df_assim_1['orig_idx'] = np.arange(len(meas_df_assim_1))
    meas_df_assim_2['orig_idx'] = np.arange(len(meas_df_assim_2))

    meas_df_assim_1['satellite'] = 'champ'
    meas_df_assim_2['satellite'] = 'grace'

    meas_df_joint = pd.concat(
        [meas_df_assim_1, meas_df_assim_2], ignore_index=True)

    # Compute time indices
    meas_times = meas_df_joint["timestamp"].values.astype("datetime64[ns]")
    meas_k = ((meas_times - np.datetime64(start_date)) /
              np.timedelta64(int(T2), "s")).astype(int)
    mask = (meas_k >= 0) & (meas_k < N)
    meas_df_joint = meas_df_joint[mask].copy()
    meas_df_joint["k"] = meas_k[mask]

    # Reset index to get sequential new indices
    meas_df_joint.reset_index(drop=True, inplace=True)

    # SPEEDUP: Build R_joint vectorized
    n_meas = len(meas_df_joint)
    R_joint = np.zeros(n_meas)  # Store only diagonal (it's diagonal anyway)

    # Vectorized R assignment
    champ_mask = meas_df_joint['satellite'] == 'champ'
    grace_mask = ~champ_mask

    champ_indices = meas_df_joint.loc[champ_mask, 'orig_idx'].values
    grace_indices = meas_df_joint.loc[grace_mask, 'orig_idx'].values

    R_joint[champ_mask] = R_assim_1[champ_indices]
    R_joint[grace_mask] = R_assim_2[grace_indices]

    # SPEEDUP: Pre-group measurements by timestep
    meas_groups = meas_df_joint.groupby('k', sort=False)

    # SPEEDUP: Extract model params once
    if model_name == "sindy-c":
        model_params_dict = {
            "functions": mat_to_lambda_dict(model_object["model_params"].functions),
            "shift": model_object["model_params"].shift,
            "dof": model_object["model_params"].dof
        }
        input_features = model_object["input_features"]

    # Precompute Jacobian cache for sindy-c (called once before KF loop)
    if model_name == "sindy-c":
        jac_cache = sindy_jacobian_cache_build(
            model_params=model_object["model_params"],
            f10_idx=f10_idx,
            kp_idx=kp_idx,
            n_components=n_components,
            ar_drivers_start=ar_drivers_start,
            X_reg_norm_dict=X_reg_norm_dict,
            Y_reg_norm_dict=Y_reg_norm_dict,
        )
    else:
        jac_cache = None

    # -------------------------------
    # 9) KF loop with block updates
    # -------------------------------
    for k in range(N):

        if k % 1440 == 0:
            print(f"[ KF step {k+1}/{N} ]")

        # x_drivers_interp was upsampled from hourly to the T2 grid by
        # interpolate_matrix_rows(u_ar_ds, time_conversion_factor), so its columns are
        # already one per filter step: the driver index is k, not k*time_conversion_factor.
        # The old expression overshot by that factor and was silently pinned to the last
        # column by the clamp inside sindy_step, which froze the drivers a few minutes
        # into every run. `propagate` uses the same convention.
        interp_idx = k

        # The AR taps live on the hourly grid; steps_per_hour filter steps make one hour.
        hour_boundary = (k > 0) and (k % steps_per_hour == 0)

        # Check if we have measurements at this timestep
        has_measurement = k in meas_groups.groups

        # Reconstruct chronological history for the model step [oldest -> newest]
        z_hist_for_step = np.fliplr(X_aug.reshape(p_lags, n_components).T)

        # ---- prediction (unconditional — always compute)
        if model_name == "sindy-c":
            # Use the same stepper `propagate` and the single-satellite filter use, rather
            # than sindy_step. On identical inputs sindy_step's one-step error is 10-5000x
            # larger, from two defects it does not share with sindy_loop_optimized:
            #   1. X_sindy_norm_dict holds 250 entries against 255 library features (the
            #      bias and 4 time rows are not normalised). sindy_loop_optimized offsets
            #      its slice by delta_idx=5 to account for that; sindy_step does not, so
            #      it normalises the AR drivers with the wrong 15 entries.
            #   2. Rows kp_idx+1 .. kp_idx+3*L of u_ar_ds already hold the AR-lagged
            #      [f107, f107a, kp] stack exactly. sindy_loop_optimized reads them;
            #      sindy_step rebuilds them from rows [f10_idx, f10_idx+1, kp_idx] using
            #      interp_idx shifts taken at the filter's T2 resolution rather than the
            #      model's hourly lag resolution.
            #
            # Commit 90cf023 made this swap in kalman_filter_logspace_ar but not in the
            # block-update filters, so the multi-satellite runs kept the old stepper.
            #
            # time_conversion_factor is passed as 1 so lag_offsets stay at consecutive
            # columns, matching the augmented state's existing one-step-shift structure.
            # z_hist is used rather than z_hist_for_step because it carries
            # max(ar_lags)+1 columns; the 5-column z_hist_for_step would make the longest
            # lag index wrap to -1.
            z_pred = sindy_loop_optimized(
                x_drivers_interp,
                X_reg_norm_dict,
                Y_reg_norm_dict,
                A_d2,
                B_ar_drivers_lst,
                B_ar_state_lst,
                B_nonlinear,
                z_hist,
                ar_drivers_start,
                interp_idx,
                interp_idx + 1,
                ar_lags,
                1,
                f10_idx,
                model_object["normalization_method"],
                {
                    "functions": mat_to_lambda_dict(model_object["model_params"].functions),
                    "shift": model_object["model_params"].shift,
                    "dof": model_object["model_params"].dof
                },
                model_object["input_features"],
                kp_idx=kp_idx,
            )[:, -1:]
        else:
            z_pred = dmd_step(
                z_hist_for_step,
                x_drivers_interp,
                A_d2,
                B_nonlinear,
                interp_idx,
                f10_idx,
                kp_idx,
                X_reg_norm_dict,
                model_object["normalization_method"],
                n_components,
            )

        # Augmented prediction. The lag taps are hourly, so they rotate only when a whole
        # hour of filter steps has elapsed; in between the older taps hold and only the
        # leading block advances.
        X_aug_pred = np.zeros_like(X_aug)
        X_aug_pred[:n_components, 0] = z_pred.flatten()
        if hour_boundary:
            X_aug_pred[n_components:, 0] = X_aug[:-n_components, 0]
        else:
            X_aug_pred[n_components:, 0] = X_aug[n_components:, 0]

        # Propagate covariance through the companion matrix -- the shifting variant on an
        # hour boundary, the holding variant otherwise, matching the mean update above.
        A_step = A_aug if hour_boundary else A_hold
        if model_name == "sindy-c":
            # EKF: linearize F_k at the prior mean ẑ_{k|k}, i.e. the input state to the
            # dynamics step, not the propagated state ẑ_{k+1|k}.
            F_jac = sindy_process_jacobian(
                z_k_norm=X_aug[:n_components, 0:1],
                x_drivers_interp=x_drivers_interp,
                interp_idx=interp_idx,
                A_aug=A_step,
                B_nonlinear=B_nonlinear,
                ar_drivers_start=ar_drivers_start,
                n_components=n_components,
                f10_idx=f10_idx,
                kp_idx=kp_idx,
                model_params=model_object["model_params"],
                X_reg_norm_dict=X_reg_norm_dict,
                Y_reg_norm_dict=Y_reg_norm_dict,
                normalization_method=model_object["normalization_method"],
                jac_cache=jac_cache,
            )
            F_jac_T = F_jac.T
            P_aug = F_jac @ P_aug @ F_jac_T + Q_aug
        else:
            # dmd-c: nonlinear input excludes z_k, so A_step is already the exact Jacobian
            P_aug = A_step @ P_aug @ A_step.T + Q_aug

        if has_measurement:
            # ---- Block measurement update
            rows_k = meas_groups.get_group(k)
            n_meas_k = len(rows_k)

            # Exactly one observation per satellite per step. More than one means the
            # frames were not decimated onto the T2 grid, which would deflate the
            # effective measurement variance by the oversampling ratio.
            if rows_k.groupby('satellite').size().max() != 1:
                # A raise, not an assert: `python -O` strips asserts, and silently
                # over-counting observations is the defect this guards against.
                raise ValueError(
                    f"step {k}: {n_meas_k} observations across "
                    f"{rows_k['satellite'].nunique()} satellites -- apply "
                    "decimate_to_filter_grid() before computing R")

            # Build H matrix and innovation vector for all measurements at this timestep.
            # SPEEDUP: Vectorize LLA extraction
            llas = rows_k[['latitude', 'local_time', 'altitude']].values
            H_block_small, innov = block_measurement(
                meas_model, X_aug_pred, n_components, llas,
                rows_k['log_density'].to_numpy(),
                Y_reg_norm_dict, normalization_method, y_scale)

            # Augmented H: [H, 0, ..., 0]
            H_block = np.zeros((n_meas_k, n_aug))
            H_block[:, :n_components] = H_block_small

            # SPEEDUP: Use diagonal R directly
            R_block_diag = R_joint[rows_k.index.values]

            # innov comes from block_measurement, already y - h(z_pred).

            # Kalman gain (block update)
            S = H_block @ P_aug @ H_block.T
            S[np.diag_indices_from(S)] += R_block_diag
            K = P_aug @ H_block.T @ np.linalg.solve(S, np.eye(n_meas_k))

            # Update state and covariance
            X_aug = X_aug_pred + K @ innov
            # Joseph form
            ImKH = I_aug - K @ H_block
            P_aug = ImKH @ P_aug @ ImKH.T + K @ (R_block_diag[:, None] * K.T)
            P_aug = 0.5 * (P_aug + P_aug.T)

            # ---- innovation diagnostics for this step
            sl_mode[k]      = 1
            sl_n_meas[k]    = n_meas_k
            sl_innov_rms[k] = float(np.sqrt(np.mean(innov ** 2)))
            sl_NIS[k]       = float(np.squeeze(
                innov.T @ np.linalg.solve(S, innov))) / n_meas_k

        else:
            # No measurement: use prediction as state
            X_aug = X_aug_pred

        # ---- store prior and update AR history
        # z_curr stays normalized: that is the space z_hist and sindy_step work in.
        # z_hist is the hourly AR buffer feeding the next prediction, so append one column
        # per hour rather than one per filter step. In between, its newest column tracks
        # the current estimate.
        z_curr = X_aug[:n_components, :1]
        if hour_boundary:
            z_hist = np.concatenate([z_hist, z_curr], axis=1)[:, -n_ic_steps:]
        else:
            z_hist[:, -1:] = z_curr

        # Outputs are reported in RAW coefficients, matching `propagate`'s z_openloop and
        # what reconstruction (rho = 10**(H z + mu)) expects. The loop itself is
        # normalized throughout -- see the NORMALISATION note in the docstring.
        z_prior[:, k] = denormalize(
            z_pred.reshape(-1, 1), Y_reg_norm_dict, normalization_method).ravel()
        z_filt[:, k] = denormalize(
            z_curr, Y_reg_norm_dict, normalization_method).ravel()

        # Covariance likewise: P_raw = diag(x_std) P_norm diag(x_std). trace_P and diag_P
        # are reported raw too, so the released steplog is in the same units as z_filt.
        P_z = P_aug[:n_components, :n_components] * np.outer(y_scale, y_scale)
        P_filt.append(P_z.copy())
        _dp = np.diag(P_z)
        sl_trace_P[k] = float(np.sum(_dp))
        sl_diag_P[k]  = _dp


    step_arrays = {
        "mode":       sl_mode,
        "n_meas":     sl_n_meas,
        "trace_P":    sl_trace_P,
        "diag_P":     sl_diag_P,
        "innov_rms":  sl_innov_rms,
        "NIS":        sl_NIS,
    }
    return {"openloop": z_openloop, "filtered": z_filt, "P_filt": P_filt,
            "step_arrays": step_arrays}


def kalman_filter_logspace_ar_joint_blockupdate_all(
    start_date,
    stop_date,
    model_object,
    interpolator,
    P0,
    Q,
    ar_lags,
    T1,
    T2,
    time_conversion_factor,
    n_components,
    meas_df_assim_1,
    meas_df_assim_2,
    meas_df_assim_3,
    R_assim_1,
    R_assim_2,
    R_assim_3,
    model_name="sindy-c",
    drivers="cls",
    f10_idx=5,
    kp_idx=None,
    nonlinear_meas=None,
):
    """
    Log-space EKF with SINDy or DMD propagation.
    Joint assimilation: CHAMP + GRACE + GOCE (single block update)
    Fully aligned with working single-satellite KF.

    See kalman_filter_logspace_ar_joint_blockupdate for the operator arguments and for the
    NORMALISATION note: this filter carried the same raw/normalised mismatch and takes the
    same correction, so the published three-satellite row moves with the two-satellite one.
    """
    if kp_idx is None:
        kp_idx = f10_idx + 3

    # -------------------------------
    # 1) Open-loop forecast (independent, external)
    # -------------------------------
    z_openloop, _, _, _ = propagate(
        start_date=start_date,
        stop_date=stop_date,
        model_object=model_object,
        ar_lags=ar_lags,
        T1=T1,
        T2=T2,
        time_conversion_factor=time_conversion_factor,
        n_components=n_components,
        f10_idx=f10_idx,
        model_name=model_name,
        drivers=drivers,
        kp_idx=kp_idx
    )

    # -------------------------------
    # 2) Load model + drivers
    # -------------------------------
    if model_name == "sindy-c":
        B_full = model_object["B_sindy"]
        X_reg_norm_dict = {
            "x_mean": model_object["X_sindy_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["X_sindy_norm_dict"].x_std.reshape(-1, 1),
        }
        Y_reg_norm_dict = {
            "x_mean": model_object["Y_sindy_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["Y_sindy_norm_dict"].x_std.reshape(-1, 1),
        }
    elif model_name == "dmd-c":
        B_full = model_object["B_dmd"]
        X_reg_norm_dict = {
            "x_mean": model_object["X_dmd_ds_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["X_dmd_ds_norm_dict"].x_std.reshape(-1, 1),
        }
        Y_reg_norm_dict = {
            "x_mean": model_object["Y_dmd_ds_norm_dict"].x_mean.reshape(-1, 1),
            "x_std":  model_object["Y_dmd_ds_norm_dict"].x_std.reshape(-1, 1),
        }
    else:
        raise ValueError(f"Invalid model_name {model_name}")

    if drivers == "tiegcm2":
        u_ar_ds = model_object["u_ar_ds_tiegcm2"]
        datetimes_ds = model_object["datetimes_ds_tiegcm2"]
    else:
        u_ar_ds = model_object["u_ar_ds"]
        datetimes_ds = model_object["datetimes_ds"]

    # -------------------------------
    # 3) Convert drivers to 2D safely
    # -------------------------------
    u_ar_ds = np.asarray(u_ar_ds)

    if u_ar_ds.ndim == 1:
        u_ar_ds = u_ar_ds.reshape(1, -1)
    elif u_ar_ds.ndim == 2 and u_ar_ds.shape[0] <= kp_idx:
        u_ar_ds = u_ar_ds.T

    if u_ar_ds.shape[0] <= kp_idx:
        raise ValueError(
            f"u_ar_ds must have at least {kp_idx+1} drivers, got {u_ar_ds.shape}"
        )

    # -------------------------------
    # 4) Time slicing & interpolation
    # -------------------------------
    datetimes_ds = get_propagator_datetimes(datetimes_ds)
    mask = np.array([(start_date <= t <= stop_date) for t in datetimes_ds])
    u_ar_ds = u_ar_ds[:, mask]

    x_drivers_interp = interpolate_matrix_rows(
        u_ar_ds, time_conversion_factor)
    x_drivers_interp = np.atleast_2d(x_drivers_interp)

    # -------------------------------
    # 5) Discrete-time matrices
    # -------------------------------
    A_d = B_full[:, -n_components:]
    B_d = B_full[:, :-n_components]
    A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)
    A_d2_T = A_d2.T

    if model_name == "sindy-c":
        n_lags = len(ar_lags)

        ar_drivers_start = (
            B_full.shape[1]
            - n_components
            - max(ar_lags) * n_components
            - 3 * max(ar_lags)
        )

        ar_state_start = (
            B_full.shape[1]
            - n_components
            - n_lags * n_components
        )

        B_nonlinear = B_d2[:, :ar_drivers_start]

        B_ar_drivers_lst = [
            B_d2[:, ar_drivers_start + 3*i :
                     ar_drivers_start + 3*(i+1)]
            for i in range(n_lags)
        ]

        B_ar_state_lst = [
            B_d2[:, ar_state_start + i*n_components :
                     ar_state_start + (i+1)*n_components]
            for i in range(n_lags)
        ]
    else:
        B_nonlinear = B_d2
        B_ar_drivers_lst = None
        B_ar_state_lst = None
        ar_drivers_start = None

    # -------------------------------
    # 6) Initial condition history
    # -------------------------------
    ic_array_data = model_object["ic_array"]
    if hasattr(ic_array_data, "ic_array"):
        ic_array_data = ic_array_data.ic_array
    ic_df = pd.DataFrame(
        ic_array_data,
        columns=["f10", "kp"] +
        [f"z_{k+1:02d}" for k in range(n_components)]
    )

    n_ic_steps = min(max(ar_lags) + 1, u_ar_ds.shape[1])

    z_init = []
    for k in range(n_ic_steps):
        f10_val = u_ar_ds[f10_idx, k]
        kp_val = u_ar_ds[kp_idx, k]
        match = find_closest_match(ic_df, f10_val, kp_val)

        if match.empty:
            row = ic_df.iloc[0, 2:2+n_components].values.reshape(-1, 1)
        else:
            row = (
                match.iloc[2:2+n_components]
                if isinstance(match, pd.Series)
                else match.iloc[0, 2:2+n_components]
            ).values.reshape(-1, 1)

        z_init.append(row)

    z_hist = np.concatenate(z_init, axis=1)
    z_hist = u.normalize_with_dict(
        z_hist, Y_reg_norm_dict,
        model_object["normalization_method"]
    )
    # z_hist stays on the HOURLY grid, matching kalman_filter_logspace_ar. B_sindy's x_ar
    # block was fit on hourly lags, so the filter's AR taps must be an hour apart. This used
    # to be upsampled to the T2 grid by interpolate_matrix_rows, which made the five "lags"
    # 1-5 minutes apart while the coefficients expected 1-5 hours -- and, because the buffer
    # holds n_ic_steps columns and gains one per hour, left it contaminated for the first
    # ~5 hours of every window. On the Halloween IC the true hourly taps of mode 0 span
    # 173.1 / 203.9 / 183.5 / 180.6 / 205.6 / 173.6; the upsampled ones span
    # 179.9 / 178.7 / 177.4 / 176.2 / 174.9 / 173.6 -- the lag structure was flattened out.
    # Commit 90cf023 removed the upsampling from the single-satellite filter but not here.
    # The taps rotate once per hour in the loop below.

    normalization_method = model_object["normalization_method"]

    # Jacobian scale factor taking d/dz_norm to d/dz_raw for the measurement operator,
    # and the same factor used to report the filter output in raw coefficients. Only the
    # 'std' normalisation rescales; 'mean' and 'none' shift or do nothing.
    if normalization_method == "std":
        y_scale = Y_reg_norm_dict["x_std"].reshape(-1)[:n_components]
    else:
        y_scale = np.ones(n_components)

    # P0 and Q arrive in *raw* coefficient units; the loop runs normalised. See the
    # NORMALISATION note in the docstring.
    _sc = np.outer(y_scale, y_scale)
    P0 = np.asarray(P0, dtype=float) / _sc
    Q = np.asarray(Q, dtype=float) / _sc

    meas_model = nonlinear_meas if nonlinear_meas is not None else interpolator
    if meas_model is None:
        raise ValueError("provide either `interpolator` or `nonlinear_meas`")

    # -------------------------------
    # 7) KF setup & augmentation
    # -------------------------------
    # State is augmented to [z_k; z_{k-1}; ...; z_{k-p+1}] where p = max(ar_lags)
    p_lags = max(ar_lags) if model_name == "sindy-c" else 1
    n_aug  = n_components * p_lags

    N = int((stop_date - start_date).total_seconds() / T2)
    z_filt = np.zeros((n_components, N))
    
    # Augmented state and covariance
    X_aug  = np.zeros((n_aug, 1))
    P_aug  = np.zeros((n_aug, n_aug))
    # Initialize blocks with P0
    for i in range(p_lags):
        P_aug[i*n_components:(i+1)*n_components, i*n_components:(i+1)*n_components] = P0.copy()

    Q_aug  = np.zeros((n_aug, n_aug))
    Q_aug[:n_components, :n_components] = Q

    # Companion matrix A_aug
    A_aug = np.zeros((n_aug, n_aug))
    if model_name == "sindy-c":
        A_aug[:n_components, :n_components] = A_d2 + B_ar_state_lst[0]
        for i in range(1, p_lags):
            A_aug[:n_components, i*n_components:(i+1)*n_components] = B_ar_state_lst[i]
        for i in range(1, p_lags):
            A_aug[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = np.eye(n_components)
    else:
        A_aug[:n_components, :n_components] = A_d2
        if p_lags > 1:
            for i in range(1, p_lags):
                A_aug[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = np.eye(n_components)

    # Companion variant for steps that are not on an hour boundary. The AR taps are
    # hourly, so between boundaries they hold their own value instead of shifting down
    # the chain -- the covariance has to follow the same bookkeeping as the mean.
    A_hold = A_aug.copy()
    for i in range(1, p_lags):
        A_hold[i*n_components:(i+1)*n_components, (i-1)*n_components:i*n_components] = 0.0
        A_hold[i*n_components:(i+1)*n_components, i*n_components:(i+1)*n_components] = np.eye(n_components)

    # Filter steps per hour, i.e. how often the hourly AR taps rotate.
    steps_per_hour = max(int(round(time_conversion_factor)), 1)

    A_aug_T = A_aug.T
    I_aug   = np.eye(n_aug)
    P_filt  = []

    # Initialize augmented state from history
    for i in range(p_lags):
        X_aug[i*n_components:(i+1)*n_components, 0] = z_hist[:, -(i+1)]

    # Allocate z_prior for open-loop storage
    z_prior = np.zeros((n_components, N))

    # Per-step filter diagnostics, mirroring ark_filter_windowed_Nsat so both paths
    # report consistency the same way. NIS = innov^T S^-1 innov / n_meas; 1.0 is a
    # consistent filter, >1 means R too small or model error, <1 means R too large.
    sl_mode      = np.zeros(N, dtype=np.uint8)
    sl_n_meas    = np.zeros(N, dtype=np.int16)
    sl_trace_P   = np.full(N, np.nan)
    sl_diag_P    = np.full((N, n_components), np.nan)
    sl_innov_rms = np.full(N, np.nan)
    sl_NIS       = np.full(N, np.nan)

    # -------------------------------
    # 8) Prepare joint measurements
    # -------------------------------
    meas_df_assim_1 = meas_df_assim_1.copy()
    meas_df_assim_2 = meas_df_assim_2.copy()
    meas_df_assim_3 = meas_df_assim_3.copy()

    # Accept R as a 1-D vector of variances or a 2-D diagonal covariance matrix.
    # The 1-D form is preferred: at raw cadence the dense matrix is ~120k x 120k.
    R_assim_1 = _as_variance_vector(R_assim_1, 'champ')
    R_assim_2 = _as_variance_vector(R_assim_2, 'grace')
    R_assim_3 = _as_variance_vector(R_assim_3, 'goce')

    # Measurements and variances must line up row-for-row. Guaranteed when
    # decimate_to_filter_grid() is applied per satellite before the variances are
    # computed; fail loudly rather than silently reusing the last variance.
    _check_meas_variance_lengths(
        [('champ', meas_df_assim_1, R_assim_1),
         ('grace', meas_df_assim_2, R_assim_2),
         ('goce', meas_df_assim_3, R_assim_3)])

    meas_df_assim_1["orig_idx"] = np.arange(len(meas_df_assim_1))
    meas_df_assim_2["orig_idx"] = np.arange(len(meas_df_assim_2))
    meas_df_assim_3["orig_idx"] = np.arange(len(meas_df_assim_3))

    meas_df_assim_1["satellite"] = "champ"
    meas_df_assim_2["satellite"] = "grace"
    meas_df_assim_3["satellite"] = "goce"

    meas_df_joint = pd.concat(
        [meas_df_assim_1, meas_df_assim_2, meas_df_assim_3],
        ignore_index=True
    )

    meas_times = meas_df_joint["timestamp"].values.astype("datetime64[ns]")
    meas_k = ((meas_times - np.datetime64(start_date)) /
              np.timedelta64(int(T2), "s")).astype(int)

    mask = (meas_k >= 0) & (meas_k < N)
    meas_df_joint = meas_df_joint[mask].copy()
    meas_df_joint["k"] = meas_k[mask]
    meas_df_joint.reset_index(drop=True, inplace=True)

    n_meas = len(meas_df_joint)
    R_joint = np.zeros(n_meas)

    champ_mask = meas_df_joint.satellite == "champ"
    grace_mask = meas_df_joint.satellite == "grace"
    goce_mask = meas_df_joint.satellite == "goce"

    R_joint[champ_mask] = R_assim_1[
        meas_df_joint.loc[champ_mask, "orig_idx"].values]
    R_joint[grace_mask] = R_assim_2[
        meas_df_joint.loc[grace_mask, "orig_idx"].values]
    R_joint[goce_mask] = R_assim_3[
        meas_df_joint.loc[goce_mask, "orig_idx"].values]

    meas_groups = meas_df_joint.groupby("k", sort=False)

    if model_name == "sindy-c":
        model_params_dict = {
            "functions": mat_to_lambda_dict(
                model_object["model_params"].functions),
            "shift": model_object["model_params"].shift,
            "dof": model_object["model_params"].dof,
        }
        input_features = model_object["input_features"]

    # Precompute Jacobian cache for sindy-c (called once before KF loop)
    if model_name == "sindy-c":
        jac_cache = sindy_jacobian_cache_build(
            model_params=model_object["model_params"],
            f10_idx=f10_idx,
            kp_idx=kp_idx,
            n_components=n_components,
            ar_drivers_start=ar_drivers_start,
            X_reg_norm_dict=X_reg_norm_dict,
            Y_reg_norm_dict=Y_reg_norm_dict,
        )
    else:
        jac_cache = None

    # -------------------------------
    # 9) KF loop
    # -------------------------------
    for k in range(N):
        if k % 1440 == 0:
            print(f"[ KF step {k+1}/{N} ]")
        # x_drivers_interp was upsampled from hourly to the T2 grid by
        # interpolate_matrix_rows(u_ar_ds, time_conversion_factor), so its columns are
        # already one per filter step: the driver index is k, not k*time_conversion_factor.
        # The old expression overshot by that factor and was silently pinned to the last
        # column by the clamp inside sindy_step, which froze the drivers a few minutes
        # into every run. `propagate` uses the same convention.
        interp_idx = k

        # The AR taps live on the hourly grid; steps_per_hour filter steps make one hour.
        hour_boundary = (k > 0) and (k % steps_per_hour == 0)

        # Check if we have measurements at this timestep
        has_measurement = k in meas_groups.groups

        # Reconstruct chronological history for the model step [oldest -> newest]
        z_hist_for_step = np.fliplr(X_aug.reshape(p_lags, n_components).T)

        # ---- prediction (unconditional — always compute)
        if model_name == "sindy-c":
            # Use the same stepper `propagate` and the single-satellite filter use, rather
            # than sindy_step. On identical inputs sindy_step's one-step error is 10-5000x
            # larger, from two defects it does not share with sindy_loop_optimized:
            #   1. X_sindy_norm_dict holds 250 entries against 255 library features (the
            #      bias and 4 time rows are not normalised). sindy_loop_optimized offsets
            #      its slice by delta_idx=5 to account for that; sindy_step does not, so
            #      it normalises the AR drivers with the wrong 15 entries.
            #   2. Rows kp_idx+1 .. kp_idx+3*L of u_ar_ds already hold the AR-lagged
            #      [f107, f107a, kp] stack exactly. sindy_loop_optimized reads them;
            #      sindy_step rebuilds them from rows [f10_idx, f10_idx+1, kp_idx] using
            #      interp_idx shifts taken at the filter's T2 resolution rather than the
            #      model's hourly lag resolution.
            #
            # Commit 90cf023 made this swap in kalman_filter_logspace_ar but not in the
            # block-update filters, so the multi-satellite runs kept the old stepper.
            #
            # time_conversion_factor is passed as 1 so lag_offsets stay at consecutive
            # columns, matching the augmented state's existing one-step-shift structure.
            # z_hist is used rather than z_hist_for_step because it carries
            # max(ar_lags)+1 columns; the 5-column z_hist_for_step would make the longest
            # lag index wrap to -1.
            z_pred = sindy_loop_optimized(
                x_drivers_interp,
                X_reg_norm_dict,
                Y_reg_norm_dict,
                A_d2,
                B_ar_drivers_lst,
                B_ar_state_lst,
                B_nonlinear,
                z_hist,
                ar_drivers_start,
                interp_idx,
                interp_idx + 1,
                ar_lags,
                1,
                f10_idx,
                model_object["normalization_method"],
                {
                    "functions": mat_to_lambda_dict(model_object["model_params"].functions),
                    "shift": model_object["model_params"].shift,
                    "dof": model_object["model_params"].dof
                },
                model_object["input_features"],
                kp_idx=kp_idx,
            )[:, -1:]
        else:
            z_pred = dmd_step(
                z_hist_for_step,
                x_drivers_interp,
                A_d2,
                B_nonlinear,
                interp_idx,
                f10_idx,
                kp_idx,
                X_reg_norm_dict,
                model_object["normalization_method"],
                n_components,
            )

        # Augmented prediction. The lag taps are hourly, so they rotate only when a whole
        # hour of filter steps has elapsed; in between the older taps hold and only the
        # leading block advances.
        X_aug_pred = np.zeros_like(X_aug)
        X_aug_pred[:n_components, 0] = z_pred.flatten()
        if hour_boundary:
            X_aug_pred[n_components:, 0] = X_aug[:-n_components, 0]
        else:
            X_aug_pred[n_components:, 0] = X_aug[n_components:, 0]

        # Propagate covariance through the companion matrix -- the shifting variant on an
        # hour boundary, the holding variant otherwise, matching the mean update above.
        A_step = A_aug if hour_boundary else A_hold
        if model_name == "sindy-c":
            # EKF: linearize F_k at the prior mean ẑ_{k|k}, i.e. the input state to the
            # dynamics step, not the propagated state ẑ_{k+1|k}.
            F_jac = sindy_process_jacobian(
                z_k_norm=X_aug[:n_components, 0:1],
                x_drivers_interp=x_drivers_interp,
                interp_idx=interp_idx,
                A_aug=A_step,
                B_nonlinear=B_nonlinear,
                ar_drivers_start=ar_drivers_start,
                n_components=n_components,
                f10_idx=f10_idx,
                kp_idx=kp_idx,
                model_params=model_object["model_params"],
                X_reg_norm_dict=X_reg_norm_dict,
                Y_reg_norm_dict=Y_reg_norm_dict,
                normalization_method=model_object["normalization_method"],
                jac_cache=jac_cache,
            )
            F_jac_T = F_jac.T
            P_aug = F_jac @ P_aug @ F_jac_T + Q_aug
        else:
            # dmd-c: nonlinear input excludes z_k, so A_step is already the exact Jacobian
            P_aug = A_step @ P_aug @ A_step.T + Q_aug

        if has_measurement:
            # ---- Block measurement update
            rows_k = meas_groups.get_group(k)
            n_meas_k = len(rows_k)

            # Exactly one observation per satellite per step. More than one means the
            # frames were not decimated onto the T2 grid, which would deflate the
            # effective measurement variance by the oversampling ratio.
            if rows_k.groupby('satellite').size().max() != 1:
                # A raise, not an assert: `python -O` strips asserts, and silently
                # over-counting observations is the defect this guards against.
                raise ValueError(
                    f"step {k}: {n_meas_k} observations across "
                    f"{rows_k['satellite'].nunique()} satellites -- apply "
                    "decimate_to_filter_grid() before computing R")

            llas = rows_k[["latitude", "local_time", "altitude"]].values
            H_block_small, innov = block_measurement(
                meas_model, X_aug_pred, n_components, llas,
                rows_k["log_density"].to_numpy(),
                Y_reg_norm_dict, normalization_method, y_scale)

            # Augmented H: [H, 0, ..., 0]
            H_block = np.zeros((n_meas_k, n_aug))
            H_block[:, :n_components] = H_block_small

            # Compute innovation
            # innov comes from block_measurement, already y - h(z_pred).

            # Kalman gain (block update)
            S = H_block @ P_aug @ H_block.T
            S[np.diag_indices_from(S)] += R_joint[rows_k.index.values]

            K = P_aug @ H_block.T @ np.linalg.solve(S, np.eye(n_meas_k))

            # Update state and covariance
            X_aug = X_aug_pred + K @ innov
            # Joseph form
            ImKH = I_aug - K @ H_block
            P_aug = ImKH @ P_aug @ ImKH.T + K @ (R_joint[rows_k.index.values][:, None] * K.T)
            P_aug = 0.5 * (P_aug + P_aug.T)

            # ---- innovation diagnostics for this step
            sl_mode[k]      = 1
            sl_n_meas[k]    = n_meas_k
            sl_innov_rms[k] = float(np.sqrt(np.mean(innov ** 2)))
            sl_NIS[k]       = float(np.squeeze(
                innov.T @ np.linalg.solve(S, innov))) / n_meas_k

        else:
            # No measurement: use prediction as state
            X_aug = X_aug_pred

        # ---- store prior and update AR history
        # z_curr stays normalized: that is the space z_hist and sindy_step work in.
        # z_hist is the hourly AR buffer feeding the next prediction, so append one column
        # per hour rather than one per filter step. In between, its newest column tracks
        # the current estimate.
        z_curr = X_aug[:n_components, :1]
        if hour_boundary:
            z_hist = np.concatenate([z_hist, z_curr], axis=1)[:, -n_ic_steps:]
        else:
            z_hist[:, -1:] = z_curr

        # Outputs are reported in RAW coefficients, matching `propagate`'s z_openloop and
        # what reconstruction (rho = 10**(H z + mu)) expects. The loop itself is
        # normalized throughout -- see the NORMALISATION note in the docstring.
        z_prior[:, k] = denormalize(
            z_pred.reshape(-1, 1), Y_reg_norm_dict, normalization_method).ravel()
        z_filt[:, k] = denormalize(
            z_curr, Y_reg_norm_dict, normalization_method).ravel()

        # Covariance likewise: P_raw = diag(x_std) P_norm diag(x_std). trace_P and diag_P
        # are reported raw too, so the released steplog is in the same units as z_filt.
        P_z = P_aug[:n_components, :n_components] * np.outer(y_scale, y_scale)
        P_filt.append(P_z.copy())
        _dp = np.diag(P_z)
        sl_trace_P[k] = float(np.sum(_dp))
        sl_diag_P[k]  = _dp


    step_arrays = {
        "mode":       sl_mode,
        "n_meas":     sl_n_meas,
        "trace_P":    sl_trace_P,
        "diag_P":     sl_diag_P,
        "innov_rms":  sl_innov_rms,
        "NIS":        sl_NIS,
    }
    return {
        "openloop": z_openloop,
        "filtered": z_filt,
        "P_filt": P_filt,
        "step_arrays": step_arrays,
    }