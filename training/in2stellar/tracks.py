import os
import warnings
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .stellar import Interstellar

# Some global parameters, until I have a better way of configuring these
TEFF = 'effective_T'
LOGG = 'log_g'
RADIAL_MODE = 'nu_0'
EVOL_STAGE = 'evol_stage'
EVOL_STAGES = {'prems': 0, 'dwarf': 1, 'giant': 2, 'CHeB': 3}
DATA_COLS = ['initial_mass', 'initial_Yinit', 'initial_feh', 'initial_MLT', EVOL_STAGE,
             'star_mass', 'star_age', 'frac_age', TEFF, LOGG, 'luminosity', 'radius', 'star_feh',
             'delta_nu_fit']
FILE_NAME = 'file_name'
SUFFIX = 'csv'

# def _validate_dataframe(name='dataframe'):
#     '''Decorator for validating and converting to DataFrame.'''
def _validate_dataframe(func):
    @functools.wraps(func)
    def wrapper(_, df, name='dataframe'):
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except ValueError as verr:
                if str(verr) == 'DataFrame constructor not properly called!':
                    msg = f'Failed to convert `{name}` to `pandas.DataFrame`.'
                    raise TypeError(msg)
                else:
                    raise verr
                
        return func(_, df)
    
    return wrapper
    
    # return decorator

def _get_dataframe(func):
    @functools.wraps(func)
    def wrapper(_, columns=None):
        df = func(_, columns=columns)
        if columns is None:
            return df.copy()
        else:
            return df.loc[:, columns]
    return wrapper

# def _sample_dataframe(func):
#     '''Samples a pandas.DataFrame with an additional `inplace` option.'''
#     @functools.wraps(func)
#     def wrapper(_, number=None, fraction=None, inplace=False, **kwargs):
#         df = func(_, number=number, fraction=fraction, inplace=inplace, **kwargs)
#         try:
#             if inplace:
#                 df = df.sample(n=number, frac=fraction, **kwargs)
#                 sample = None
#             else:
#                 sample = df.sample(n=number, frac=fraction, **kwargs)
#         except ValueError as verr:
#             if str(verr) == 'Please enter a value for `frac` OR `n`, not both':
#                 msg = 'Please enter a value for `number` OR `fraction`, not both'
#                 raise ValueError(msg)
#             else:
#                 raise verr
        
#         return sample

#     return wrapper

# def _select_dataframe(func):
#     def wrapper(_, df, condition, inplace=False):
#         ''' Select from dataframe given some `condition` '''
#         df = func(_, df, condition, inplace=False)
#         if inplace:
#             df = df.loc[condition]
#             selection = None
#         else:
#             selection = df.loc[condition]
        
#         return selection     
    
#     return wrapper

class Tracks(Interstellar):
    '''
    Class containing information about a set of MESA stellar evolutionary tracks.
    
    Parameters
    ----------
    path : str, optional
        The path to the directory containing simple MESA track data files. Each file name should
        contain metadata about the track (e.g. initial mass and metallicity) separated in the following
        fasion <name><value>...<name><value>.csv where <name> and <value are alphabetical and
        numerical respectively. If path is not required, set to None (default).

    metadata : dict-like or pandas.DataFrame, optional
        An iterable or dict-like object containing metadata for the track data found at `path`.
        If not a pandas.DataFrame, an attempt at converting it to one is made. If None (default)
        the Track metadata is initialised as an empty pandas.DataFrame.

    data : dict-like or pandas.DataFrame, optional
        An iterable or dict-like object containing data for each point in the grid of tracks.
        If not a pandas.DataFrame, an attempt at converting it to one is made. If None (default)
        the Track data is initialised as an empty pandas.DataFrame.

    Examples
    --------
    Constructing Tracks and compiling metadata for files in path '/path/to/tracks/data'
    
    >>> tracks = Tracks(path='/path/to/tracks/data')
    >>> tracks.get_path()
    '/path/to/tracks/data'
    >>> tracks.compile_metadata()  # <--- Scans `path` for file name metadata
    >>> tracks.get_metadata()
        c1  c2
    0   0   0.0
    1   1   1.0
    2   2   2.0
    
    '''

    def __init__(self, path=None, name=None, metadata=None, data=None):

        # We don't want to make directories if path doesn't exist
        super().__init__(path=path, name=name, make_dirs=False)

        self._metadata = self._validate_metadata(metadata)
        self._data = self._validate_data(data)
        self.norm_factors = pd.DataFrame(columns=['column', 'norm_column', 'loc', 'scale'])  # , index=pd.Index([], name='column'))
        # self._summary = self.summary()

    def __repr__(self):
        return f'Tracks(path={self.path}, metadata={self._metadata}, data={self._data})'
    
    def __str__(self):
        return f'path\n----\n{self.path}\n\nmetadata\n--------\n{self._metadata}\n\ndata\n----\n{self._data}'

    def __len__(self):
        return len(self._data)
    
    # def _validate_path(self, path):
    #     if type(path) is not str and path is not None:
    #         msg = 'Argument `path` must be of type `str`'
    #         raise TypeError(msg)
        
    #     return path
                
    @_validate_dataframe
    def _validate_metadata(self, metadata, name='metadata'):
        '''Validate metadata. Check for "file_name" column.'''
        if FILE_NAME not in metadata.columns and len(metadata) != 0:
            msg = f'Column "file_name" not in `metadata.columns`.'
            raise ValueError(msg)
        return metadata

    @_validate_dataframe
    def _validate_data(self, data, name='data'):
        '''Validate data.'''
        return data
    
    def set_path(self, path):
        '''
        Sets the path to tracks data files.
        
        Parametes
        ---------
        path : str
            Path to tracks data files.

        '''
        self.path = self._validate_path(path)
    
    def get_path(self):
        '''Returns path to tracks data files.'''
        return self.path
    
    def scan_filenames(self):
        '''Returns a list of filenames under the Tracks data path.'''
        if self.path is None:
            msg = 'No path specified, current working directory assumed. Otherwise, set_path and retry.'
            warnings.warn(msg, UserWarning)
        
        with os.scandir(self.path) as it:
            file_names = [entry.name for entry in it if entry.is_file() and \
                          entry.name.endswith(SUFFIX) and not entry.name.startswith('.')]
        
        return file_names

    def _separate_filenames(self, column_regex=r'[0-9\.\+\-]+', value_regex='[a-zA-z]+'):
        '''
        Returns the column names and values from a list of file names by using regular
        expressions which exclude certain character types from the filename to highlight
        the column names and values.
        
        TODO: Allow for separator characters in file names.
        '''
        file_prefix = self._metadata[FILE_NAME].str.replace('.csv', '', regex=False)

        cols_vals = []
        for re in [column_regex, value_regex]:
            cols_vals.append(file_prefix.str.replace(re, ' ').str.split())
            cols_vals[-1] = np.stack(cols_vals[-1].to_numpy())

        return cols_vals

    def _filenames_to_metadata(self, **kwargs):
        '''Gets metadata from filenames.'''
        columns, values = self._separate_filenames(**kwargs)

        for i in range(columns.shape[1]):
            # Check that the each set of column names are all the same
            unique_column_name = np.unique(columns[:,i])
            if len(unique_column_name) == 1:
                self._metadata[unique_column_name[0]] = pd.to_numeric(values[:,i])
            else:
                raise Exception('Track names do not have unique grid point names.')
        
    def compile_metadata(self):
        '''Compiles the metadata from Tracks data path directory'''
        if len(self._metadata != 0):
            self._metadata = pd.DataFrame()
        self._metadata[FILE_NAME] = self.scan_filenames()
        self._filenames_to_metadata()
    
    def set_metadata(self, metadata):
        '''
        Sets the metadata.
        
        Parametes
        ---------
        metadata : dict-like or pandas.DataFrame
            An iterable or dict-like object containing metadata for the track data found at `path`.
            If not a pandas.DataFrame, an attempt at converting it to one is made. If None,
            the Track metadata is set as an empty pandas.DataFrame.

        '''
        self._metadata = self._validate_metadata(metadata)
    
    @_get_dataframe 
    def get_metadata(self, columns=None):
        '''Returns the Tracks metadata.'''
        return self._metadata

    def compile_data(self, usecols=None):
        '''
        Loads data from the file_name column of Tracks metadata under the Tracks path

        Parameters
        ----------
        usecols : list of str or str, optional
            Default is None, columns from the config.yml file are used. Otherwise, a list of
            strings corresponding to the columns to load. If 'all' then all will be loaded.
        '''

        if usecols is None:
            usecols = DATA_COLS
        elif usecols == 'all':
            usecols = None
        elif not isinstance(usecols, (list, tuple)):
            usecols = list(usecols)
        
        # self.sample_metadata()
        paths = self._metadata[FILE_NAME].apply(lambda x: os.path.join(self.path, x))
        data = paths.apply(pd.read_csv, usecols=usecols)

        self._data = pd.concat(data.to_list(), ignore_index=True)

    def set_data(self, data):
        self._data = self._validate_data(data)
    
    @_get_dataframe
    def get_data(self, columns=None):
        '''Get data. Specify a list of, or individual columns to return (optional).'''
        # if columns is not None:
        #     return self._data.loc[:, columns]
        # else:
        return self._data
   
    # def _sample_dataframe(self, df, number=None, fraction=None, inplace=False, **kwargs):
    #     '''Samples a pandas.DataFrame with an additiona `inplace` option.'''
    #     try:
    #         if inplace:
    #             df = df.sample(n=number, frac=fraction, **kwargs)
    #             sample = None
    #         else:
    #             sample = df.sample(n=number, frac=fraction, **kwargs)
    #     except ValueError as verr:
    #         if str(verr) == 'Please enter a value for `frac` OR `n`, not both':
    #             msg = 'Please enter a value for `number` OR `fraction`, not both'
    #             raise ValueError(msg)
    #         else:
    #             raise verr
        
    #     return sample

    def _sample_dataframe(self, df, number=None, fraction=None, **kwargs):
        try:
            sample = df.sample(n=number, frac=fraction, **kwargs)
        except ValueError as verr:
            if str(verr) == 'Please enter a value for `frac` OR `n`, not both':
                msg = 'Please enter a value for `number` OR `fraction`, not both'
                raise ValueError(msg)
            else:
                raise verr
        
        return sample

    def sample_metadata(self, number=None, fraction=None, columns=None, inplace=False, **kwargs):
        '''
        Randomly sample either some `number` or `fraction of Tracks metadata.
        If neither `number` or `fraction` is specified, default sample size is 1.
        
        Parameters
        ----------
        number : int, optional
        
        fraction : float between 0 and 1, optional
        
        inplace : bool, optional
        
        **kwargs :
            Keyword arguments to pass to pandas.DataFrame.sample.

        Returns
        -------
        sample : pandas.DataFrame or None
        
        Raises
        ------
        ValueError :
            If both `number` and `fraction` are given.

        '''
        sample = self._sample_dataframe(self.get_metadata(columns=columns),
                                        fraction=fraction, number=number)
        if inplace:
            self._metadata = sample
        else:
            return sample
        # return self._sample_dataframe(self._metadata, *args, **kwargs)

    def sample_data(self, number=None, fraction=None, columns=None, inplace=False, **kwargs):
        '''
        Randomly sample either some `number` or `fraction of Tracks data.
        If neither `number` or `fraction` is specified, default sample size is 1.
        
        Parameters
        ----------
        number : int, optional
        
        fraction : float between 0 and 1, optional
        
        inplace : bool, optional
        
        **kwargs :
            Keyword arguments to pass to pandas.DataFrame.sample.

        Returns
        -------
        sample : pandas.DataFrame or None
        
        Raises
        ------
        ValueError :
            If both `number` and `fraction` are given.

        '''
        sample = self._sample_dataframe(self.get_data(columns=columns),
                                        fraction=fraction, number=number)
        if inplace:
            self._data = sample
        else:
            return sample
        # return self._sample_dataframe(self._data, *args, **kwargs)     

    def _select_dataframe(self, df, condition):
        ''' Select from dataframe given some `condition` '''
        return df.loc[condition]  
    
    def select_metadata(self, condition, inplace=False):
        '''Select from metadata given some `condition`.
        
        Parameters
        ----------
        condition : boolean mappable
            A boolean mappable object which is used to select from metadata.
        
        inplace : bool, optional
            Whether assign the selection to the Track object (True) and return None,
            or return the selection as a separate DataFrame (False, default).
        
        Returns
        -------
        selection : pandas.DataFrame or None
        
        '''
        # return self._select_dataframe(self._metadata,  *args, **kwargs)
        if inplace:
            self._metadata = self._select_dataframe(self._metadata, condition)
        else:
            return self._select_dataframe(self._metadata, condition)
    
    def select_data(self, condition, inplace=False):
        '''Select from data given some `condition`.
        
        Parameters
        ----------
        condition : boolean mappable
            A boolean mappable object which is used to select from data.
        
        inplace : bool, optional
            Whether assign the selection to the Track object (True) and return None,
            or return the selection as a separate DataFrame (False, default).
        
        Returns
        -------
        selection : pandas.DataFrame or None
        
        '''
        # return self._select_dataframe(self._data,  *args, **kwargs)
        if inplace:
            self._data = self._select_dataframe(self._data, condition)
        else:
            return self._select_dataframe(self._data, condition)
    
    def clean_data(self):
        '''
        TODO: Function which removes NaN values and "cleans" self._data
        '''
        pass

    def log10(self, columns=None, inplace=False):
        '''Takes base-10 logarithm of tracks data, columns optional.'''
        log_df = np.log10(self.get_data(columns))
        log_cols = [f'log_{col}' for col in log_df.columns]
        if inplace:
            self._data[log_cols] = log_df
        else:
            return log_df.rename(columns={c: lc for c, lc in zip(log_df.columns, log_cols)})
    
    def normalize(self, loc=None, scale=None, columns=None, suffix='norm', inplace=False, summary_kw={}):
        ''' Normalize columns in track data by a chosen statistic.
        
        Parameter loc is one of 'mean', '50%', 'min', 'max' or others defined in summary_kw.
        
        Warning, if you change the data after normalizing then it will cause
        a loss in the summary stats which provided the locations, so this function
        returns the normalized columns and the stats which produced them.
        
        Loc and scale are 0 and 1 respectively if None.
        
        Summary keywords lets you specify, e.g. the percentiles to use, which may
        be used in the normalization.
        
        Parameters
        ----------
        loc : str, array-like or dict-like, optional
        
        scale : str, array-like or dict-like, optional
        
        columns : array-like, optional
        
        suffix : str, optional
        
        inplace : bool, optional
        
        summary_kw : dict, optional
        
        Returns
        -------
        factors, norm_df:
            If inplace is True, nothing is returned

        Examples
        --------
        TODO: make better examples with test data.
        >>> tracks = Tracks()
        >>> tracks.load(path, metadata_filename, data_filenmae)
        >>> tracks.normalize(self, loc='mean', scale='std', column['a', 'b'])
            a_norm  b_norm
        0   1.23    4.56
        1   7.89    0.12
        '''
        if columns is None:
            if isinstance(loc, (dict, pd.Series)) and isinstance(scale, (dict, pd.Series)):
                # Max keys chosen
                columns = loc.keys() if len(loc.keys()) > len(scale.keys()) else scale.keys()
            elif isinstance(loc, (dict, pd.Series)):
                    columns = loc.keys()
            elif isinstance(scale, (dict, pd.Series)):
                    columns = scale.keys()                    
            else:
                columns = self._data.columns
            
        stats = self.summary(columns=columns)  # So a summary is created immediately
        cols = ['loc', 'scale']
        values = [loc, scale]
        factors = pd.DataFrame(columns=['norm_column']+cols,
                               index=columns
                               )
        # factors.index.rename('column', inplace=True)
        # factors['column'] = columns
        factors['loc'], factors['scale'] = [0.0, 1.0]
        
        for col, val in zip(cols, values):
            if isinstance(val, str):
                factors[col] = stats.loc[val]
            elif isinstance(val, (dict, pd.Series)):
                factors.loc[val.keys(), col] = val
            elif isinstance(val, (list, np.ndarray)):
                factors[col] = val
            elif val is not None:
                msg = f'Invalid type for {col}. Must be str, dict-like or array-like.'
                raise TypeError(msg)
        
        norm_df = (self.get_data(columns) - factors['loc']) / factors['scale']
        norm_cols = [f'{col}_{suffix}' for col in norm_df.columns]
        new_cols = {c: lc for c, lc in zip(norm_df.columns, norm_cols)}
        factors['norm_column'] = norm_cols
        factors.reset_index(inplace=True)
        factors.rename(columns={'index': 'column'}, inplace=True)
        # factors = factors.rename(index=new_cols)
        norm_df = norm_df.rename(columns=new_cols)
        
        if inplace:
            self._data[norm_cols] = norm_df
            self.norm_factors = self.norm_factors.append(factors, ignore_index=True)
            self.norm_factors = self.norm_factors.loc[~self.norm_factors['norm_column'].duplicated(keep='last')]
            self.norm_factors.reset_index(inplace=True, drop=True)
            # return factors
        else:
            return factors, norm_df

    def renormalize(self, df, factors=None):
        '''Renormalises columns in `df` according the the location and scale in factors.
        If `factors` is None (default) then the track's factors are used.

        Returns a dataframe containing the renormalised data
        '''
        if factors is None:
            factors = self.norm_factors.copy()  # TODO: get_factors function to specify columns.

        f = factors[['loc', 'scale']].to_numpy()
        df = df.loc[:, factors['norm_column']] * f[:, 1] + f[:, 0]
        new_cols = {c.iloc[0]: c.iloc[1] for i, c in factors[['norm_column','column']].iterrows()}
        df.rename(columns=new_cols, inplace=True)

        return df
        
    def select_evolution(self, stages, inplace=False):
        '''
        Parameters
        ----------
        stages : str or list of str
            At least one of 'prems', 'dwarf', 'giant' and 'CHeB'
        
        inplace : bool, optional
            If True, sets Track data to selection. If False (default) returns
            the selection.
        
        Returns
        -------
        
        data : pandas.DataFrame or None
            Returns data corresponding to `stages` or None.
        '''
        stage_ids = []
        for stage in stages:
            stage_ids.append(EVOL_STAGES[stage])

        condition = self._data[EVOL_STAGE].isin(stage_ids)
        return self.select_data(condition, inplace=inplace)

    def _check_new_column_name(self, name, attr):
        if (attr == 'data' and name in self._data.columns) or \
           (attr == 'metadata' and name in self._metadata.columns):
            msg = f'New column name "{name}" already exists in {attr}."'
            raise ValueError(msg)
            
    def calculate_numax(self, name, scale_teff=5777.0, scale_logg=4.44, scale_numax=3100.0):
        '''
        calculate numax with the scaling relation:
        nu_max = (teff/scale_teff)^-0.5 * (10**logg/10**scale_logg)*scale_numax (Brown 1991)

        Parameters
        ----------
        name : str
            The name of the new column to be made, must not already exist in tracks data.
        
        scale_teff : float, optional
            Effective temperature to scale by. Default is the solar value, 5777 K.
            
        scale_logg : float, optional
            Log surface gravity to scale by. Default is the solar value, 4.44 dex.
            
        scale_numax : float, optional
            Frequency at maximum power to scale by. Defualt is the solar value, 3100.0 muHz (TODO: REFERENCE)
        
        Raises
        ------
        ValueError
            If `name` already exists in Tracks data.

        '''
        self._check_new_column_name(name, 'data')

        self._data[new_column_name] = (self._data[TEFF]/scale_teff)**(-0.5) * \
                                      10**(self._data[LOGG] - scale_logg) * scale_numax

    def _mode_fit(self, row, mode_column_names, nu_max_column_name, sigma_of_weights_factor, mode_n_orders):
        '''
        Tanda Li
        The function is called in '_calculate_delta_nu_with_radial_modes'
        to calculate delta_nu by fiting individual modes with specified weights.
        Weights follow a Gaussian distribution. The centre is nu_max from scalling relation
        and the sigma is a guess value of delta_nu (the two modes around nu_max) times a user-specified
        factor --- sigma_of_weights_factor.
        '''
        usedmodes = row[mode_column_names]
        nsort = np.argsort(abs(usedmodes - row[nu_max_column_name]))
        guess_delta_nu = abs(usedmodes[nsort[1]] - usedmodes[nsort[0]])
        sigma = guess_delta_nu*sigma_of_weights_factor
        if sigma == 0:
            return np.nan
        weights = np.exp(-(usedmodes - row[nu_max_column_name])**2 / (2 * sigma**2))
        try:
            fit = np.polyfit(mode_n_orders, usedmodes, 1, w=weights)
            return fit[0]
        except:
            warnings.warn('Fit failed.', UserWarning)
            return np.nan

    def _calculate_delta_nu_with_radial_modes(self, name, numax_column=None, sigma_of_weights_factor=1.5,
                                              max_n_orders=40):
        '''
        Tanda Li
        
        Calculate delta_nu with radial modes.
        
        Use sigma_of_weights_factor to adjust the weights around nu_max.
        
        Weight of each modes follow a Gaussian function, where nu = nu_max
        and sigma = the space between the two modes around nu_max.")
        '''
        if type(max_n_orders) is not int:
            msg = 'Argument `max_n_orders` is not of type int.'
            raise TypeError(msg)
        
        mode_n_orders = np.arange(1,max_n_orders+1)
        mode_columns = [f'{RADIAL_MODE}_{i}' for i in mode_n_orders]
        
        mode_fit_args = (mode_columns, numax_column, sigma_of_weights_factor, mode_n_orders)
        self._data[new_column_name] = self._data.apply(self._mode_fit, axis=1, args=mode_fit_args)

    def calculate_delta_nu(self, name, numax_column=None, method='mode fit', method_kw={}):
        ''' General function to calculate delta_nu given a method (currently only 'mode fit').'''
        self._check_new_column_name(name, 'data')

        if method == 'mode fit':
            self._calculate_delta_nu_with_radial_modes(name, numax_column, **method_kw)
        else:
            raise ValueError('Invalid method chosen.')            
    
    def save(self, path, float_format=None, summary_kw={}):
        '''Save the Tracks data and metadata and a summary of track data.'''
        path = self._validate_path(path)
        
        # metadata_filepath = os.path.join(path, f'{self.name}_metadata.csv')
        # data_filepath = os.path.join(path, f'{self.name}_data.csv')
        # summary_filepath = os.path.join(path, f'{self.name}_summary.csv')
        metadata_filepath = os.path.join(path, self._make_filename('metadata', 'csv'))
        data_filepath = os.path.join(path, self._make_filename('data', 'csv'))
        norm_filepath = os.path.join(path, self._make_filename('norm', 'csv'))
        # summary_filepath = self._make_filepath(self._make_filename('summary', 'csv'))
        
        self._metadata.to_csv(metadata_filepath, index=False, float_format=float_format)
        self._data.to_csv(data_filepath, index=False, float_format=float_format)
        self.norm_factors.to_csv(norm_filepath, index=False, float_format=float_format)
        # self.summary(**summary_kw).to_csv(summary_filepath, float_format=float_format)
    
    def load(self, path):
        '''Load tracks from data csv and metadata csv.'''
        
        metadata_filename = self._make_filename('metadata', 'csv')
        data_filename = self._make_filename('data', 'csv')
        norm_filename = self._make_filename('norm', 'csv')
        
        try:
            metadata = pd.read_csv(os.path.join(path, metadata_filename))
            self.set_metadata(metadata)
        except FileNotFoundError:
            msg = f'File "{metadata_filename}" not found at "{path}"'
            warnings.warn(msg, UserWarning)

        try:
            data = pd.read_csv(os.path.join(path, data_filename))
            self.set_data(data)
        except FileNotFoundError:
            msg = f'File "{data_filename}" not found at "{path}"'
            warnings.warn(msg, UserWarning)
        
        try:
            norm_fac = pd.read_csv(os.path.join(path, norm_filename)) 
            self.norm_factors = norm_fac  # TODO: set factors and validate factors function
        except FileNotFoundError:
            msg = f'File "{norm_filename}" not found at "{path}"'
            warnings.warn(msg, UserWarning)

    def summary(self, columns=None, percentiles=[0.5], **kwargs):
        '''Returns a dataframe summarising the Tracks data'''
        summary = pd.DataFrame() if len(self._data) == 0 else self._data.describe(percentiles=percentiles, **kwargs)
        if columns is None:
            return summary.copy()
        else:
            return summary.loc[:, columns]
