import os
import warnings
import functools
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import theano.tensor as T

# Activation functions for theano tensor
T_ACT = {
    'sigmoid': T.nnet.sigmoid,
    'tanh': T.tanh,
    'relu': T.nnet.relu,
    'elu': T.nnet.elu,
}

SIMPLE_COLS = ['name', 'depth', 'width', 'activation', 'batch_size', 'regularization_l2',
               'model_filename', 'history_filename']

# def _return_filename(save_func):
#     @functools.wraps(save_func)
#     def wrapper(_, return_filename=True, **kwargs):
#         filename = save_func(_, return_filename=return_filename, **kwargs)
#         if return_filename:
#             return filename
    
#     return wrapper

def set_seed(seed):
    ''' Set the seed '''
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
class Interstellar:
    '''
    Base class for the Interstellar module.
    
    TODO: Is params even needed, or should I just get rid? Depends how useful it is for
    individual networks?
    
    Parameters
    ----------
    path : str, optional
        Path to save. Defaults to ''.
    
    name : str, optional
        Object name. Defaults to 'interstellar'.
    
    **param_dict :
        DEPRECIATED. Arbitrary keyword arguments to be stored in the object's `params` dictionary.
    
    Attributes
    ----------
    path : str
        Path to save.
    
    name : str
        Name of the object, e.g. used in file-naming.
    
    params : dict
        DEPRECIATED. Dictionary containing other parameters initialised by the object.
    
    '''
    # _seed = None
    
    def __init__(self, path=None, name=None, make_dirs=True):
        # , **param_dict):
        self.path = self._validate_path(path, make_dirs=make_dirs)
        self.name = self._validate_name(name)
        # self.params = self._validate_params(param_dict)

    def _validate_path(self, path, make_dirs=True):
        '''Validate path is a directory, otherwise make directories.
        Raises FileNotFoundError if not a directory, or TypeError if not string. '''
        if path is None or path == '':
            path = ''
        elif isinstance(path, str):
            if os.path.exists(path) and not os.path.isdir(path):
                msg = f'Path: "{path}" is not a directory.'
                raise FileNotFoundError(msg)
            elif not os.path.exists(path):
                if make_dirs:
                    os.makedirs(path)
                else:
                    msg = f'Directory: {path} does not exist.'
                    raise FileNotFoundError(msg)
        else:
            msg = f'Path: {path} is not an instance of string.'
            raise TypeError(msg)

        return path

    def _validate_name(self, name):
        '''Validate name. Raises TypeError if not a string.'''
        if name is None:
            name = self.__class__.__name__.lower()
        elif not isinstance(name, str):
            msg = f'Name: {name} is not an instance of string.'
            raise TypeError(msg)

        return name
    
    # def _validate_params(self, param_dict):
    #     '''Validate params. Raises TypeError if not a dictionary.'''
    #     if param_dict is None:
    #         param_dict = {}
    #     elif not isinstance(param_dict, dict):
    #         msg = f'Params: {param_dict} is not an instance of dict.'
    #         raise TypeError(msg)

    #     return param_dict

    def _make_filename(self, suffix, ext):
        '''Returns a file name from a given suffix and extension.'''
        return f'{self.name}_{suffix}.{ext}'
    
    def _make_filepath(self, filename):
        '''Returns a file path to a given filename.'''
        return os.path.join(self.path, filename)    
    
    # # @_return_filename
    # def save_params(self, suffix='params', return_filename=False):
    #     '''
    #     Saves `params` in a JSON format with filename <name>_<suffix>.json where
    #     name is the objects name.
        
    #     Parameters
    #     ----------
    #     suffix : str, optional
    #         The suffix used in the file name. Default (recommended) is 'params'.
        
    #     return_filename : bool, optional
    #         If True, returns the file name generated. Default is False.
        
    #     Returns
    #     -------
    #     filename : None or str  
        
    #     '''
    #     filename = self._make_filename(suffix, 'json')
    #     filepath = self._make_filepath(filename)
    #     with open(filepath, 'w') as file_out:
    #         json.dump(self.params, file_out)
        
    #     # return filename

    @classmethod
    def _name_from_filepath(cls, filepath):
        '''Automatically extracts the object name from a filepath, 
        assuming default naming convention.'''
        return '_'.join(os.path.split(filepath)[1].split('_')[:-1])

    # @classmethod
    # def from_params(cls, filepath, name=None):
    #     '''
    #     Loads object from params file, if name is None, will assume
    #     defaults and strip name from filename.
        
    #     Parameters
    #     ----------
    #     filepath : str
    #         Path to the params file.
        
    #     name : str, optional
    #         Name to give the returned object. If None (default) the name is
    #         automatically detected in the filename, assuming defaults for this
    #         module.
        
    #     Returns
    #     -------
    #     interstellar_object : object
    #         Instance of the class from which the function is called.
            
    #     Raises
    #     ------
    #     ValueError :
    #         If JSONDecoder fails to load file.
        
    #     TypeError :
    #         If params data is not a dictionary.
        
    #     '''        
    #     with open(filepath) as file_in:
    #         try:
    #             kwargs = json.load(file_in)
    #         except json.JSONDecodeError:
    #             msg = f'Failed to decode file at {filepath}. File be a valid JSON data format.'
    #             raise ValueError(msg)
        
    #     path = os.path.split(filepath)[0]
    #     if not name:
    #         name = cls._name_from_filepath(filepath)

    #     if isinstance(kwargs, dict):
    #         interstellar_object = cls(path=path, name=name, **kwargs)
    #     else:
    #         msg = 'Invalid params data loaded. Must be of type dict.'
    #         raise TypeError(msg)
        
    #     return interstellar_object

            
class Network(Interstellar):
    '''
    Base neural network class. Subclass this to configure custom neural network
    architectures and shapes with the `build`, `compile` and `train` methods.
    
    TODO: Move model and history filename, and metrics into the params dict
    if appropriate so they can be more easily loaded in. Not necessary now.
    
    Parameters
    ----------
    path :
    
    name :
    
    **param_dict : DEPRECIATED
    
    Attributes
    ----------
    path : str
    
    name : str
    
    params : dict DEPRECIATED
    
    model : tf.keras.Model
    
    model_filename : str
    
    history : pd.DataFrame
    
    history_filename : str
    
    metrics : list of str
        List of metrics (including the loss in position 0) tracked by the `model`.
        Specify the loss and additional metrics in the `compile` method.
    
    x_cols : list of str
        List of input data column names.
    
    y_cols : list of str
        List of output data column names.
    
    '''
    def __init__(self, path=None, name=None, model_filename=None, history_filename=None):
    # , **param_dict):
        super().__init__(path=path, name=name)
        # , **param_dict)
        self.model = None
        self.model_filename = model_filename
        self.history = pd.DataFrame()
        self.history_filename = history_filename
        self.metrics = []
        self.x_cols = []
        self.y_cols = []
    
    # @_return_filename
    def save_history(self, suffix='history', nth_row=None, return_filename=False):
        '''Saves every nth row of history.'''
        filename = self._make_filename(suffix, 'csv')
        self.history.iloc[::nth_row, :].to_csv(self._make_filepath(filename), index=False)
        self.history_filename = filename
        # return filename
    
    # @_return_filename
    def save_model(self, suffix='model', returns=None, return_filename=False):
        filename = self._make_filename(suffix, 'h5')
        self.model.save(self._make_filepath(filename))
        self.model_filename = filename
        # return filename
    
    def load_model(self, filepath=None, suffix='model'):
        if filepath is None:
            filepath = self._make_filepath(self._make_filename(suffix, 'h5'))
        self.model = tf.keras.models.load_model(filepath)
        self.model_filename = os.path.split(filepath)[1]
        self.metrics = [self.model.loss] + [m.name for m in self.model.metrics]
        
    def load_history(self, filepath=None, suffix='history'):
        if filepath is None:
            filepath = self._make_filepath(self._make_filename(suffix, 'csv'))
        self.history = pd.read_csv(filepath)
        self.history_filename = os.path.split(filepath)[1]

    @classmethod
    def from_grid(cls, filepath, index, param_columns=None):
        '''Loads network from grid data file.
        
        The path is inferred from filepath.
        
        param_columns contain any other parameters used in the initialization 
        of the Network instance (see examples). If None or 'all', all columns
        will be used.
        
        If columns exist for model and history filenames, an attempt to load these
        is made. The user is warned if the files are not found.
        
        Returns an instance of Network
        '''
        df = pd.read_csv(filepath)
        # name = df.loc[index, 'name']
        
        if param_columns is None or param_columns == 'all':
            param_columns = df.columns

        # name = df.loc[index, 'name']        
        kwargs = df.loc[index, param_columns]
        
        path = os.path.split(filepath)[0]
        network = cls(path=path,
                    #   name=name,
                      **kwargs)
        
        # In future, move this to a separate function?
        def fnf_warn(file, filepath):
            msg = f'{file} not found at {filepath}.'
            warnings.warn(msg, UserWarning)
        
        load_files = {'model_filename': network.load_model,
                  'history_filename': network.load_history}
        
        for file, load in load_files.items():
            if file in param_columns:
                try:
                    filepath = network._make_filepath(df.loc[index, file])
                    load(filepath)
                except FileNotFoundError:
                    fnf_warn(file, filepath)
        
        return network

    def set_metrics(self, metrics):
        self.metrics = metrics

    def build(self, inputs, outputs):
        ''' Build the neural Network '''
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

    def compile(self, optimizer='SGD', loss='mae', metrics=None, **kwargs):
        ''' Compile the model '''
        # TODO: comptible with dict metrics for difference outputs.
        self.metrics = [loss] if metrics is metrics is None else [loss] + metrics
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        self.history = pd.DataFrame()

    def _make_history(self, logs):
        epochs = pd.DataFrame({'epochs': logs.epoch})
        loss = pd.DataFrame(logs.history)
        return pd.concat([epochs, loss], axis=1)
    
    def _get_xy(self, data, dtype=None):
        x = data[self.x_cols].to_numpy(dtype=dtype)
        y = data[self.y_cols].to_numpy(dtype=dtype)
        return x, y
    
    def _tensorboard(self):
        logdir = os.path.join(self.path, f'logs/scalars/{self.name}')
        return tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)
        
    def _checkpoint(self, monitor='val_loss'):
        filepath = self._make_filepath(self._make_filename('best_model', 'h5'))
        return tf.keras.callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                                  mode='min', save_best_only=True)
    
    def _earlystopping(self, monitor='val_loss', min_delta=1e-7, baseline=1e-5, patience=100):
        return tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='min',
                                                baseline=baseline, patience=patience,
                                                min_delta=min_delta)
                   
    def train(self, data, x_cols, y_cols, epochs=100,
                batch_size=None,
                validation_split=0.2,
                validation_data=None,
                dtype=np.float32,
                callbacks=None,
                tensorboard_kw={},
                earlystopping_kw = {},
                checkpoint_kw = {},
                fit_kw={}):
        '''
        Trains the model associated with Network.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
            
        x_cols: list of str
            Columns associated with inputs
            
        y_cols: np.ndarray or pd.DataFrame
            Columns associated with outputs

        epochs: int
            The maximum number of training epochs to run for.

        batch_size : int, optional
            Training batch size: the number of rows to forward pass before updating the
            weights. If None (default) the batch_size is 32.
        
        validation_split: floatXX between 0 and 1
            The fraction of data (x & y) to be used for the calculation
            of validation metrics during training.

        validation_data: pd.DataFrame, optional
            Overrides `validation_split`. This is a dataframe used for validation
            at the end of each epoch. Typically should be about 20% of the combined
            size of `data` and `validation_data`. If None (default) `validation_split`
            is used.
            
        baseline: floatXX
            A metric for early stopping.  When the validation loss falls
            below this value the early stop mechanism will be triggered.  The
            training will run for a little longer (defined by patience).  The
            ModelCheckpoint callback will store the best model from the entire
            training run.

        fractional_patience: floatXX between 0 and 1
            When the early stopping mechanism is triggerred the the training
            will run for n more epochs where n = fractional_patience * epochs.
        
        callbacks : list of str, optional
            List of callbacks to add. One or many of 'tensorboard', 'model checkpoint',
            'early stopping'.
        
        dtype : 
            Choice of float precision to cast training data to.

        '''
        self.x_cols = x_cols
        self.y_cols = y_cols
        
        x, y = self._get_xy(data, dtype=dtype)
        
        if validation_data is not None:
            validation_data = self._get_xy(validation_data, dtype=dtype)
        
        # Setup callbacks TODO!
        # logdir = os.path.join(self.path, f'logs/{self.name}')

        # #TODO make optional
        # tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

        # # Change EarlyStopping to care about the change in loss
        # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
        #                                       baseline=baseline, patience=patience,
        #                                       min_delta=min_delta, )
        
        # mc = tf.keras.callbacks.ModelCheckpoint(f'{self.name}_best_model.h5',
        #                                         monitor='val_loss',
        #                                         mode='min', save_best_only=True, )
        
        cbks = []
        
        if callbacks is not None:
            cbk_dict = {'tensorboard': [self._tensorboard, tensorboard_kw],
                        'earlystopping': [self._earlystopping, earlystopping_kw],
                        'checkpoint': [self._checkpoint, checkpoint_kw]}
            for c in callbacks:
                cbks.append(cbk_dict[c][0](**cbk_dict[c][1]))
        
        initial_epoch = self.history['epochs'].iloc[-1] + 1 if len(self.history) != 0 else 0
        
        logs = self.model.fit(x, y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            validation_data=validation_data,
                            callbacks=cbks,
                            verbose=0,
                            initial_epoch=initial_epoch,
                            **fit_kw)
        
        # Update history
        if len(self.history) == 0:
            self.history = self._make_history(logs)
        elif logs.history and logs.epoch:
            self.history = self.history.append(self._make_history(logs), ignore_index=True)

    def predict(self, x):
        # Need to return dataframe to remain consistent
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x[self.x_cols]
            x = x.to_numpy()
        y = self.model.predict(x)
        return pd.DataFrame(y, columns=self.y_cols)
    
    def evaluate(self, data):
        '''Wrapper for tf.keras.Model.evaluate'''
        x, y = self._get_xy(data)
        scores = self.model.evaluate(x, y, verbose=0)
        if self.metrics is not None:
            series = pd.Series(scores, index=self.metrics, name=self.name)
        else:
            series = pd.Series(scores, name=self.name)
        return series

    def _absolute_error(self, y, y_pred):
        return np.abs(y - y_pred)
    
    def _squared_error(self, y, y_pred):
        return (y - y_pred)**2

    def scores(self, data, metric='absolute error', percentiles=None):
        '''Produce a dataframe of scores for a given metric.'''
        x, y = self._get_xy(data)
        y_pred = self.predict(x)
        
        if metric == 'absolute error':
            y_metric = self._absolute_error(y, y_pred)
        elif metric == 'squared error':
            y_metric = self._squared_error(y, y_pred)
        else:
            msg = 'Invalid metric chosen. Choose one of "absolute error" or ' +\
                '"squared error".'
            raise ValueError(msg)
        
        # summary = {'mean': np.mean(y_metric, axis=0),
        #            'std': np.std(y_metric, axis=0),
        #            '16%': np.quantile(y_metric, 0.16, axis=0),
        #            '50%': np.quantile(y_metric, 0.5, axis=0),
        #            '84%': np.quantile(y_metric, 0.84, axis=0),
        #            }
        
        # add y_cols from data to the rows of this
        # df = pd.DataFrame(summary, index=self.y_cols)
        
        return y_metric.describe(percentiles=percentiles)
    
    
class DenseNetwork(Network):
    '''
    Simple neural network class. This assumes a fully-connected regression
    neural network with a constant depth, width and activation. A batch size
    may be specified for using in the `train` method. L2 regularization may also
    be specified to be applied to each layer. The final layer has a linear
    output function and no regularization.
    
    Due to its simplicity, a `to_theano` method is provided which returns a
    theano.tensor-compatible model which may be used by the likes of PyMC3.
    Note there are very small differences in output from the two methods of
    the order ~ 1e-7. Use the `predict` method to test this for yourself if
    necessary.
    
    TODO: Move model and history filename, and metrics into the params dict
    if appropriate so they can be more easily loaded in. Not necessary now.
    
    Parameters
    ----------
    path : str
    
    name : str
    
    depth : int
    
    width : int
    
    activation : str
    
    batch_size : int
    
    regularization_l2 : float
    
    **kwargs :
        Additional keyword arguments to pass to Network.
    
    Attributes
    ----------
    path : str
    
    name : str
    
    params : dict
        
    model : tf.keras.Model
    
    model_filename : str
    
    history : pd.DataFrame
    
    history_filename : str
    
    metrics : list of str
        List of metrics (including the loss in position 0) tracked by the `model`.
        Specify the loss and additional metrics in the `compile` method.
    
    '''
    def __init__(self, path=None, name=None, depth=None, width=None, activation=None, batch_size=None,
                 regularization_l2=None, **kwargs):
        
        super().__init__(path=path, name=name, # param_dict=param_dict)
                         **kwargs)
                            # depth=depth,
                            # width=width,
                            # activation=activation,
                            # batch_size=batch_size,
                            # regularization_l2=regularization_l2)
        
        # DEFAULTS
        self.depth = 1 if depth is None else depth
        self.width = 1 if width is None else width
        self.activation = activation
        self.batch_size = batch_size
        self.regularization_l2 = 0.0 if regularization_l2 is None else regularization_l2
        
    @classmethod
    def from_grid(cls, filepath, index):
        super().from_grid(filepath, index, param_columns=SIMPLE_COLS)

    def build(self, input_dimension, output_dimension, *args, **kwargs):
        ''' Build the neural Network '''
        inputs = tf.keras.Input(shape=(input_dimension,), name='inputs')
        xx = layers.Dense(self.width, activation=self.activation,
                          kernel_regularizer=l2(self.regularization_l2))(inputs)
        for i in range(self.depth - 1):
            xx = layers.Dense(self.width, activation=self.activation,
                              kernel_regularizer=l2(self.regularization_l2))(xx)
        outputs = layers.Dense(output_dimension, activation='linear')(xx)
        super().build(inputs, outputs, *args, **kwargs)
    
    def train(self, *args, **kwargs):
        '''Trains using batch size parameter'''
        super().train(*args, batch_size=self.batch_size, **kwargs)

    def to_theano(self):
        '''Converts model to theano compatible neural network function.
        This is written for DenseNetworkGrid only, cannot be generalised.'''
        weights = self.model.get_weights()
        
        def theano_nnet(x):
            n_hidden_layers = int(len(weights) / 2 - 2)
            xx = T_ACT[self.activation](
                T.dot(x, weights[0]) + weights[1])
            for i in range(1, n_hidden_layers + 1):
                xx = T_ACT[self.activation](
                    T.dot(xx, weights[i * 2]) + weights[i * 2 + 1])
            xx = T.dot(xx, weights[-2]) + weights[-1]
            
            return xx
        
        return theano_nnet


class Grid(Interstellar):
    '''An object for performing a grid-based search of nerual network hyperparameters.
    
    Parameters
    ----------
    path : str, optional
    
    name : str, optional
    
    **params_kwargs : 
        Keyword arguments corresponding to the names and values along each
        axis of the grid. Each axis may be array-like or single-valued.
        
    Attributes
    ----------
    path : str
    
    name : str
    
    params : dict
        A dictionary containing the names and values of params on the grid.
        
    data : pandas.DataFrame
        A pandas.DataFrame with a row corresponding to each coordinate on the
        grid.

    '''
    def __init__(self, path=None, name=None, **params_kwargs):
        # , **param_dict):
        super().__init__(path=path, name=name)
        # , **param_dict)
        
        self.data = None
        self.params = params_kwargs
        if self.params:
            self.make_grid()
        else:
            self.data = pd.DataFrame(columns=['name'])
        
        self.data['model_filename'], self.data['history_filename'] = [None, None]

        self.networks = pd.Series()

    def __repr__(self):
        return f'{type(self).__name__}(path={self.path}, name={self.name}, **{self.params})'
        
    def make_grid(self):
        '''Returns a DataFrame containing all points on the grid. '''
        meshgrid = np.meshgrid(*self.params.values())
        keys = ['name'] + [*self.params.keys()]
        values = [axis.ravel() for axis in meshgrid]
        values = [[f'{self.name}_{i}' for i in range(len(values[0]))]] + values
        data = {key: value for key, value in zip(keys, values)}
        self.data = pd.DataFrame(data)
        
    def save_data(self, suffix='data'):
        self.data.to_csv(os.path.join(self.path, f'{self.name}_{suffix}.csv'), index=False)
    
    # def _save_model(self, network):
    #     self.data.at[self.networks==network, 'model_filename'] = network.save_model(return_filename=True)
    
    # def _save_history(self, network):
    #     self.data.at[self.networks==network, 'history_filename'] = network.save_history(return_filename=True)
    
    # def save_networks(self, model=True, history=True):
    #     '''Save networks data, choice of model and/or history files.'''
    #     if model:
    #         self.networks.apply(self._save_model)
    #     if history:
    #         self.networks.apply(self._save_history)

    @classmethod
    def from_data(cls, filepath, name=None):
        '''Loads grid data file for inspection, returns grid'''
        if name is None:
            # Attempts autodetection of name
            # filename = os.path.split(filepath)[1]
            name = cls._name_from_filepath(filepath)
        
        data = pd.read_csv(filepath)
        
        path = os.path.split(filepath)[0]
        grid = cls(path=path, name=name)
        grid.data = data
        
        return grid      

    def make_networks(self, network_class=Network, param_columns=None):
        if param_columns is None or param_columns == 'all':
            self.networks = self.data.apply(lambda x: network_class(path=self.path, **x), axis=1)
        elif isinstance(param_columns, (tuple, list)):
            if 'name' not in param_columns:
                param_columns = ['name'] + param_columns
            self.networks = self.data[param_columns].apply(lambda x: network_class(path=self.path, **x), axis=1)
        else:
            msg = f'Argument `param_columns` must be a tuple or list.'
            raise ValueError(msg)

    def save_networks(self, model=True, history=True):
        if model:
            self.networks.apply(lambda network: network.save_model())
            self.data['model_filename'] = self.networks.apply(lambda n: n.model_filename)
        if history:
            self.networks.apply(lambda network: network.save_history())
            self.data['history_filename'] = self.networks.apply(lambda n: n.history_filename)

    def load_networks(self, model=True, history=True, model_kw={}, history_kw={}):
        # self.make_networks(**kwargs)
        
        # Change this to allow for filenames in grid
        if model:
            self.networks.apply(lambda network: network.load_model(**model_kw))
        if history:
            self.networks.apply(lambda network: network.load_history(**history_kw))

    def build_all(self, *args, **kwargs):
        self.networks.apply(lambda network: network.build(*args, **kwargs))

    def compile_all(self, optimizer='SGD', loss='mae', metrics=None, **kwargs):
        self.networks.apply(lambda network: network.compile(
            optimizer=optimizer, loss=loss, metrics=metrics, **kwargs))
                    
    def train_all(self, data, x_cols, y_cols, epochs=100,
                  save_model=True, save_history=True, save_data=True, **kwargs):
        
        if save_data:
            self.save_data()
        # Here a loop is appropriate in order to track exceptions clearly
        for index, network in self.networks.iteritems():
            try:
                network.train(data, x_cols, y_cols, epochs=epochs, **kwargs)
            except Exception as err:
                msg = f'Training failed for {index}th network due to exception:\n{err}.' +\
                    '\nThis will be noted in the grid data attribute.'
                warnings.warn(msg, UserWarning)
                self.data.at[index, 'exceptions'] = str(err)
                
            if save_model:
                network.save_model()
                self.data.at[index, 'model_filename'] = network.model_filename
            if save_history:
                network.save_history()
                self.data.at[index, 'history_filename'] = network.history_filename
            if save_data:
                self.save_data()  # On each loop for live progress.
            
    def evaluate_all(self, data, save_data=True, inplace=True, **kwargs):
        '''Inplace adds stats to data'''
        scores = self.networks.apply(lambda network: network.evaluate(data, **kwargs))
        
        if inplace:
            matching_cols = np.array([scol in self.data.columns for scol in scores.columns])
            if any(matching_cols):
                self.data[scores.columns[matching_cols]] = scores.loc[:, matching_cols]
                self.data = self.data.join(scores.loc[:, ~matching_cols])
            else:
                self.data = self.data.join(scores)
            scores = None

        if save_data:
            self.save_data()
        
        return scores

    def reset_path(self, path):
        path = self._validate_path(path)
        self.path = path
        if len(self.networks) != 0:
            def set_path(network):
                network.path = path
            self.networks.apply(set_path)


class DenseNetworkGrid(Grid):
    '''Generate a dense network grid with column names corresponding to that of 
    a fully connected neural network model.
        
    Parameters
    ----------
    path : str, optional
    
    name : str, optional
    
    depth : int or list of int, optional
        The number of hidden layers in the model.
    
    width : int or list of int, optional
        The number of neurons per hidden layer in the model.
    
    activation : str or list of str, optional
    
    batch_size : int or list of int, optional
        The size of the batches.
    
    regularization_l2 : float or list of float, optional    
    
    Attributes
    ----------
    params : dict
        A dictionary containing the names and values of params on the grid.
        
    data : pandas.DataFrame
        A pandas.DataFrame with a row corresponding to each coordinate on the
        grid.
    
    '''
    def __init__(self, path=None, name=None, depth=None, width=None, activation=None, batch_size=None,
                 regularization_l2=None):
        
        super().__init__(path=path, name=name, depth=depth, width=width, activation=activation,
                         batch_size=batch_size, regularization_l2=regularization_l2)

    def make_networks(self):
        super().make_networks(DenseNetwork, SIMPLE_COLS)
