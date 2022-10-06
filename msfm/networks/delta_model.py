import tensorflow as tf

import horovod.tensorflow as hvd

from deepsphere import HealpyGCNN

# local imports
from .base_model import BaseModel
from . import losses


class DeltaLossModel(BaseModel):
    """
    This class subclasses the BaseModel to have a HealpyGCNN with delta loss
    """

    def __init__(self, nside, indices, layers, n_neighbors=20, input_shape=None, optimizer=None, save_dir=None,
                 restore_point=None, summary_dir=None, init_step=0, is_chief=True):
        """
        Initializes a graph convolutional neural network using the healpy pixelization scheme
        :param nside: integeger, the nside of the input
        :param indices: 1d array of inidices, corresponding to the pixel ids of the input of the NN
        :param layers: a list of layers that will make up the neural network
        :param n_neighbors: Number of neighbors considered when building the graph, currently supported values are:
                            8, 20 (default), 40 and 60.
        :param input_shape: Optional input shape of the network, necessary if one wants to restore the model
        :param optimizer: Optimizer of the model, defaults to Adam
        :param save_dir: Directory where to save the weights and so, can be None
        :param restore_point: Possible restore point, either directory (of which the latest checkpoint will be chosen)
                              or a checkpoint file
        :param summary_dir: Directory to save the summaries
        :param init_step: Initial step, defaults to 0
        :param is_chief: Chief in case of distributed setting
        """

        # get the network
        if nside is None and indices is None:
            print("Initializing DeltaLossModel with a normal Sequential model...")
            network = tf.keras.Sequential(layers=layers)
        else:
            network = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=n_neighbors)

        # init the base model
        super(DeltaLossModel, self).__init__(network=network, input_shape=input_shape, optimizer=optimizer,
                                          save_dir=save_dir, restore_point=restore_point, summary_dir=summary_dir,
                                          init_step=init_step, is_chief=is_chief)

    def setup_delta_loss_step(self, n_params, n_same, off_sets, n_points=1, n_input=None, n_channels=1, n_output=None,
                              jac_weight=0.0, force_params=None, force_weight=1.0, jac_cond_weight=None,
                              use_log_det=True, no_correlations=False, tikhonov_regu=True, weights=None, eps=1e-32,
                              n_partial=None, clip_by_value=None, clip_by_norm=None, clip_by_global_norm=None,
                              img_summary=False, train_indices=None, cov_weight=False, l2_norm_weight=None):
        """
        This sets up a function that performs one training step with the delta loss, which tries to maximize the
        information of the summary statistics. Note  it needs the maps need to be ordered in a specific way:
            * The shape of the maps is (n_points*n_same*(2*n_params+1), len(indices), n_channels)
            * If one splits the maps into (2*n_params+1) parts among the first axis one has the following scheme:
                * The first part was generated with the unperturbed parameters
                * The second part was generated with parameters where off_sets[0] was subtracted from the first param
                * The third part was generated with parameters where off_sets[0] was added from to first param
                * The fourth part was generated with parameters where off_sets[1] was subtracted from the second param
                * and so on
        The training step function that is set up will only work if the input has a shape:
        (n_points*n_same*(2*n_params+1), len(indices), n_channels)
        If multiple clippings are requested the order will be:
            * by value
            * by norm
            * by global norm
        :param n_params: Number of underlying model parameter
        :param n_same: How many summaries (unperturbed) are coming from the same parameter set
        :param off_sets: The off_sets used to perturb the original parameters and used for the Jacobian calculation
        :param n_points: number of different parameter sets
        :param n_input: Input dimension of the network, must be provided if the network is not a HealpyGCNN
        :param n_channels: number of channels from the input
        :param n_output: Dimensionality of the summary statistic, defaults to predictions.get_shape()[-1]
        :param jac_weight: The weight of the Jacobian loss (loss that forces the Jacobian of the summaries to be close
                           to unity (or identity matrix).
        :param force_params: Either None or a set of parameters with shape (n_points, 1, n_output) which is used to
                             compute a square loss of the unperturbed summaries. It is useful to set this for example to
                             zeros such that the network does not produces arbitrary high summary values
        :param force_weight: The weight of the square loss of force_params
        :param jac_cond_weight: If not None, this weight is used to add an additional loss using the matrix condition
                            number of the jacobian
        :param use_log_det: Use the log of the determinants in the information inequality, should be True. If False the
                            information inequality is not minimized in a proper manner and the training can become
                            unstable.
        :param no_correlations: Do not consider correlations between the parameter, this means that one tries to find
                                an optimal summary (single value) for each underlying model parameter, only possible
                                if n_output == n_params
        :param tikhonov_regu: Use Tikhonov regularization of matrices e.g. to avoid vanishing determinants. This is the
                              recommended regularization method as it allows the usage of some optimized routines.
        :param weights: An 1d array of length n_points, used as weights in means of the different points.
        :param eps: A small positive value used for regularization of things like logs etc. This should only be
                    increased if tikhonov_regu is used and a error is raised.
        :param n_partial: To train only on a subset of parameters and not all underlying model parameter. Defaults to
                          None which means the information inequality is minimized in a normal fashion. Note that due to
                          the necessity of some algebraic manipulations n_partial == None and n_partial == n_params lead
                          to slightly different behaviour.
        :param clip_by_value: Clip the gradients by given 1d array of values into the interval [value[0], value[1]],
                              defaults to no clipping
        :param clip_by_norm: Clip the gradients by norm, defaults to no clipping
        :param clip_by_global_norm: Clip the gradients by global norm, defaults to no clipping
        :param img_summary: image summary of jacobian and covariance
        :param train_indices: A list of indices, if not None only [trainable_variables[i] for i in train_indices] will
                              be trained
        :param cov_weight: If true, the jac weight will be used as cov_weight, i.e. loss cov mat will be forced to be
                           close to the identity matrix
        :param l2_norm_weight: weight for the L2 norm of the trainable weights
        """
        # check if we run in distributed fashion
        try:
            num_workers = hvd.size()
        except ValueError:
            num_workers = None

        # setup a loss function
        def loss_func(predictions):
            return losses.delta_loss(predictions=predictions, n_params=n_params, n_same=n_same, off_sets=off_sets,
                                     n_output=n_output, jac_weight=jac_weight, force_params=force_params,
                                     force_weight=force_weight, jac_cond_weight=jac_cond_weight,
                                     use_log_det=use_log_det, no_correlations=no_correlations,
                                     tikhonov_regu=tikhonov_regu, summary_writer=self.summary_writer, training=True,
                                     weights=weights, eps=eps, n_partial=n_partial, num_workers=num_workers,
                                     img_summary=img_summary, cov_weight=cov_weight)

        # get the backend float and input shape
        current_float = losses._get_backend_floatx()
        if isinstance(self.network, HealpyGCNN):
            in_shape = (n_points * n_same * (2 * n_params + 1), len(self.network.indices_in), n_channels)
        else:
            if n_channels is None:
                in_shape = (n_points * n_same * (2 * n_params + 1), n_input)
            else:
                in_shape = (n_points * n_same * (2 * n_params + 1), n_input, n_channels)

        # tf function with nice signature
        @tf.function(input_signature=[tf.TensorSpec(shape=in_shape, dtype=current_float)])
        def delta_train_step(input_batch):
            self.base_train_step(input_tensor=input_batch, loss_function=loss_func, input_labels=None,
                                 clip_by_value=clip_by_value, clip_by_norm=clip_by_norm,
                                 clip_by_global_norm=clip_by_global_norm, training=True, num_workers=num_workers,
                                 train_indices=train_indices, l2_norm_weight=l2_norm_weight)

        self.delta_train_step = delta_train_step

        if num_workers is not None:
            print("It it important to call the function <broadcast_variables> after the first gradient descent step, "
                  "to ensure that everything is correctly initialized (also the optimizer)")
