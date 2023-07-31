import numbers
from typing import Union

import numpy as np

from river.base import DriftAndWarningDetector, DriftDetector

from abcd_supplementary.feature_extraction import *
from abcd_supplementary.windowing import *
from abcd_supplementary.util import handles_dicts, convert_to_univariate_if_possible


class ABCD(DriftAndWarningDetector):
    def __init__(self,
                 delta: float = 0.01,
                 delta_warning: float = 0.1,
                 model_type: EncoderDecoderType = EncoderDecoderType.AE,
                 encoding_factor: float = 0.5,
                 update_epochs: int = 50,
                 min_instances_pretrain: int = 100,
                 split_type: SplitType = SplitType.Equidistant,
                 num_splits: int = 20,
                 max_size: int = np.infty,
                 tau: float = 2.5,
                 custom_encoder_decoder: Union[EncoderDecoder, None] = None,
                 bonferroni: bool = False,
                 seed: int = 0):
        """
        Adaptive Bernstein change detector for high-dimensional data streams (ABCD).
        ABCD combines an encoder-decoder model, Bernstein’s inequality, and adaptive windowing to detect changes.
        Parameters
        ----------
        delta : float
            The p-value threshold for detecting changes
        delta_warning : float
            The p-value threshold for detecting a warning
        model_type : EncoderDecoderType
            The type of model used (supports PCA, Kernel PCA, Autoencoder, and custom models)
        encoding_factor : float
            The relative size of the bottleneck. Gets passed to the encoder decoder model.
        update_epochs : int
            The number of epochs to train the autoencoder
        min_instances_pretrain : int
            The minimum number of instances required before retraining happens
        split_type : SplitType
            The type of split to use (supports 'all' and 'equidistant')
        num_splits : int
            The number of window splits that get evaluated when a new instance arrives
        max_size : int
            The maximum size of the adaptive windows
        tau : float
            The threshold for change subspace detection
        custom_encoder_decoder : EncoderDecoder
            The custom encoder-decoder class (uninitialized)
        bonferroni : bool
            Whether to use Bonferroni correction when computing the p-value
        seed : int
            The random seed used during random number generation (e.g., when initializing the weights of the autoencoder)
        """
        self._split_type = split_type
        self._delta = delta
        self._delta_warning = delta_warning
        self._bonferroni = bonferroni
        self._num_splits = num_splits
        self._max_size = max_size
        self.tau = tau
        self._window = AdaptiveWindow(delta=delta,
                                      delta_warning=delta_warning,
                                      split_type=split_type,
                                      max_size=max_size,
                                      bonferroni=bonferroni,
                                      n_splits=num_splits)
        self._drift_detected = False
        self._warning_detected = False
        self._seen_elements = 0
        self._last_loss = np.nan
        self._epochs = update_epochs
        self._eta = encoding_factor
        self.severity = np.nan
        self.drift_dimensions = None
        self.model_type = model_type
        self._dict_keys: Union[np.ndarray, None] = None
        self.model: Union[EncoderDecoder, None] = None
        self._model_class: Union[callable, None] = None
        self._min_instances_pretrain = min_instances_pretrain
        self._pretrain_buffer = []
        self._pretrained = False
        if model_type == EncoderDecoderType.PCA:
            self._model_class = PCAModel
        elif model_type == EncoderDecoderType.KernelPCA:
            self._model_class = KernelPCAModel
        elif model_type == EncoderDecoderType.AE:
            torch.manual_seed(seed)
            self._model_class = AutoEncoder
        elif model_type == EncoderDecoderType.Custom:
            assert custom_encoder_decoder is not None;
            "If using EncoderDecoderType.Custom you must provide a valid EncoderDecoder class"
            self._model_class = custom_encoder_decoder
        else:
            raise ValueError
        super(ABCD, self).__init__()

    def _pre_train(self, data: np.ndarray):
        self.model.update(data, epochs=self._epochs)
        self._pretrain_buffer = []
        self._pretrained = True

    def _new_model(self, n_dims: int):
        self.model = self._model_class(input_size=n_dims, eta=self._eta)
        self._pretrained = False

    @handles_dicts
    @convert_to_univariate_if_possible
    def update(self, x: Union[numbers.Number, np.ndarray, dict]) -> DriftDetector:
        self._seen_elements += 1
        self._reset_flags()
        if isinstance(x, numbers.Number):  # handle univariate data
            new_tuple = (x, np.array([0]), np.array([x]))  # we want to find changes in x
        else:  # handle multivariate data
            if self.model is None:
                self._new_model(x.shape[-1])
            if not self._pretrained:
                self._pretrain_buffer.append(x)
                if len(self._pretrain_buffer) >= self._min_instances_pretrain:
                    self._pre_train(np.array(self._pretrain_buffer, ndmin=2))
                return self
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)
            new_tuple = self.model.new_tuple(x)

        # append reconstruction / univariate data point to window
        self._window.grow(new_tuple)
        self._last_loss = self._window.most_recent_loss()
        self._drift_detected, self._warning_detected, detection_point = self._window.has_change()

        if self.drift_detected or self.warning_detected:
            self._evaluate_subspace()
            self._evaluate_magnitude()

        if self.drift_detected:
            self.model = None
            self._pretrain_buffer = self._window.data_new().tolist()
            self._window.reset()
        return self

    def metric(self):
        return self._last_loss

    def get_dims_p_values(self) -> np.ndarray:
        return self.drift_dimensions

    def get_drift_dims(self) -> np.ndarray:
        drift_dims = np.array([
            i for i in range(len(self.drift_dimensions)) if self.drift_dimensions[i] < self.tau
        ])
        return np.arange(len(self.drift_dimensions)) if len(drift_dims) == 0 else drift_dims

    def get_severity(self):
        return self._severity

    def _evaluate_subspace(self):
        data = self._window.data()
        output = self._window.reconstructions()
        error = output - data
        squared_errors = np.power(error, 2)
        window1 = squared_errors[:self._window.t_star]
        window2 = squared_errors[self._window.t_star:]
        mean1 = np.mean(window1, axis=0)
        mean2 = np.mean(window2, axis=0)
        eps = np.abs(mean2 - mean1)
        sigma1 = np.std(window1, axis=0)
        sigma2 = np.std(window2, axis=0)
        n1 = len(window1)
        n2 = len(window2)
        p = p_bernstein(eps, n1=n1, n2=n2, sigma1=sigma1, sigma2=sigma2)
        self.drift_dimensions = p

    def _evaluate_magnitude(self):
        drift_point = self._window.t_star
        data = self._window.data()
        recons = self._window.reconstructions()
        drift_dims = self.get_drift_dims()
        if len(drift_dims) == 0:
            drift_dims = np.arange(data.shape[-1])
        input_pre = data[:drift_point, drift_dims]
        input_post = data[drift_point:, drift_dims]
        output_pre = recons[:drift_point, drift_dims]
        output_post = recons[drift_point:, drift_dims]
        se_pre = (input_pre - output_pre) ** 2
        se_post = (input_post - output_post) ** 2
        mse_pre = np.mean(se_pre, axis=-1)
        mse_post = np.mean(se_post, axis=-1)
        mean_pre, std_pre = np.mean(mse_pre), np.std(mse_pre)
        mean_post = np.mean(mse_post)
        if std_pre == 0:
            std_pre = 1e-10
        z_score_normalized = np.abs(mean_post - mean_pre) / std_pre
        self.severity = float(z_score_normalized)

    def _reset_flags(self):
        self._drift_detected = False
        self._warning_detected = False
