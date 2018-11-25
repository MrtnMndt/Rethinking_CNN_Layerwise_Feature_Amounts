import numpy as np
import scipy.special as spcl
import math


class SkewNormalModels:
    # TODO: expose distribution parameters to command line parameters
    def __init__(self, depth=16, num_classes=10, patch_size=32, num_fc_layers=2, min_layer_filters=2,
                 conv_filter_size=3, dist_step=1.0, dist_skew_low=-40.0, dist_skew_high=40.0, dist_skew_disc=4,
                 dist_var_low=0.0, dist_var_high=5.5, dist_var_disc=0.5):

        self.num_classes = num_classes
        self.num_fc_layers = num_fc_layers
        self.min_layer_filters = min_layer_filters

        if patch_size > 100:
            self.num_classifier_features = 4096
        else:
            self.num_classifier_features = 512

        self.layers_per_vgg_block = int((depth - 7) / 3)
        self.orig_filters = 2 * [64] + 2 * [128] + self.layers_per_vgg_block * [256] +\
                            2 * self.layers_per_vgg_block * [512] + 2 * [self.num_classifier_features]
        self.blocks = len(self.orig_filters)
        self.conv_filter_sizes = self.blocks * [conv_filter_size * conv_filter_size]
        self.total_filters = np.sum(self.orig_filters)

        self.dist_step = dist_step
        self.skews = np.arange(dist_skew_low, dist_skew_high + dist_skew_disc, dist_skew_disc)
        self.variances = np.arange(dist_var_low, dist_var_high + dist_var_disc, dist_var_disc)
        self.means = np.arange(0, self.blocks + dist_step, dist_step)

    # implemented according to https://en.wikipedia.org/wiki/Skew_normal_distribution
    def __skew_normal_univariate(self, x, xi, omega, alpha):
        phi = np.exp(-1 * np.square(x - xi) / (2 * np.square(omega)))
        PHI = 0.5 * (1 + spcl.erf((alpha * ((x - xi)/omega))/np.sqrt(2)))
        norm_fac = (2 / omega*(np.sqrt(2*math.pi))) / (2*np.pi)

        return norm_fac * phi * PHI

    # trapezoidal integration to discretize/bin distribution
    def __disc_integrate(self, data, intervals):
        integral_step = len(data) / float(intervals)
        integral = []

        for i in range(intervals):
            lower_bound = int(i * integral_step)
            upper_bound = int((i + 1) * integral_step)
            dx = 1.0 / (upper_bound - lower_bound)

            integral.append(np.trapz(data[lower_bound:upper_bound:1], np.arange(i, i + 1, dx)))

        return np.array(integral)

    # Calculate filter percentage from distribution
    def __dist_to_percent_filters_per_block(self, blocks, mean, var, skew, eps=0.05, discretization=0.01):
        x = np.arange(0, blocks, discretization)
        # pdf not defined for variance equal to zero (division by zero)
        if var == 0:
            return False, np.zeros(blocks)
        else:
            pdf = self.__skew_normal_univariate(x, mean, var, skew)
            bins = self.__disc_integrate(pdf, blocks)

            # we need to calculate the whole area in case of the distribution getting truncated
            # in such a case we will assign the left-over area to the end-points depending on the skew
            total_area = np.sum(bins)

            # check if the combination of parameters is valid from a loss of area point of view (truncation)
            if total_area <= (1 - eps):
                return False, bins
            else:
                return True, bins

    def __total_filters_per_layer(self, filters_percent, total_filters, min_layer_filters=2):
        filters_per_layer = np.ceil(filters_percent * total_filters)

        # if any layer has less than set amount of parameters then the architecture isn't valid
        valid = True
        for i in range(len(filters_per_layer)):
            if filters_per_layer[i] < min_layer_filters:
                valid = False

        return valid, filters_per_layer

    def __calculate_capacity(self, filters_per_layer, conv_filter_sizes, num_fc_layers=2, num_classes=10):
        total_params = 0

        if len(filters_per_layer) < 1:
            raise("no filters per layer found")

        for i in range(len(filters_per_layer)):
            if i == 0:
                total_params += filters_per_layer[i] * conv_filter_sizes[i] * 3
            elif 0 < i < (len(filters_per_layer) - num_fc_layers):
                total_params += filters_per_layer[i - 1] * filters_per_layer[i] * conv_filter_sizes[i]
            else:
                # For other architectures this is missing spatial dimensions of convolution output!
                total_params += filters_per_layer[i - 1] * filters_per_layer[i]

        # ultimate later mapping to classes
        total_params += filters_per_layer[-1] * num_classes

        return total_params

    def get_valid_models(self):
        valid_x = []
        valid_y = []
        valid_z = []
        layer_filters = []
        model_params = []

        for i in range(len(self.means)):
            for j in range(len(self.variances)):
                for k in range(len(self.skews)):
                    valid_area, layer_filters_percentage = self.__dist_to_percent_filters_per_block(self.blocks,
                                                                                                    self.means[i],
                                                                                                    self.variances[j],
                                                                                                    self.skews[k])
                    valid_params, filters = self.__total_filters_per_layer(layer_filters_percentage, self.total_filters,
                                                                           min_layer_filters=self.min_layer_filters)
                    total_params = self.__calculate_capacity(filters, self.conv_filter_sizes,
                                                             num_fc_layers=self.num_fc_layers,
                                                             num_classes=self.num_classes)

                    if valid_area and valid_params:
                        valid_x.append(self.means[i].tolist())
                        valid_y.append(self.variances[j].tolist())
                        valid_z.append(self.skews[k].tolist())
                        layer_filters.append(filters.tolist())
                        model_params.append(total_params.tolist())

        return {"filters": layer_filters, "total_params": model_params,
                "means": valid_x, "vars": valid_y, "skews": valid_z}
