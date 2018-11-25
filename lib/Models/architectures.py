from collections import OrderedDict
import torch
import torch.nn as nn


def get_num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    
    return num_features


def get_feat_spatial_size(block, spatial_size, ncolors=3):
    x = torch.randn(1, ncolors, spatial_size, spatial_size)
    out = block(x)
    channels = out.size()[1]
    spatial_dim_x = out.size()[2]
    spatial_dim_y = out.size()[3]

    return channels, spatial_dim_x, spatial_dim_y


class ConvBlock(nn.Module):
    def __init__(self, fans, batch_norm=1e-5, kernel_size=3,
                 padding=1, stride=1, dropout=0.0, pool=True, pool_size=2, pool_stride=2):
        super(ConvBlock, self).__init__()

        self.gpu_params = self.gpu_activations = 0

        self.filter_dims = fans
        num_layers = len(self.filter_dims) - 1

        self.block = nn.Sequential(OrderedDict([
            ('conv_layer' + str(l+1), SingleConvLayer(l + 1, int(self.filter_dims[l]), int(self.filter_dims[l+1]),
                                                      kernel_size=kernel_size, padding=padding,
                                                      stride=stride, batch_norm=batch_norm))
            for l in range(num_layers)
        ]))

        for i in range(num_layers):
            self.gpu_params += 32 * kernel_size * kernel_size * self.filter_dims[i] * self.filter_dims[i+1]

        if pool:
            self.block.add_module('mp', nn.MaxPool2d(pool_size, pool_stride))

        if not dropout == 0.0:
            self.block.add_module('dropout', nn.Dropout2d(p=dropout, inplace=True))

    def get_gpu_usage(self, batch_size, image_size=None, no_channels=None):
        for layer in self.block:
            no_channels, image_size, _ = get_feat_spatial_size(layer, image_size, no_channels)
            self.gpu_activations += 32 * image_size * image_size * no_channels * batch_size
        return self.gpu_params, self.gpu_activations, image_size, no_channels

    def forward(self, x):
        x = self.block(x)
        return x


class SingleConvLayer(nn.Module):
    def __init__(self, l, fan_in, fan_out, kernel_size=3, padding=1, stride=1, batch_norm=1e-5):
        super(SingleConvLayer, self).__init__()

        self.convlayer = nn.Sequential(OrderedDict([
            ('conv' + str(l), nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                        bias=False))
        ]))

        if batch_norm > 0.0:
            self.convlayer.add_module('bn' + str(l), nn.BatchNorm2d(fan_out, eps=batch_norm))

        self.convlayer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.convlayer(x)
        return x


class ClassifierBlock(nn.Module):
    def __init__(self, fans, num_classes, batch_norm=1e-5, dropout=0.0):
        super(ClassifierBlock, self).__init__()

        self.num_classes = num_classes
        self.filter_dims = fans
        num_layers = len(self.filter_dims) - 1

        self.fc_block = nn.Sequential(OrderedDict([
            ('fc_layer' + str(l), SingleLinearLayer(l+1, int(self.filter_dims[l]), int(self.filter_dims[l+1]),
                                                    batch_norm=batch_norm))
            for l in range(num_layers)
        ]))

        if not dropout == 0.0:
            self.fc_block.add_module('dropout', nn.Dropout(p=dropout, inplace=True))

        self.fc_block.add_module('final_layer', nn.Linear(int(self.filter_dims[-1]), self.num_classes))

        self.gpu_params = 0
        if num_layers > 0:
            for i in range(num_layers):
                self.gpu_params += 32 * self.filter_dims[i] * self.filter_dims[i+1]
        self.gpu_params += 32 * self.filter_dims[-1] * self.num_classes

    def get_gpu_usage(self, batch_size):
        if len(self.filter_dims) > 1:
            self.gpu_activations = 32 * batch_size * (self.filter_dims[1] * (len(self.filter_dims) - 1) +
                                                      self.num_classes)
        else:
            self.gpu_activations = 32 * batch_size * self.num_classes
        return self.gpu_params, self.gpu_activations, None, None

    def forward(self, x):
        x = x.view(-1, get_num_flat_features(x))
        x = self.fc_block(x)
        return x


class SingleLinearLayer(nn.Module):
    def __init__(self, l, fan_in, fan_out, batch_norm=1e-5):
        super(SingleLinearLayer, self).__init__()

        self.fclayer = nn.Sequential(OrderedDict([
            ('fc' + str(l), nn.Linear(fan_in, fan_out, bias=False)),
        ]))

        if batch_norm > 0.0:
            self.fclayer.add_module('bn' + str(l), nn.BatchNorm1d(fan_out, eps=batch_norm))

        self.fclayer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.fclayer(x)
        return x


class VGG(nn.Module):
    def __init__(self, device, num_classes, num_colors, args, filters, custom_filters=False):
        super(VGG, self).__init__()

        batch_size = args.batch_size
        num_channels = num_colors
        image_size = args.patch_size

        self.batch_norm = args.batch_norm
        self.num_classes = num_classes
        self.device = device

        # subtract 3 classifier layers and first two static 2 layer blocks
        self.block_depth = args.vgg_depth - 7
        assert(self.block_depth % 3 == 0)
        self.layers_per_block = int(self.block_depth / 3)

        # classifier differentiation for ImageNet vs small images
        if args.patch_size > 100:
            self.num_classifier_features = 4096
        else:
            self.num_classifier_features = 512

        if custom_filters:
            self.filters = filters
            assert(len(self.filters) == (args.vgg_depth - 1))
        else:
            self.filters = 2 * [64] + 2 * [128] + self.layers_per_block * [256] + 2 * self.layers_per_block * [512] +\
                           2 * [self.num_classifier_features]

        self.features = nn.Sequential(OrderedDict([
            ('block1', ConvBlock([num_colors, self.filters[0], self.filters[1]], batch_norm=self.batch_norm)),
            ('block2', ConvBlock(self.filters[1:4], batch_norm=self.batch_norm)),
            ('block3', ConvBlock(self.filters[3:3+self.layers_per_block+1], batch_norm=self.batch_norm)),
            ('block4', ConvBlock(self.filters[3+self.layers_per_block:3+(2*self.layers_per_block)+1],
                                 batch_norm=self.batch_norm)),
            ('block5', ConvBlock(self.filters[3+(2*self.layers_per_block):3+(3*self.layers_per_block)+1],
                                 batch_norm=self.batch_norm))
        ]))

        _, self.feat_spatial_size_x, self.feat_spatial_size_y = get_feat_spatial_size(self.features,
                                                                                      args.patch_size,
                                                                                      ncolors=num_colors)
        self.classifier_feat_in = self.filters[3+(3*self.layers_per_block)] * \
                                  self.feat_spatial_size_x * self.feat_spatial_size_y
        self.classifier_droprate = 0.5

        self.classifier = nn.Sequential(
            ClassifierBlock([int(self.classifier_feat_in)] + self.filters[-2:], num_classes=self.num_classes,
                            batch_norm=self.batch_norm, dropout=self.classifier_droprate)
        )

        self.gpu_usage = 32 * batch_size * num_channels * image_size * image_size

        for conv_block in self.features:
            gpu_params, gpu_activations, image_size, num_channels = conv_block.get_gpu_usage(batch_size,
                                                                                             image_size,
                                                                                             num_channels)
            self.gpu_usage += (gpu_params + gpu_activations)

        gpu_params, gpu_activations, _, _ = self.classifier[0].get_gpu_usage(batch_size)

        self.gpu_usage += (gpu_params + gpu_activations)
        self.gpu_usage /= (8. * 1024 * 1024 * 1024)

    def encode(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        output = self.classifier(x)
        return output
