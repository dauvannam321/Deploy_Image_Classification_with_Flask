import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG16, self).__init__()

        self.feature_extractor = nn.Sequential(
            self.create_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2),
            self.create_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2),
            self.create_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2),
            self.create_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2),
            self.create_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2),
        )

        self.flatten = nn.Flatten() 

        self.FC_layer = nn.Sequential(
            self.create_fc_layer(7*7*512, 1028),
            self.create_fc_layer(1028, 512),
        )

        self.classifier = nn.Linear(in_features=512, out_features=n_classes)
    
    def forward(self, x):
        output = self.feature_extractor(x)
        output = self.flatten(output)
        output = self.FC_layer(output)
        output = self.classifier(output)

        return output

    
    def create_conv_layer(self, chann_in, chann_out, kernel_size, padding_size):
        layer = nn.Sequential(
            nn.Conv2d(chann_in, chann_out, kernel_size=kernel_size, padding=padding_size),
            nn.BatchNorm2d(chann_out),
            nn.ReLU()
        )

        return layer 
    
    def create_conv_block(self, list_chann_in, list_chann_out, list_kernel_size, list_padding_size, pooling_kernel, pooling_Stride):
        layers = [self.create_conv_layer(list_chann_in[i], list_chann_out[i], list_kernel_size[i], list_padding_size[i])
                    for i in range(len(list_chann_in))]
        
        layers += [nn.MaxPool2d(kernel_size=pooling_kernel, stride=pooling_Stride)]

        return nn.Sequential(*layers)
    
    def create_fc_layer(self, chann_in, chann_out):
        layer = nn.Sequential(
            nn.Linear(chann_in, chann_out),
            nn.ReLU()
        )

        return layer 