class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #TODO: define the layers
        # 64 x 64
        self.conv1 = nn.Conv2d(1, 4, (5, 5), stride=1, padding=2, padding_mode=r"replicate") # 64 x 64 x 4
        self.pool1 = nn.MaxPool2d(2, stride=2) #32 x 32 x 4
        self.conv2 = nn.Conv2d(4, 8, (3, 3), stride=1, padding=1, padding_mode=r"replicate") # 32 x 32 x 8
        self.pool2 = nn.MaxPool2d(2, stride=2) #16 x 16 x 8
        self.conv3 = nn.Conv2d(8, 16, (3, 3), stride=1, padding=1, padding_mode=r"replicate") # 16 x 16 x 16
        self.pool3 = nn.MaxPool2d(2, stride=2) # 8 x 8 x 16
        self.conv4 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1, padding_mode=r"replicate") # 8 x 8 x 32
        self.pool4 = nn.MaxPool2d(2, stride=2) # 4 x 4 x 32
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 10)
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax()
        self.flat = nn.Flatten()
        self.dropout1 = nn.Dropout(p = 0.5)
        self.dropout2 = nn.Dropout(p = 0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(32)

    def forward(self, x):

        #TODO: run layer

        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act1(self.conv2(x))
        x = self.pool2(x)
        x = self.act1(self.conv3(x))
        x = self.pool3(x)
        x = self.act1(self.conv4(x))
        x = self.flat(self.pool4(x))
        x = self.act1(self.linear1(x))
        x = self.dropout1(self.batchnorm1(x))
        x = self.act1(self.linear2(x))
        x = self.dropout2(self.batchnorm2(x))
        output = self.linear3(x)
        return output