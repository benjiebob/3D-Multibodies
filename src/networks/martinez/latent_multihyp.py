
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel_LatentEncoding_MultiHyp(nn.Module):
    def __init__(self,
                # batch_size,
                 linear_size=1024,
                 num_stage=2,
                 num_modes=5,
                 p_dropout=0.5,
                inputDim=48,        #16*3
                outputDim=85,       #19*3 +10
                b_optShape=False,    #if true, optimize shape as well
                 ):
        super(LinearModel_LatentEncoding_MultiHyp, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        # self.input_size =  15 * 2
        self.input_size =   inputDim #inputJointNum* 2
        # 3d joints
        self.output_size = outputDim * num_modes #outputJointNum * 3       #No 9, 8

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.b_optShape  = b_optShape

    def forward(self, x):

        # print("batchSize: {}".format(x.shape[0]))
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)          #(N, 85) #Pose 72, shape 10, scale 1, trans 2        
        return y.contiguous()

class LinearModel_CVAE(nn.Module):
    def __init__(self,
                # batch_size,
                 linear_size=1024,
                 num_stage=2,
                 num_modes=5,
                 p_dropout=0.5,
                inputDim=48,        #16*3
                outputDim=85,       #19*3 +10
                b_optShape=False,    #if true, optimize shape as well
                 ):
        super(LinearModel_CVAE, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        # self.input_size =  15 * 2
        self.input_size =   inputDim #inputJointNum* 2
        # 3d joints
        self.output_size = outputDim * num_modes #outputJointNum * 3       #No 9, 8

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        # self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.b_optShape  = b_optShape

        self.fc_mu = nn.Linear(self.linear_size, self.output_size)
        self.fc_logvar = nn.Linear(self.linear_size, self.output_size)

    def forward(self, x):

        # print("batchSize: {}".format(x.shape[0]))
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        mu = self.fc_mu(y)
        log_variance = self.fc_logvar(y)
        
        return mu, log_variance