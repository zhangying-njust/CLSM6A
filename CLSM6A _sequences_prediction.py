import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
import sys


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
from pandas import DataFrame


class m_A549(nn.Module):
    def __init__(self):
        super(m_A549, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_brain(nn.Module):
    def __init__(self):
        super(m_brain, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_CD8T(nn.Module):
    def __init__(self):
        super(m_CD8T, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HCT116(nn.Module):
    def __init__(self):
        super(m_HCT116, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(64, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HEK293(nn.Module):
    def __init__(self):
        super(m_HEK293, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HEK293T(nn.Module):
    def __init__(self):
        super(m_HEK293T, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(3200, 128),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(128, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HeLa(nn.Module):
    def __init__(self):
        super(m_HeLa, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HepG2(nn.Module):
    def __init__(self):
        super(m_HepG2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(3200, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_kidney(nn.Module):
    def __init__(self):
        super(m_kidney, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(400, 128),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(128, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_liver(nn.Module):
    def __init__(self):
        super(m_liver, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_MOLM13(nn.Module):
    def __init__(self):
        super(m_MOLM13, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

def read_test_txt(data_path):

    seq_name = []
    seq = []

    with open(data_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if line.startswith('>'):
                line = line[0:-1]
                seq_name.append(line)
            elif line.strip() == "":
                pass
            else:
                line = line[0:201]
                seq.append(line)
    return seq_name, seq

def read_test_txt_to01_to0123(data_path):

    seq_name, seq = read_test_txt(data_path)

    d = {'AA': [1, 0, 0, 0],
         'AC': [0.5, 0.5, 0, 0],
         'AG': [0.5, 0, 0.5, 0],
         'AU': [0.5, 0, 0, 0.5],
         'CA': [0.5, 0.5, 0, 0],
         'CC': [0, 1, 0, 0],
         'CG': [0, 0.5, 0.5, 0],
         'CU': [0, 0.5, 0, 0.5],
         'GA': [0.5, 0, 0.5, 0],
         'GC': [0, 0.5, 0.5, 0],
         'GG': [0, 0, 1, 0],
         'GU': [0, 0, 0.5, 0.5],
         'UA': [0.5, 0, 0, 0.5],
         'UC': [0, 0.5, 0, 0.5],
         'UG': [0, 0, 0.5, 0.5],
         'UU': [0, 0, 0, 1]}

    nrows = len(seq)
    seq_len = len(seq[0])

    all_ENAC = []
    seq_cut = []
    for i in range(nrows):
        one_seq = seq[i]
        one_ENAC = []
        seq_start = 0
        seq_cut.append(seq[i][0:201])
        for jj in range(200):
            one_ENAC.append(d[one_seq[jj + seq_start:jj + 2 + seq_start]])
        all_ENAC.append(one_ENAC)

    return seq_name, seq, np.array(all_ENAC)

def main(data_path, model_path):

    [seq_name, seq, X_test] = read_test_txt_to01_to0123(data_path + jobid + ".txt")

    X_test = (torch.from_numpy(X_test)).to(torch.float32)
    test_dataset = Data.TensorDataset(X_test)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model.eval()
    pred_all = np.empty(0)
    with torch.no_grad():
        for [x_batch] in test_iter:

            output_batch = model(x_batch)
            pred_all = np.concatenate((pred_all, output_batch[:, 0].detach().cpu().numpy()), axis=0)





    df = DataFrame({'Name': seq_name, 'Sequence': seq, 'Score': pred_all.flatten()},
                   index=range(len(seq_name)))

    csv_path = data_path + jobid + 'results.csv'
    df.to_csv(csv_path, index=False, encoding='gbk',float_format='%.3f')

    # attentions = get_activations(m, seq_01, print_shape_only=True, layer_name='attention_vec')[0]

    return 0



if __name__ == '__main__':


    jobid = '01prediction'
    model_choose = 'liver'
    # 'A549', 'CD8T', 'HCT116', 'HEK293', 'HEK293T', 'HeLa', 'HepG2', 'MOLM13', 'brain', 'kidney', 'liver'

    if jobid.split() != "":
        print(jobid)
        # data path
        data_path = curPath + "/example/"
        # model path
        model_path = curPath + "/models/" + model_choose + ".pkl"
        # model_path = model_path.encode("utf-8").decode("utf-8")
        batch_size = 32

        global model
        exec('model = m_' + model_choose + '()')
        model.load_state_dict(torch.load(model_path))

        main(data_path=data_path, model_path=model_path)


