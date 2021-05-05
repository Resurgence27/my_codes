import numpy as np
import torchvision
import torch
import torch_geometric
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import mydatasets


class MagicPoint(nn.Module):
    def __init__(self, device, iterations, homo_num):
        super(MagicPoint, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(65)

        )
        self.dataset = torch_geometric.datasets.GeometricShapes("/datasets/geometric_data",
                                                                train=True, transform=transforms.ToTensor(),
                                                                pre_transform=torch_geometric.transforms.FaceToEdge,
                                                                pre_filter=None)
        self.train_dataset = self.dataset[:len(self.dataset)]
        self.test_dataset = self.dataset[:len(self.dataset)]

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.self.model = MagicPoint().to(device)

        self.iteration = iterations  # 200,000
        self.homo_trans_num = homo_num  # 100

    def forward(self, input_data):
        return self.net(input_data)

        # all_data = self.dataset.shuffle()
        # train_dataset = all_data[:len(all_data)]
        # test_dataset = all_data[:len(all_data)]
        # test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=32)
        train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=32)

    def train(self, loader, epoches):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

        self.model.train()
        loss_all = 0
        for data in loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / len(loader.dataset)

    def test(self, loader):
        self.model.eval()
        correct = 0
        for data in loader:
            data = data.to(self.device)
            pred = self.model(data).max(dim=1)[1]
            correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / len(loader.dataset)

    def show_result(self, epoch, train_loader, test_loader):
        loss = self.train(train_loader,epoch)
        train_acc = self.test(train_loader)
        test_acc = self.test(test_loader)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
              format(epoch, loss, train_acc, test_acc))

    # input:
    # img: COCO 2014
    # img number: 80000
    # repeat twice
    def generate_label(self):
        datasets.coco
        pass


# traindata: COCO
# evaldata: HPatches : 116 scenes with 696 unique images
class SuperPoint(nn.Module):
    def __init__(self):
        super(SuperPoint, self).__init__()
        self.ShareEnconder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.KeyPointDecoderHeader = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        )
        self.DescriptionDecoderHeader = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )
        self.keypointCheck = 0.05
        self.NMS_scope = 3

    # TODO: random homotransform
    # img size(240,320), gray
    def preprocess(self, img):
        H, W = img.shape[0], img.shape[1]
        input_data = img.copy()
        input_data = torch.from_numpy(input_data)
        input_data = torch.autograd.variable(input_data).view(1, 1, H, W)

    def simulate_train(self, input_data):
        common_data = self.ShareEnconder(input_data)
        keyPoint_data = self.KeyPointDecoderHeader(common_data)  # (N,65,H/8,W/8)
        description_data = self.DescriptionDecoderHeader(common_data)  # (N,256,H/8,W/8)

        # process keypoint
        softmax_data = np.exp(keyPoint_data)
        softmax_data = softmax_data / (np.sum(softmax_data, axis=0))
        softmax_64d = softmax_data[0:63, :, :]
        Hc = int(input_data.shape[3] / self.cell)
        Wc = int(input_data.shape[4] / self.cell)

        # get pixel along channel dimesion
        # transpose change the storage but reshape not
        # (N,C,H,W)->(N,H,W,C)
        softmax_64d = np.transpose(softmax_64d, [0, 2, 3, 1])
        # pixel along channel dimesion(1,C)->(cell,cell)
        softmax_64d = np.reshape(softmax_64d, [Hc, Wc, self.cell, self.cell])
        # index next cell starts from the W dimension pixel
        softmax_64d = np.transpose(softmax_64d, [Hc, self.cell, Wc, self.cell])
        pre_result = np.reshape(Hc * self.cell, Wc * self.cell)
        # check the point is keypoint in different channel,as it different feature
        rows, cols = np.where(pre_result > self.keypointCheck)
        # size:(3,m)
        keypoint_wait = np.zeros((3, len(pre_result[1])))
        keypoint_wait[0, :] = rows
        keypoint_wait[1, :] = cols
        keypoint_wait[2, :] = pre_result[rows, cols]
        sort_start_max = np.argsort(keypoint_wait)[::-1]
        keypoint_wait[sort_start_max]

        # NMS
        # -1: keep the pixel
        # 0 : surpress or empty
        # 1 : wait to be processed
        coordinate = keypoint_wait[:2, :]
        coordinate = coordinate.torch
        map = np.ones(input_data.shape[2], input_data.shape[3])
        index = np.ones(input_data.shape[2], input_data.shape[3])

        # todo: pad the map
        for enum, x_y in enumerate(coordinate):
            map[x_y[0], x_y[1]] = -1
            map[x_y[0]:x_y[0] + self.NMS_scope, x_y[1]:x_y[1] + self.NMS_scope] = 0
            index[x_y[0], x_y[1]] = enum
        # get remain point
        index_x, index_y = np.where(map < 0)
        keypoint = np.zeros(3, len(index_x))
        keypoint[0, :] = index_x
        keypoint[1, :] = index_y
        keypoint[2, :] = keypoint_wait[index_x, index_y]
        keypoint_index = np.argsort(keypoint)
        keypoint[:2, :] = keypoint[:2, keypoint_index]

        # TODO: process border pixel

        # process description
        description_data = self.DescriptionDecoderHeader(common_data)
        D = description_data.shape[1]

        # get (x,y)
        grid = torch.from_numpy(keypoint[:2, :].copy())
        # normalise to [-1,1]
        grid[0, :] = grid[0, :] / (float(input_data.shape[3]) / 2) - 1  # x
        grid[1, :] = grid[1, :] / (float(input_data.shape[2]) / 2) - 1  # y
        grid.contiguous()
        grid = grid.view(1, 1, -1, 2)

        # reverse uniform
        # grid:(N,H_out,W_out,2)
        # input:(N,C,H_in,W_in)
        # output:(N,C,H_out,W_out)
        description = torch.nn.functional.grid_sample(description_data, grid)
        description.reshape(D, -1)
        # L2_normalise
        description = description / np.linalg.norm(description, axis=0)
        return keypoint, description, pre_result

    # batch_size=32
    # ally loss : lamada= 0.0001
    # hinge loss:m_pos= 1, m_neg 0.2 lamada_d = 2500
    # ADAM: lr=0.001,beta=(0.9,0.999)
    def forward(self):
        pass
