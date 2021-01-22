import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.optim as optim
import numpy as np
import copy, pickle
import matplotlib.pyplot as plt


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


class one_layer_NN(nn.Module):
    def __init__(self, test_data, lr=0.01, moment=0.25, num_of_neurons=100, type_of_activation='Relu',
                 optimizer='Adam', epochs=100, batch_size=5, num_classes=2):
        super(one_layer_NN, self).__init__()
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.test_data = test_data
        self.activation = type_of_activation
        self.device = T.device('cuda')
        self.len = data[0][0].shape[1] * data[0][0].shape[1]
        self.fc1 = nn.Linear(self.len, num_of_neurons)
        self.fc2 = nn.Linear(num_of_neurons, num_classes)
        self.moment = moment
        self.optim_list = {'Adam': optim.Adam(self.parameters(), lr=self.lr),
                           'SGD': optim.SGD(self.parameters(), lr=self.lr, momentum=self.moment)}
        self.optimizer = self.optim_list[optimizer]
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, batch_data, type_of_activation):
        batch_data = T.tensor(batch_data).to(self.device)
        batch_data = batch_data.view(batch_data.size()[0], -1)
        batch_data = self.fc1(batch_data)
        list_activations = {'Relu': T.relu(batch_data), 'tanH': T.tanh(batch_data), 'Sigmoid': T.sigmoid(batch_data),
                            'Linear': batch_data}
        batch_data = list_activations[type_of_activation]
        classes = self.fc2(batch_data)
        return classes

    def testing_the_model(self,):
        self.eval()
        ep_loss = 0
        ep_acc = []
        data = T.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
        for j, (input, label) in enumerate(data):
            threshold = []
            label = label.to(self.device).long() - 1
            prediction = self.forward(input, self.activation)
            loss = self.loss(prediction, label)
            label = label.to(self.device).long()
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction.clamp_min(0.8)
            predict, pred = T.max(prediction, dim=1)
            for m in range(predict.size()[0]):
                if round(predict[m].item(), 2) == 0.8:
                    threshold.extend([-1])
                else:
                    threshold.extend([pred[m].item()])
            classes = T.tensor(threshold).to('cuda')
            wrong = T.where(classes != label,
                            T.tensor([1.]).to(self.device),
                            T.tensor([0.]).to(self.device))
            if self.batch_size > label.size()[0]:
                self.batch_size = label.size()[0]
            acc = 1 - T.sum(wrong) / self.batch_size
            ep_acc.append(acc.item())
            ep_loss += loss.item()
        print('total loss %.5f' % ep_loss, 'accuracy %.5f' % np.mean(ep_acc))
        return classes.item()

    def get_image(self, data):
        sample_im = data.to('cpu').numpy().transpose(1, 2)
        image = sample_im.clip(0, 1)
        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    def plot(self):
        plt.figure()
        plt.plot(self.loss_history, 'r')
        plt.plot(self.acc_comp)
        plt.title("loss/accuracy vs epochs")
        plt.xlabel("epochs")
        plt.ylabel("cumulative loss/ accuracy")
        plt.show()


class Two_layer_NN(nn.Module):
    def __init__(self, test_data, lr=0.001, moment=0.025, num_of_neurons=(250, 250), type_of_activation='Relu',
                 optimizer='Adam', epochs=100, batch_size=50, num_classes=8):
        super(Two_layer_NN, self).__init__()
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.test_data = test_data
        self.activation = type_of_activation
        self.device = T.device('cuda')
        self.len = data[0][0].shape[1] * data[0][0].shape[1]
        self.fc1 = nn.Linear(self.len, num_of_neurons[0])
        self.fc2 = nn.Linear(num_of_neurons[0], num_of_neurons[1])
        self.fc3 = nn.Linear(num_of_neurons[1], num_classes)
        self.moment = moment
        self.optim_list = {'Adam': optim.Adam(self.parameters(), lr=self.lr),
                           'SGD': optim.SGD(self.parameters(), lr=self.lr, momentum=self.moment)}
        self.optimizer = self.optim_list[optimizer]
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, batch_data, type_of_activation):
        batch_data = T.tensor(batch_data).to(self.device)
        batch_data = batch_data.view(batch_data.size()[0], -1)
        batch_data = self.fc1(batch_data)
        list_activations = {'Relu': T.relu(batch_data), 'tanH': T.tanh(batch_data), 'Sigmoid': T.sigmoid(batch_data),
                            'Linear': batch_data}
        batch_data = list_activations[type_of_activation]
        batch_data = self.fc2(batch_data)
        list_activations = {'Relu': T.relu(batch_data), 'tanH': T.tanh(batch_data), 'Sigmoid': T.sigmoid(batch_data),
                            'Linear': batch_data}
        batch_data = list_activations[type_of_activation]
        classes = self.fc3(batch_data)
        return classes

    def testing_the_model(self):
        self.eval()
        ep_loss = 0
        ep_acc = []
        data = T.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
        for j, (input, label) in enumerate(data):
            threshold = []
            label = label.to(self.device).long() - 1
            prediction = self.forward(input, self.activation)
            loss = self.loss(prediction, label)
            label = label.to(self.device).long()
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction.clamp_min(0.4)
            predict, pred = T.max(prediction, dim=1)
            for m in range(predict.size()[0]):
                if round(predict[m].item(), 2) == 0.4:
                    threshold.extend([-1])
                else:
                    threshold.extend([pred[m].item()])
            classes = T.tensor(threshold).to('cuda')
            wrong = T.where(classes != label,
                            T.tensor([1.]).to(self.device),
                            T.tensor([0.]).to(self.device))
            if self.batch_size > label.size()[0]:
                self.batch_size = label.size()[0]
            acc = 1 - T.sum(wrong) / self.batch_size
            ep_acc.append(acc.item())
            ep_loss += loss.item()
        print('total loss %.5f' % ep_loss, 'accuracy %.5f' % np.mean(ep_acc))
        return classes.item(), np.mean(ep_acc)

    def get_image(self, data):
        sample_im = data.to('cpu').numpy().transpose(1, 2)
        image = sample_im.clip(0, 1)
        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    def plot(self):
        plt.figure()
        plt.plot(self.loss_history, 'r')
        plt.plot(self.acc_comp)
        plt.title("loss/accuracy vs epochs")
        plt.xlabel("epochs")
        plt.ylabel("cumulative loss/ accuracy")
        plt.show()

def data_curation(data_file_name, labels_file_name, image_resolution):
    input_file = load_pkl(data_file_name)
    raw_data = [np.array(Image.fromarray(1.0 * np.array(input_file[i], dtype=np.bool)).resize(image_resolution))
                for i in range(len(input_file))]
    label = T.tensor(np.load(labels_file_name))
    reshaped_data = T.tensor(raw_data).reshape((len(input_file), 1, image_resolution[0], image_resolution[1]))
    labeled_data = list(zip(reshaped_data, label))
    random.shuffle(labeled_data)
    return labeled_data


if __name__ == '__main__':

    # values for different parameters that i used for training.
    lr = [0.001, 0.003, 0.01, 0.03, 0.1]
    momentum = [0.01, 0.025, 0.05, 0.075]
    dimensions = [25, 30, 40, 50]
    num_of_neurons = [(20, 20), (50, 50), (100, 100), (250, 250), (1000, 1000)]
    batch_size = [1, 5, 50, 1000, 6000]
    activation_functions = ['Relu', 'Sigmoid', 'tanH', 'Linear']
    optimizer = ['Adam', 'SGD']
    l, m, n, a, o, b = 1, 0, 2, 0, 0, 2

    # Enter the file name for testing data and testing labels.
    data = data_curation("train_data.pkl", "finalLabelsTrain.npy", (50, 50))

    hard_test = Two_layer_NN(test_data=data)
    hard_test.load_state_dict(T.load('trained_model_parameters'))
    x = hard_test.testing_the_model()
    print(x[1]
    # easy_test = one_layer_NN(test_data=data)
    # easy_test.load_state_dict(T.load('trained_model_parameters_2'))
    # easy_test.testing_the_model()


