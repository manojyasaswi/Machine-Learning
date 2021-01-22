import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics
import random
import time, copy, pickle
from PIL import Image


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


class NN(nn.Module):
    def __init__(self, train_data, test_data, lr=0.001, moment=0.025, num_of_neurons=(250, 250), type_of_activation='Relu',
                 optimizer='Adam', epochs=100, batch_size=50, num_classes=8):
        super(NN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.train_data = train_data
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

    def training_the_model(self):
        self.train()
        self.acc_comp = []
        data = T.utils.data.DataLoader(self.train_data, batch_size=self.batch_size)
        for i in range(self.epochs):
            if i > 1 and self.acc_comp[i - 1] - self.acc_comp[i - 2] == 0:
                print(f"number for epochs = {0}", i)
                break
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(data):
                self.optimizer.zero_grad()
                label = label.to(self.device).long() - 1
                prediction = self.forward(input, self.activation)
                loss = self.loss(prediction, label)
                label = label.to(self.device).long()
                prediction = F.softmax(prediction, dim=1)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device))
                if self.batch_size > label.size()[0]: self.batch_size = label.size()[0]
                acc = 1 - T.sum(wrong) / self.batch_size
                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            self.acc_comp.extend([np.mean(ep_acc)])
            print('Finish epoch ', i, 'total loss %.3f' % ep_loss, 'accuracy %.3f' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)
        self.plot()
        return copy.deepcopy(NN.state_dict(self))

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
        return classes

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
        plt.title("loss/accuracy vs epochs")
        plt.xlabel("epochs")
        plt.ylabel("cumulative loss/ accuracy")
        plt.show()
        plt.figure()
        plt.title("accuracy vs epochs")
        plt.plot(self.acc_comp)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()


def data_curation(data_file_name, labels_file_name, image_resolution):
    input_file = load_pkl(data_file_name)
    raw_data = [np.array(Image.fromarray(1.0 * np.array(input_file[i], dtype=np.bool)).resize(image_resolution))
                for i in range(len(input_file))]
    label = T.tensor(np.load(labels_file_name))
    reshaped_data = T.tensor(raw_data).reshape((len(input_file), 1, image_resolution[0], image_resolution[1]))
    labeled_data = list(zip(reshaped_data, label))
    random.shuffle(labeled_data)
    # save_pkl("labeled_data_{0}".format(image_resolution[0]), labeled_data)
    # print("Labeled and shuffled data is stored in the file 'labeled_data_{0}'".format(image_resolution[0]))
    return labeled_data

def experiments():
    lr = [0.001, 0.003, 0.01, 0.03, 0.1]
    momentum = [0.01, 0.025, 0.05, 0.075]
    dimensions = [25, 30, 40, 50]
    num_of_neurons = [(20, 20), (50, 50), (100, 100), (250, 250), (1000, 1000)]
    batch_size = [1, 5, 50, 1000, 6000]
    activation_functions = ['Relu', 'Sigmoid', 'tanH', 'Linear']
    optimizer = ['Adam', 'SGD']
    l, m, n, a, o, b = 1, 0, 2, 0, 0, 2
    data = data_curation("train_data.pkl", "finalLabelsTrain.npy", (50, 50))
    list_labels = [i for i in range(len(data))]
    train_index = random.sample(list_labels, int(len(list_labels) / 1.05) - 95)  # 20-80
    test_index = [i for i in range(len(data)) if i not in train_index]
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    list_of_accuracy = []
    list_of_accuracy_per_value = []
    list_of_high_acc_parameter = []

    for a in range(len(activation_functions)):
        start_time = time.time()
        net = NN(lr=lr[l], moment=momentum[m], num_of_neurons=num_of_neurons[n],
                  type_of_activation=activation_functions[a], optimizer=optimizer[o], epochs=100,
                  batch_size=batch_size[b], num_classes=8, train_data=train_data, test_data=test_data)
        
        model_parameters = net.training_the_model(train_data)
        accuracy = net.testing_the_model(test_data)
        end_time = time.time()
        print("\n\n the accuracy with activation function parameter of value {0} is {1} in time {2} \n\n".format(
            activation_functions[a], accuracy, end_time-start_time))
        list_of_accuracy_per_value.extend([accuracy])
    
    a = np.argmax(list_of_accuracy_per_value)
    print("The highest accuracy is achieved with activation function parameter is for: {0}\n\n".format(
        activation_functions[a]))
    list_of_high_acc_parameter.extend([str(activation_functions[a])])
    list_of_accuracy_per_value = []

    for b in range(len(batch_size)):
        start_time = time.time()
        net = NN(lr=lr[l], moment=momentum[m], num_of_neurons=num_of_neurons[n],
                  type_of_activation=activation_functions[a], optimizer=optimizer[o], epochs=100,
                  batch_size=batch_size[b], num_classes=8, train_data=train_data, test_data=test_data)
        model_parameters = net.training_the_model(train_data)
        accuracy = net.testing_the_model(test_data)
        end_time = time.time()
        print("\n\n the accuracy with batch size parameter of value {0} is {1} in time{2}\n\n".format(batch_size[b], accuracy, end_time-start_time))
        list_of_accuracy_per_value.extend([accuracy])
    
    b = np.argmax(list_of_accuracy_per_value)
    print("The highest accuracy is achieved with value parameter batch_size for the value: {0}\n\n".format(
        batch_size[b] ))
    list_of_high_acc_parameter.extend([str(batch_size[b])])


    list_of_accuracy_per_value = []
    for n in range(len(num_of_neurons)):
        start_time = time.time()
        net = NN(lr=lr[l], moment=momentum[m], num_of_neurons=num_of_neurons[n],
                  type_of_activation=activation_functions[a], optimizer=optimizer[o], epochs=100,
                  batch_size=batch_size[b], num_classes=8, train_data=train_data, test_data=test_data)
       
        model_parameters = net.training_the_model(train_data)
        accuracy = net.testing_the_model(test_data)
        end_time = time.time()
        print("\n\n the accuracy with batch_size parameter of value {0} is {1} in time{2}\n\n".format(num_of_neurons[n],
                                                                                                accuracy, end_time-start_time))
        list_of_accuracy_per_value.extend([accuracy])
        print(n, len(num_of_neurons))
    
    n = np.argmax(list_of_accuracy_per_value)
    print(n)
    print("The highest accuracy is achieved with value parameter number of neurons for the value: {0}\n\n".format(num_of_neurons[n]))
    list_of_high_acc_parameter.extend([str(num_of_neurons[n])])


    list_of_accuracy_per_value = []
    for l in range(len(lr)):
        start_time = time.time()
        net = NN(lr=lr[l], moment=momentum[m], num_of_neurons=num_of_neurons[n],
                  type_of_activation=activation_functions[a], optimizer=optimizer[o], epochs=100,
                  batch_size=batch_size[b], num_classes=8, train_data=train_data, test_data=test_data)
        model_parameters = net.training_the_model(train_data)
        accuracy = net.testing_the_model(test_data)
        end_time = time.time()
        print("\n\n the accuracy with batch size parameter of value {0} is {1} in time {2}\n\n".format(lr[l], accuracy, end_time-start_time))
        list_of_accuracy_per_value.extend([accuracy])
    
    l = np.argmax(list_of_accuracy_per_value)
    print("The highest accuracy is achieved with value parameter activation function for the value: {0}\n\n".format(
        lr[l]))
    list_of_high_acc_parameter.extend([str(lr[l])])


    list_of_accuracy_per_value = []
    for o in range(len(optimizer)):
        start_time = time.time()
        net = NN(lr=lr[l], moment=momentum[m], num_of_neurons=num_of_neurons[n],
                  type_of_activation=activation_functions[a], optimizer=optimizer[o], epochs=100,
                  batch_size=batch_size[b], num_classes=8, train_data=train_data, test_data=test_data)
       
        model_parameters = net.training_the_model(train_data)
        accuracy = net.testing_the_model(test_data)
        end_time = time.time()
        print("\n\n the accuracy with batch size parameter of value {0} is {1} in time {2}\n\n".format(optimizer[o], accuracy, end_time-start_time))
        list_of_accuracy_per_value.extend([accuracy])
    
    o = np.argmax(list_of_accuracy_per_value)
    print("The highest accuracy is achieved with value parameter activation function for the value: {0}\n\n".format(
        optimizer[o]))
    list_of_high_acc_parameter.extend([str(optimizer[o])])

    print("The accuracy for the combination {0} is {1}".format(" - ".join(list_of_high_acc_parameter),
                                                               np.max(list_of_accuracy_per_value)))
    return activation_functions[a], momentum[m], lr[l], num_of_neurons[n], batch_size[b], optimizer[o]

if __name__ == '__main__':
    data = data_curation("train_data.pkl", "finalLabelsTrain.npy", (50, 50))
    list_labels = [i for i in range(len(data))]
    train_index = random.sample(list_labels, int(len(list_labels) / 1.05) - 95)  # 20-80
    test_index = [i for i in range(len(data)) if i not in train_index]
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    labels=[test[1].item() for test in test_data]

    list_of_accuracy = []
    list_of_accuracy_per_value = []
    list_of_high_acc_parameter = []
    network = NN(train_data=train_data, test_data=test_data)
    copy_of_weights= network.training_the_model()
    T.save(network.state_dict(), 'trained_model_parameters_2')
    classes = network.testing_the_model()
    network.testing_the_model()
