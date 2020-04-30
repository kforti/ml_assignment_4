from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter

from problem_4_3.utils import *
from problem_4_3.models import *

writer = SummaryWriter("runs/mnist")

num_epochs = 15
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])


trainset = MNIST('./MNIST_TRAINSET', download=True, train=True, transform=img_transform)
valset = MNIST('./MNIST_TESTSET', download=True, train=False, transform=img_transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=True)

examples = iter(valloader)
example_data, example_targets = examples.next()

# Autoencoder
model = LinearAutoencoder()
writer.add_graph(model, example_data.view(example_data.size(0), -1))
writer.close()
m_state = torch.load('./autoencoder.pth')
model.load_state_dict(m_state)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

auto_path = "autoencoder.pth"

#model = train_autoencoder(model, criterion, optimizer, num_epochs, trainloader, 'autoencoder', auto_path, writer)
#test_autoencoder(valloader, model, "./test_autoencoder")
#sys.exit()
writer = SummaryWriter("runs/regular_classifier")
# Regular Classifier
classify_model = Linearclassifier()

classify_criterion = nn.NLLLoss()
classify_optimizer = optim.SGD(classify_model.parameters(), lr=0.003, momentum=0.9)

classify_model = train_classifier(classify_model, classify_criterion, classify_optimizer, num_epochs, trainloader, 'classifier',
                                  'regular_classifier.pth', writer=writer)
test_classifier(valloader, classify_model)

# Classifier from Autoencoder
writer = SummaryWriter("runs/pre_trained_classifier")
classify_model = Linearclassifier.from_autoencoder(auto_path)

classify_criterion = nn.NLLLoss()
classify_optimizer = optim.SGD(classify_model.parameters(), lr=0.003, momentum=0.9)

classify_model = train_classifier(classify_model, classify_criterion, classify_optimizer, num_epochs, trainloader, 'classifier',
                                  'autencoder_classifier.pth', count=94, writer=writer)
test_classifier(valloader, classify_model)

# Regular Classifier minimal data
writer = SummaryWriter("runs/min_regular_classifier")
classify_model = Linearclassifier()

classify_criterion = nn.NLLLoss()
classify_optimizer = optim.SGD(classify_model.parameters(), lr=0.003, momentum=0.9)

classify_model = train_classifier(classify_model, classify_criterion, classify_optimizer, num_epochs, trainloader, 'classifier',
                                  'min_classifier.pth', count=94, writer=writer)
test_classifier(valloader, classify_model)