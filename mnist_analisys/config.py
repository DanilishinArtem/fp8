
class Config:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.01
        self.num_epochs = 3
        self.split = 0.8
        self.pathToData = "/home/adanilishin/fp8/mnist_analisys/mnist_dataset"
        self.pathToLogs = "/home/adanilishin/fp8/mnist_analisys/Logs/test"
        
        self.fault_time = []
        self.rate = 0
        # position = 7 (first bit flip), position = 6 (second bit flip)
        self.position = 6
        # self.target = "lin2"
        self.target = "6"

# 8, 6, 4, 2
# registered hook for layer relu1
# registered hook for layer conv2
# registered hook for layer relu2
# registered hook for layer pool
# registered hook for layer flatten
# registered hook for layer lin1
# registered hook for layer relu3
# registered hook for layer lin2
# registered hook for layer log_softmax
        