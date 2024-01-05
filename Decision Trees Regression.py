import math
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data_set.csv", delimiter = ",")    

test_data = data[151:273]
data = data[1:151]
x_test = test_data[:,0]
y_test = test_data[:,1]
Ntest = len(x_test)

global x_train, y_train, N

x_train = data[:,0]
y_train = data[:,1]
N = len(x_train)

P = 25

def regression_tree(P):
    node_indices = {}
    is_terminal = {}
    need_split = {}
    
    node_splits = {}
    node_mean = {}
    
    node_indices[1] = np.array(range(N))
    is_terminal[1] = False
    need_split[1] = True
    
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        
        if len(split_nodes) == 0:
            break
        
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean[split_node] = np.mean(y_train[data_indices])
            if len(data_indices) <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                values = np.sort(np.unique(x_train[data_indices]))
                split_points = (values[1:] + values[:-1]) / 2
                split_scores = np.repeat(0.0, len(split_points))
                
                i = 0
                for split in split_points:
                    av1 = np.mean([y_train[index] for index in data_indices if x_train[index] > split])
                    av2 = np.mean([y_train[index] for index in data_indices if x_train[index] <= split])
                    y_head = np.zeros(N)                   
                    for index in data_indices:
                        if x_train[index] > split:
                            y_head[index] = av1
                        else:
                            y_head[index] = av2
                    Error = np.sum(np.square(y_head[data_indices]-y_train[data_indices]))/len(data_indices)
                    split_scores[i] = Error
                    i +=1
                    
                split_point = split_points[np.argmin(split_scores)]
                node_splits[split_node] = split_point
                
                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices] > split_point]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
                
                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] <= split_point]
                node_indices[2 * split_node +1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    return [is_terminal, node_mean, node_splits]

[is_terminal, node_mean, node_splits] = regression_tree(P)
            
def predictor(is_terminal, node_mean, node_splits, N, x):
    y = np.repeat(0, N)
    for i in range(N):
        index = 1
        while True:
            if is_terminal[index] == True:
                y[i] = node_mean[index]
                break
            else:
                if x[i] > node_splits[index]:
                    index = 2 * index
                else:
                    index = 2 * index + 1
    return y

y_predicted = predictor(is_terminal, node_mean, node_splits, N, x_train)
RMSE_training = math.sqrt(np.sum(np.square(y_predicted-y_train))/N)
    
ytest_predicted = predictor(is_terminal, node_mean, node_splits, Ntest, x_test)    
RMSE_test = math.sqrt(np.sum(np.square(ytest_predicted-y_test))/N)

print('RMSE on training set is ', RMSE_training, ' when P is ', P,'\n')
print('RMSE on test set is ', RMSE_test, ' when P is ', P,'\n')

y = np.repeat(0, N)
X = np.linspace(1.5,5.1,1001)
values = np.repeat(0.0,len(X))
for i in range(len(X)):
    index = 1
    while True:
        if is_terminal[index] == True:
            values[i] = node_mean[index]
            break
        else:
            if X[i] > node_splits[index]:
                index = 2 * index
            else:
                index = 2 * index + 1

plt.figure()
plt.plot(x_test,y_test,"r.", markersize = 6, label = 'test')
plt.plot(x_train,y_train,"b.", markersize = 6, label = 'training')
plt.plot(X,values,"k-")
plt.legend()
plt.ylabel('Waiting time to next erruption(min)')
plt.xlabel('Erruption time (min)')
plt.show()
    
  
P_sizes = np.linspace(5,50,10)
RMSEs_training = []
RMSEs_test = []

for P in P_sizes:
    [is_terminal, node_mean, node_splits] = regression_tree(P)
    training = predictor(is_terminal, node_mean, node_splits, N, x_train)
    test =  predictor(is_terminal, node_mean, node_splits, Ntest, x_test)  
    RMSEs_training.append(math.sqrt(np.sum(np.square(training-y_train))/N))
    RMSEs_test.append(math.sqrt(np.sum(np.square(test-y_test))/N))
    
plt.figure()
plt.plot(P_sizes, RMSEs_training, "b-o", label = 'training')
plt.plot(P_sizes, RMSEs_test, "r-o", label = 'test')
plt.legend()
plt.xlabel('Pre-pruning size (P)')
plt.ylabel('RMSE') 
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
