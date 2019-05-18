import numpy as np

"""
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):

    #IMPLEMENT HERE
    losses = []
    for e in range(epoch):
        print(e)
        if shuffle:
            train = np.column_stack((x_train,y_train))
            np.random.shuffle(train)
            x_train = train[:,:-1]
            y_train = train[:,-1]
        loss = 0
        for i in range(len(x_train)//200):
            x_batch = x_train[i*200:(i+1)*200]
            y_batch = y_train[i*200:(i+1)*200]
            curr_loss = four_nn(x_batch,[w1,w2,w3,w4],[b1,b2,b3,b4],y_batch,False)
            loss += curr_loss
        losses.append(loss)
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    classes = four_nn(x_test,[w1,w2,w3,w4],[b1,b2,b3,b4],y_test,True)
    results = np.bincount(classes == y_test)
    tot_correct = results[1]
    avg_class_rate = tot_correct/len(classes)
    class_results = np.bincount((classes == y_test)*y_test)
    class_results[0] -= results[0]
    class_rate_per_class = class_results/(np.bincount(y_test))
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
"""
def four_nn(A,W,b,y,test):
    Z1, acache1 = affine_forward(A, W[0], b[0])
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, W[1], b[1])
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, W[2], b[2])
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3, W[3], b[3])
    if test == True:
        c = np.argmax(F,axis=1)
        return c
    loss, dF = cross_entropy(F, y)
    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1,rcache1)
    dX, dW1, db1 = affine_backward(dZ1, acache1)
    eta = 0.1
    W[0] -= eta*dW1
    W[1] -= eta*dW2
    W[2] -= eta*dW3
    W[3] -= eta*dW4
    b[0] -= eta*db1
    b[1] -= eta*db2
    b[2] -= eta*db3
    b[3] -= eta*db4
    A = A - eta*dX
    return loss

def affine_forward(A, W, b):
    Z = np.matmul(A,W) + b
    cache = (A, W, b)
    return Z, cache

def affine_backward(dZ, cache):
    dA = np.matmul(dZ,np.transpose(cache[1]))
    dW = np.matmul(np.transpose(cache[0]),dZ)
    dB = np.sum(dZ,axis=0)
    return dA, dW, dB

def relu_forward(Z):
    A = np.maximum(np.zeros(np.shape(Z)),Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    bools = np.zeros(np.shape(cache)) <= cache
    dZ = bools*dA
    return dZ

def cross_entropy(F, y):
    Fy = F[np.array(range(len(y))),y.astype(int)]
    loss = -(1/np.size(y))*(np.sum(Fy - np.log(np.sum(np.exp(F),axis=1))))
    bools = np.indices((np.shape(F)))
    bools = bools[1]
    bools = np.transpose(bools) == y
    bools = np.transpose(bools)
    sum_Fik = np.sum(np.exp(F),axis=1)
    sum_Fik = sum_Fik.reshape((len(sum_Fik),1))
    dF = -(1/np.size(y))*(bools - np.exp(F)/sum_Fik)

    return loss, dF
