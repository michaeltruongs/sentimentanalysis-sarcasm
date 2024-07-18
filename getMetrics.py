def getMetrics(model, valid_dataloader, mode='accuracy'):

    sigmoid = nn.Sigmoid() 
    
    y_true = []
    y_pred = []
    for (x, x_lengths), y in valid_dataloader:
       # print(x, x_lengths)
        output = sigmoid(model(x, x_lengths))
        y_true = y_true + y.tolist()
        y_pred = y_pred + torch.squeeze(output).tolist()
        
    y_pred = list(map(lambda x: 0 if x < 0.5 else 1, y_pred))
    if mode == 'accuracy': #accuracy for model training
        accuracy = accuracy_score(y_true, y_pred)
        print('accuracy: {}'.format(accuracy))
        return accuracy
    elif mode == 'f1':
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
        print('f1: {}'.format(f1))
        return f1
    