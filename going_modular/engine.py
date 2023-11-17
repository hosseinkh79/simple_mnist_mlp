import torch
from torch import nn
from tqdm.auto import tqdm


#Train and test 
def one_step_train(model, train_dataloader, loss_fn, optimizer, device):
    model = model.to(device)

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X.view(X.size(0), -1))
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += ((y_pred_class == y).sum().item())/len(y_pred)

    train_loss = train_loss/len(train_dataloader)
    train_acc = train_acc/len(train_dataloader)

    return train_loss, train_acc



def one_step_test(model, test_dataloader, loss_fn, device):
    model = model.to(device)

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X.view(X.size(0), -1))
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += ((y_pred_class == y).sum().item())/len(y_pred)

        test_loss = test_loss/len(test_dataloader)
        test_acc = test_acc/len(test_dataloader)

    return test_loss, test_acc
        

def train(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          device,
          epochs):
    
    results = {
            'train_loss':[],
            'train_acc':[],
            'test_loss':[],
            'test_acc':[]
        }
    
    for epoch in range(epochs):

        train_loss, train_acc = one_step_train(model,
                                                train_dataloader,
                                                loss_fn, optimizer,
                                                device)

        test_loss, test_acc = one_step_test(model,
                                            test_dataloader,
                                            loss_fn,
                                            device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
        
    return results

