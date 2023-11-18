import matplotlib.pyplot as plt
import torch
import os


def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(10, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# create output folder if dosn't exist then create model_info.txt file and then write model
# info and save our correspond model in .pth file

folder_name = 'E:\\Programming\\Per\\Python\\Uni_Projects\\Neural_Networks\\mnist_project\\output'


def save_model(model, results, hidden_layers, epochs, lr):

    file_name = 'model_info.txt'
    info_file_path = os.path.join(folder_name, file_name)

    model_info = f'layers = {hidden_layers} | epochs:{epochs} | lr:{lr} | accuracy:{results["test_acc"][-1]}'

    if os.path.exists(folder_name):

        if os.path.exists(info_file_path):

            with open(info_file_path, 'a') as file:
                file.write(model_info + '\n')

            torch.save(model.state_dict(), os.path.join(
                folder_name, str(hidden_layers)+'.pth'))

        else:

            with open(info_file_path, 'w') as file:
                file.write(model_info + '\n')

            torch.save(model.state_dict(), os.path.join(
                folder_name, str(hidden_layers)+'.pth'))

    else:
        os.makedirs(folder_name)

        with open(info_file_path, 'w') as file:
            file.write(model_info + '\n')

        torch.save(model.state_dict(), os.path.join(
            folder_name, str(hidden_layers)+'.pth'))

    return 'Files and infos saved'
