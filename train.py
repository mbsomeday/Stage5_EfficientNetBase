import torch
from efficientnet_pytorch import EfficientNet
from training_strategy import EarlyStopping
from time import time
from torch.utils.data import DataLoader

from dataset import my_dataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100

def train_one_epoch(model, loss_fn, optimizer, epoch, train_dataset, train_loader):
    model.train()

    training_loss = 0.0
    training_correct_num = 0
    start_time = time()

    for batch, data in enumerate((train_loader)):
        # 将image和label放到GPU中
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        out = model(images)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, pred = torch.max(out, 1)
        training_correct_num += (pred == labels).sum()
        # break

    print(f'Training time for epoch:{epoch + 1}: {(time() - start_time):.2f}s, training loss:{training_loss:.6f}')
    return model


def val_model(model, loss_fn, val_dataset, val_loader):
    model.eval()
    val_loss = 0.0
    val_correct_num = 0

    with torch.no_grad():
        for data in (val_loader):
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images)
            loss = loss_fn(out, labels)
            _, pred = torch.max(out, 1)
            val_correct_num += (pred == labels).sum()
            val_loss += loss.item()

            # break

    val_accuracy = val_correct_num / len(val_dataset)
    val_acc_100 = val_accuracy * 100
    print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {})'.format(val_loss, val_acc_100, val_correct_num, len(val_dataset)))

    return val_loss, val_accuracy



def train(dataset_name, model_save_dir, train_dataset, train_loader, val_dataset, val_loader, reload=False, reload_weights=None):
    model_name = 'EfficientB0'
    early_stopping = EarlyStopping(model_name=model_name, dataset_name=dataset_name,
                                   model_save_dir=model_save_dir)

    model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
    if reload:
        checkpoints = torch.load(reload_weights, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoints['model_state_dict'])

        early_stopping.patience = checkpoints['patience']
        early_stopping.counter = checkpoints['counter']
        early_stopping.best_val_acc = checkpoints['best_val_acc']

    model = model.to(DEVICE)


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('-' * 20 + 'training Info' + '-' * 20)
    print('Total training Samples:', len(train_dataset))
    print('Total Batch:', len(train_loader))
    print('Total EPOCH:', EPOCHS)
    print('Runing device:', DEVICE)

    print('-' * 20 + 'Validation Info' + '-' * 20)
    print('Total Val Samples:', len(val_dataset))

    for epoch in range(EPOCHS):
        print('-' * 30 + 'begin EPOCH ' + str(epoch + 1) + '-' * 30)
        model = train_one_epoch(model, loss_fn, optimizer, epoch, train_dataset, train_loader)
        val_loss, val_accuracy = val_model(model, loss_fn, val_dataset, val_loader)

        # break

        # Early Stopping 策略
        early_stopping(val_acc=val_accuracy, model=model, optimizer=optimizer, epoch=epoch+1)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        # print('*' * 50)

if __name__ == '__main__':
    print('Start running ...')

    # ds_dir = r'D:\my_phd\dataset\Stage3\D2_CityPersons'
    batch_size = 64
    ds_dir = r'/veracruz/home/j/jwang/data/Stage4_D2_CityPersons_7Augs'

    train_dataset = my_dataset(ds_dir=ds_dir,txt_name='augmentation_train.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = my_dataset(ds_dir=ds_dir,txt_name='val.txt')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_save_dir = r'/veracruz/home/j/jwang/data/model_weights'
    train(dataset_name='D2', model_save_dir=model_save_dir,
          train_dataset=train_dataset, train_loader=train_loader,
          val_dataset=val_dataset, val_loader=val_loader,
          reload=False, reload_weights=None
          )













