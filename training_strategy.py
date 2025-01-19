import numpy as np
import os, torch


class EarlyStopping():
    '''
        保存当前为止最好的模型(loss最低)，
        当loss稳定不变patience个epoch时，结束训练
    '''

    def __init__(self, model_name, dataset_name, model_save_dir, patience=30, verbose=False, delta=0.0001):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_save_dir = model_save_dir

        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # 记录loss不变的epoch数目
        self.early_stop = False # 是否停止训练
        self.best_val_acc = -np.Inf
        self.delta = delta
        print('Create early stopping')

    def __call__(self, val_acc, model, optimizer, epoch):

        # 表现没有超过best
        if val_acc < self.best_val_acc + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        # 比best表现好
        else:
            self.save_checkpoint(val_acc, model, optimizer, epoch)
            self.counter = 0

    # 删除多余的权重文件
    def del_redundant_weights(self):
        # 删除已经有的文件,只保留n+1个模型
        num_saved = 4
        all_weights_temp = os.listdir(self.model_save_dir)
        all_weights = []
        for weights in all_weights_temp:
            if weights.endswith('.pth'):
                all_weights.append(weights)

        # 按存储格式来： save_name = f"netD_A-D1toD4-{epoch + 1:03d}-{val_acc:.4f}.pth"
        if len(all_weights) > num_saved:
            sorted = []
            for weight in all_weights:
                val_acc = weight.split('-')[-1]
                sorted.append((weight, val_acc))

            sorted.sort(key=lambda w: w[1], reverse=False)
            print('after sorting:', sorted)

            del_path = os.path.join(self.model_save_dir, sorted[0][0])
            os.remove(del_path)
            print('del file:', del_path)

    def save_checkpoint(self, val_acc, model, optimizer, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.best_val_acc:.4f} --> {val_acc:.4f}).  Saving model ...')

        self.del_redundant_weights()
        save_name = f"{self.model_name}-{self.dataset_name}-{epoch:03d}-{val_acc:.4f}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),

            'patience': self.patience,
            'counter': self.counter,
            'best_val_acc': self.best_val_acc
        }

        save_path = os.path.join(self.model_save_dir, save_name)

        # 存储权重
        torch.save(checkpoint, save_path)
        self.best_val_acc = val_acc


# if __name__ == '__main__':
#     fn = EarlyStopping(model_name='a', dataset_name='b', model_save_dir=r'C:\Users\wangj\Desktop\Chunchun\test_files')










