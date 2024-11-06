import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    # 画Loss曲线看收敛情况
    # 读取pth文件，获得loss_list
    checkpoint = torch.load('./checkpoints/v1.1-cfg/miniunet_49.pth')
    loss_list = checkpoint['loss_list']
    # print(loss_list)

    fig = plt.figure(figsize=(10, 7)) 
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1) 
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Ori Loss Curve')
    plt.tight_layout()
    
    fig.add_subplot(rows, columns, 2) 
    checkpoint = torch.load('./checkpoints/v1.1-cfg-modified-dropout-3e-1/miniunet_49.pth')
    loss_list = checkpoint['loss_list']
    # plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Modified Loss Curve')
    plt.tight_layout()

    # 画图
    
    plt.show()


    # 画图
    plt.plot(loss_list)