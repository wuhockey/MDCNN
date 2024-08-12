# from models import ViT
from torchvision import transforms
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# from einops import rearrange, repeat
# from info_nce import InfoNCE
from mydataset import My_Dataset
from torch.utils.data import DataLoader, Dataset
from model.merge import MergedNN
from model.cnn import CNN


def main(epoch=20, LR=0.0001, batch_size=8):

    # step one: select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # step two: train/test set loader
    train_dataset = My_Dataset('datasets/trainset')
    val_dataset = My_Dataset('datasets/valset')
    test_dataset = My_Dataset('datasets/testset')
    train_dataloader= DataLoader(train_dataset)
    val_dataloader = DataLoader(val_dataset)
    test_dataloader = DataLoader(test_dataset)


    # step three: model, loss optimizer
    model = CNN(base_line = True).to(device)
    cross_entropy_loss =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)

    # step four: train lop
    best_acc = 0.0
    print('train start')

    for step, epoch_num in enumerate(range(epoch)):
        total_loss = 0
        train_correct = 0
        val_correct_top5 = 0
        val_correct = 0
        flag = 0

        # train_model
        model.train()
        print(len(train_dataloader))
        for num, data in enumerate(train_dataloader):
            if num % 100 == 0:
                print(num, "/", len(train_dataloader))
            result = model(data[0].to(device))
            loss = cross_entropy_loss (result,data[2].to(device))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predict = result.argmax(dim=1)
            with torch.no_grad():
                train_correct += (predict==data[2].to(device)).sum()
        lr_scheduler.step()
        train_acc = train_correct / (len(train_dataloader) * batch_size)

        writer.add_scalar('epoch_loss', total_loss, epoch_num)
        print('epoch:', epoch_num, 'total_loss : {:.4f}'.format(total_loss), 'train_acc : {:.4f} %'.format(100 * train_acc))

        # test model
        model.eval()
        with torch.no_grad():
            for num, data in enumerate(val_dataloader):
                result = model(data[0].to(device))
                predict = result.argmax(dim=1)
                val_correct += (predict == data[2].to(device)).sum()
                # fpn = rearrange(val_product_image_n.to(device), 'b c (h p1) w -> b h (p1 w c)', p1=patch_size)
                # fpn = model.outfit_to_embedding(fpn)
                # fpn += model.z_pos_embedding
                # fpn = model.transformer_y(fpn, None)
                # fpn = fpn.mean(dim=1)
                # fpn = model.to_latent(fpn)
                # fpn = model.mlp_head(fpn)

            val_acc = val_correct / (len(val_dataloader) * batch_size)
            if val_acc > best_acc:
                flag = 1
                best_acc = val_acc
                torch.save(model.state_dict(), 'ours_best_model_params.pth')

            test_correct = 0
            for num, data in enumerate(test_dataloader):

                result = model(data[0].to(device))
                predict = result.argmax(dim=1)
                test_correct += (predict == data[2].to(device)).sum()

            import random
            test_acc = test_correct / (len(test_dataloader) * batch_size)
            print('epoch:', epoch_num, 'total_loss : {:.4f}'.format(total_loss),
                  'train_acc : {:.4f} %'.format(100 * train_acc),
                  'val_acc : {:.4f} %'.format(100 * val_acc),
                  'test_acc_FITB : {:.4f} %'.format(100 * test_acc))

            with open("result.txt", "a") as f:
                if flag == 1:
                    f.write("best ")
                f.write("epoch : {}".format(epoch_num) + "\n"
                        + "total_loss : {:.4f}".format(total_loss) + "\n"
                        + "train_acc : {:.4f} %".format(100 * train_acc) + "\n"
                        + "val_acc : {:.4f} %".format(100 * val_acc) + "\n"
                        + "test_acc_FITB : {:.4f} %".format(100 * test_acc) + "\n")
                f.write("\n")

main()