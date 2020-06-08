import torch
from torch import nn, optim
from torch.nn import functional as F
import PIL
from PIL import Image
from torchvision import transforms, models
from matplotlib import pyplot as plt

import copy


from config import INPUT_IMG, CONTENT_IMG, STYLE_IMG


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

imsize = 128 # 输出图片的大小

transform = transforms.Compose([
    transforms.Resize((imsize, imsize)), # 注意，这里必须一样，否则图片会是等比例缩放
    transforms.ToTensor()
])

'''
img_name 图片路径，字符串
return 图片数据Tensor
'''
def img_loader(img_name):
    img = Image.open(img_name)
    img = transform(img)
    img = img.unsqueeze(0) # CNN处理4维数据，所以必须在之前添加一个维度
    # print(img.size()) # torch.Size([1, 3, 128, 128])
    return img

style_img = img_loader(STYLE_IMG).to(device)
content_img = img_loader(CONTENT_IMG).to(device)


# 重新转为PIL图片，看看是否导入成功
unloader = transforms.ToPILImage()

plt.ion() # 打开交互模式，可以同时打开多个窗口显示图片 在shou之前必须ioff关闭

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(1) # 和plt.show类似，显示10秒，然后关闭窗口

plt.figure()
imshow(style_img, title='StyleImage')

plt.figure()
imshow(content_img, title="ContentImage")
#
# plt.ioff()
# plt.show()


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()

        self.target = target.detach()


    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)

        return input


def gram_matrix(input):
    a, b ,c, d = input.size()
    '''
    a:batch size b:特征图数量，可以理解为通道数量 c,d:特征图维度，可以认为是图片宽高
    '''
    features = input.view(a*b, c*d) # a*b个特征图
    G = torch.mm(features, features.t()) #
    return G.div(a*b*c*d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target  = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)

        return input

'''
pytorch的VGG把模型分为了两个Sequential,一个是features（包含卷积池化），一个是classifier（包含全连接层）
需要features，因为需要计算style loss 和content loss
某些层在训练和测试的行为不同，所以需要用.eval()方法
'''
cnn = models.vgg19(pretrained=True).features.to(device).eval()


mean = torch.tensor([0.485,0.456,0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean)/self.std

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_loss(cnn, mean, std, style_img, content_img,
                   content_layers=content_layers, style_layers=style_layers):
    cnn = copy.deepcopy(cnn)
    norma = Normalization(mean, std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(norma)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)

        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else :
            raise RuntimeError("unrecongnized layer:{}".format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)

            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1, -1, -1): # 从模型最末尾开始，到模型开头
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]  # 也就是取了模型开头到Content或者Style层，后面的都不要了
    return model, style_losses, content_losses
# here style_img = img_loader("picasso.jpg").to(device)
input_img = copy.deepcopy(content_img).to(device) # 少用copy()浅拷贝，减少bug
# input_img = img_loader(INPUT_IMG).to(device)
# plt.figure()
#
# imshow(input_img, title='Input Image')


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    return optimizer

def run_style_transfer(cnn, mean, std, content_img, style_img,
                       input_img, num_steps=300, style_weight=100000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_loss(cnn, mean, std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()

            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


output = run_style_transfer(cnn, mean, std, content_img, style_img, input_img)
plt.figure()
imshow(output, title="Output Image")

plt.ioff()

plt.show()








