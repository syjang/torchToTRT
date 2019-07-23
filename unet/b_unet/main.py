import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import csv
import os
from advanced_model import CleanU_Net


# def test_model(model_path, data_test, epoch, save_folder_name='prediction'):
#     """
#         Test run
#     """
#     for batch, (images_t) in enumerate(data_test):
#         stacked_img = torch.Tensor([]).cuda()
#         for index in range(images_t.size()[1]):
#             with torch.no_grad():
#                 image_t = Variable(images_t[:, index, :, :].unsqueeze(0).cuda())
#                 # print(image_v.shape, mask_v.shape)
#                 output_t = model(image_t)
#                 output_t = torch.argmax(output_t, dim=1).float()
#                 stacked_img = torch.cat((stacked_img, output_t))
#         im_name = batch  # TODO: Change this to real image name so we know
#         # _ = save_prediction_image(stacked_img, im_name, epoch, save_folder_name)
#     print("Finish Prediction!")




def main():
    model_path = "model/model_epoch_dict_400.pwf"
    # model = torch.load(model_path)
    # model = torch.nn.DataParallel(model, device_ids=list(
        #  range(torch.cuda.device_count()))).cuda()
    model = CleanU_Net(1,2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    example = torch.randn(1, 1, 572, 572)
    traceModel = torch.jit.trace(model, example)
    traceModel.save("unet.pt")
    outputnames = ["output"]
    torch.onnx.export(model,example,"unet.onnx",verbose=True ,output_names=outputnames)

    print("Finish Job!")
    return 0


if __name__ == "__main__":
    main()
    pass