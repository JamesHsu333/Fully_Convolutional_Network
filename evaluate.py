import argparse
import itertools 
import logging
import os


import numpy as np
import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import model.net as net
import model.data_loader as data_loader

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/JPEGImages',
                    help="Directory containing the dataset")
parser.add_argument('--mask_dir', default='data/SegmentationClass',
                    help="Directory containing the mask dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--show_images', default='no', help="Show image")

def evaluate(model, loss_fn, dataloader, metrics, params):
    model.eval()

    summ = []
    args = parser.parse_args()

    for data_batch, labels_batch in dataloader:
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)

        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        labels_batch=labels_batch.long()
        
        output_batch = model(data_batch).float()

        loss = loss_fn(output_batch, labels_batch)

        output_batch = output_batch.data.cpu().numpy()
        data_batch = data_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # Show image
        if args.show_images != 'no':
            show_images(data_batch, labels_batch, output_batch)
            plt.show()

        summary_batch = {metric: metrics[metric](output_batch, labels_batch, 21)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

# Decode Segmentation Mask
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Show images
def show_images(data_images, labels_images, output_images):
    n=output_images.shape[0]
    for index, (data_image, label_image, output_image) in enumerate(zip(data_images, labels_images, output_images)):
        plt.subplot(n, 3, index*3+1)
        plt.title('input')
        plt.imshow(data_image.transpose(1,2,0))
        plt.axis('off')

        plt.subplot(n, 3, index*3+2)
        plt.title('label')
        plt.imshow(decode_segmap(label_image))
        plt.axis('off')
        
        output_image = output_image.transpose(1,2,0)
        output_image = np.argmax(output_image, axis=2)
        plt.subplot(n, 3, index*3+3)
        plt.title('prediction')
        plt.imshow(decode_segmap(output_image))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    params.batch_size = 5

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    dataloader = data_loader.fetch_dataloader(['test'], args.data_dir, args.mask_dir, params)

    test_dl = dataloader['test']

    logging.info("- done.")

    model = net.FCN(20+1).cuda() if params.cuda else net.FCN(20+1)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    evaluate(model, loss_fn, test_dl, metrics, params)

    logging.info("- done.")
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)