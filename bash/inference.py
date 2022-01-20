from PIL import Image
from torchvision import transforms

from src.models.plant_module import PlantCls


def predict():
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    # CKPT_PATH = "last.ckpt"
    CKPT_PATH = "/nfs2/personal/cmpark/dacon/dacon-template/logs/runs/2022-01-19/05-55-05/checkpoints/epoch_008.ckpt"
    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = PlantCls.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("data/example_img.png").convert("L")  # convert to black and white
    # img = Image.open("data/example_img.png").convert("RGB")  # convert to RGB

    # preprocess
    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    img = mnist_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


if __name__ == "__main__":
    predict()
