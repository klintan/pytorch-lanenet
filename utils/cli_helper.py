import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--save", required=False, help="Directory to save model checkpoint", default="./checkpoints")
    parser.add_argument("--epochs", required=False, type=int, help="Training epochs", default=10)
    parser.add_argument("--pretrained", required=False, default=None, help="pretrained model path")
    parser.add_argument("--image", default="./output", help="output image folder")
    parser.add_argument("--net", help="backbone network")
    parser.add_argument("--json", help="post processing json")
    return parser.parse_args()
