import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default = 'train')
    parser.add_argument("--epoch", type = int, default = 200)
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--checkpoint", type = int, default = 10)
    parser.add_argument("--LR", type = float, default = 0.0001)
    parser.add_argument("--predict_epoch", type = int, default = 100)
    parser.add_argument("--predict_write", type = bool, default = False)
    parser.add_argument("--preload", type = str, default = None)
    parser.add_argument("--data_path", type = str, default = './image')
    parser.add_argument("--result_path", type = str, default = './result')
    parser.add_argument("--neighbor", type = int, default = 5)
    parser.add_argument("--model_mode", type = str, default = 'detection')
    args = parser.parse_args()
    return args