import argparse
import torch
from math import sqrt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str)
    parser.add_argument("inp_dim", type=int)
    parser.add_argument("num_measurements", type=int)

    args = parser.parse_args()

    inp_dim = args.inp_dim
    num_measurements = args.num_measurements

    A = torch.randn((inp_dim, num_measurements), device='cuda') * (1/sqrt(num_measurements))
    torch.save(A, args.save_path)

if __name__ == "__main__":
    main()
