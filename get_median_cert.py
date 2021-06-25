import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cert_file", type=str, help="File containing certified radii")
parser.add_argument("--error_idx", type=int, default=2, help="Column index for smoothing error")
parser.add_argument("--eps_out_idx", type=int, default=1, help="Column index for certified output radius")
args = parser.parse_args()

if __name__ == "__main__":
    smoothing_error =[]
    eps_out = []
    with open(args.cert_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter='\t')
        i = 0
        for row in csvReader:
            if i == 0:
                print(row)
            if i > 1:
                eps_val = float(row[args.eps_out_idx])
                if eps_val >= 0:
                    smoothing_error.append(float(row[args.error_idx]))
                    eps_out.append(eps_val)
            i += 1

    smoothing_error = np.array(smoothing_error)
    # np.random.shuffle(smoothing_error)

    eps_out = np.array(eps_out)
    # np.random.shuffle(eps_out)

    median_smoothing_error = np.median(smoothing_error)
    median_eps_out = np.median(eps_out)

    print('Median smoothing error = %.3f' % median_smoothing_error)
    print('Median epsilon out = %.3f' % median_eps_out)
