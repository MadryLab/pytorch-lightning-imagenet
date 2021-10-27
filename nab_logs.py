import argparse
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', required=True)
parser.add_argument('--ffcv-out-path', required=True)

def main():
    args = parser.parse_args()

    expected_str = 'lightning_logs/version_0/'
    assert args.logdir[-len(expected_str):] == expected_str
    print(f'Extracting from: {Path(args.logdir).absolute()}')

    ea = event_accumulator.EventAccumulator(args.logdir)
    ea.Reload()
    time = ea.Scalars('train_acc1_step')[-1].wall_time - ea.Scalars('train_acc1_epoch')[-2].wall_time
    acc1 = ea.Scalars('val_acc1_epoch')[-1].value

    pd.Series({
        'test_acc':acc1,
        'per_epoch_time':time
    }).to_csv(args.ffcv_out_path)

if __name__ == '__main__':
    main()