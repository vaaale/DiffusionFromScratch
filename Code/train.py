from ddpm import train
import yaml
import os


def main(args):

    config_path = os.path.join(args.experiment, args.run)
    if not os.path.isdir(args.experiment):
        print(2, f"Experiment {args.experiment} does not exist")
        exit(-1)
    elif not os.path.isfile(config_path):
        print(2, f"Run {args.run} does not exist")
        exit(-1)

    with open(config_path, "r") as inp:
        config = yaml.safe_load(inp)

    train(config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="Config", help='Experiment folder')
    parser.add_argument('--run', type=str, default=None, required=True, help='Name of run yml file')
    args = parser.parse_args()

    main(args)





