import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=("exploration", "testing"), required=True)
    args = parser.parse_args()
    if args.mode == "exploration":
        parser.add_argument('--num_opponents', type=int, default=1)
        args = parser.parse_args()
        print(args)
    elif args.mode == "testing":
        parser.add_argument('--num_teammates', type=int, required=True)
        args = parser.parse_args()
        print(args)