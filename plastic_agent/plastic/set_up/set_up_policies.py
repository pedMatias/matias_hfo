import argparse
import os

from plastic_agent.plastic.policy import Policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default=None)
    parser.add_argument('--team_name', type=str, default=None)
    args = parser.parse_args()

    directory = args.models_dir
    team_name = args.team_name
    if os.path.isdir(directory):
        print(f"[Set Up Policies] Dir FOUND: {directory};")
    else:
        print(f"[Set Up Policies] Dir Not Found: {directory};")
        raise NotADirectoryError(directory)

    policies = list()
    if os.path.isdir(os.path.join(directory)):
        policy = Policy.create(team_name=team_name, directory=directory)

    print(f"\n\n!!!!!!!!! Set Up Policies Done  !!!!!!!!!!!!\n\n")
