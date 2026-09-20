import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from runtime_config import project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Label and combine explicit member/nonmember score files')
    parser.add_argument('--member-file', required=True)
    parser.add_argument('--nonmember-file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    frames = []
    for filename, label in [(args.member_file, 1), (args.nonmember_file, 0)]:
        frame = pd.read_csv(project_path(filename))
        if frame.empty or 'diff_score' not in frame or not np.isfinite(frame.diff_score).all():
            raise ValueError(f'Empty or invalid score file: {filename}')
        frame['label'] = label
        frames.append(frame)
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f'Saved {output}: member={len(frames[0])}, nonmember={len(frames[1])}')


if __name__ == '__main__':
    main()
