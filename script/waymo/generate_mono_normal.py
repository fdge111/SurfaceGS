import os
import argparse
import subprocess
import sys

#TODO: try other normal estimation models (GeoWizard etc.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str)
    parser.add_argument('--dsine_dir', required=True, type=str,
                        help='Path to the DSINE checkout containing test_waymo.py.')

    args = parser.parse_args()
    
    subprocess.run(
        [sys.executable, 'test_waymo.py', '--datadir', os.path.abspath(args.datadir)],
        cwd=os.path.abspath(args.dsine_dir),
        check=True,
    )

        
