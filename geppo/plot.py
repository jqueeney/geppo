"""Entry point for visualizing results."""
import numpy as np

from geppo.common.plot_utils import create_plotparser, open_and_aggregate
from geppo.common.plot_utils import plot_compare

def main():
    """Parses inputs, creates and saves plot."""
    parser = create_plotparser()
    args = parser.parse_args()

    x = np.arange(0,args.timesteps+1,args.interval)

    ppo_data = open_and_aggregate(
        args.import_path,args.ppo_file,x,args.window,args.metric)
    geppo_data = open_and_aggregate(
        args.import_path,args.geppo_file,x,args.window,args.metric)

    plot_compare(ppo_data,geppo_data,
        x,args.se_val,args.save_path,args.save_name)


if __name__=='__main__':
    main()