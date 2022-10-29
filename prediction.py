import numpy as np
import pandas as np

from dataloader import load_data_raw


if __name__ == "__main__":
    df_user, df_problem, df_content = load_data_raw(subset=True)

    pass
