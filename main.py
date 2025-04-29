import Dataset

import Texture_Computation as tc



import Debug
import My_Approach




if __name__ == "__main__":

    # Loading dataset
    load_data = False
    if load_data:
        Dataset.TILDA_400()


    # We select randomly image for test and training for binary classification (good - bad)
    split = False
    if split:
        n_bad = 100
        n_good = n_bad * 4
        My_Approach.randmly_selecting_Binary_clasification(n_good, n_bad)
        My_Approach.split_kFold(k=5)

    feat_ext = False
    if feat_ext:
        My_Approach.feature_extraction()


    training = False
    if training:
        My_Approach.training(k=5)


    testing = True
    if testing:
        My_Approach.testing(k=5)