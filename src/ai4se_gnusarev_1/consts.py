from ai4se_gnusarev_1.internal.models import random_forest_train, random_forest_save


MODELS = {
    "random_forest": {
        "train": random_forest_train,
        "save": random_forest_save,
    }
}
