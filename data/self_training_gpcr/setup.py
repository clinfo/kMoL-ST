import os 

def main():

    iterations = 10
    models = [
            "gpcr_self_training",
            ]
    controls = [
            #"gpcr_random_negative",
            #"gpcr_weighted_loss",
            #"gpcr_random_undersampling"
            ]
    all_models  = models + controls

    for i in range(5):
        for model in all_models:
            os.makedirs("./cv{}/{}/cache".format(i+1, model),  exist_ok=True)
            os.makedirs("./cv{}/{}/config".format(i+1, model), exist_ok=True)
            os.makedirs("./cv{}/{}/data".format(i+1, model), exist_ok=True)
            os.makedirs("./cv{}/{}/result".format(i+1, model), exist_ok=True)
            os.makedirs("./cv{}/{}/split".format(i+1, model), exist_ok=True)
            if model in models:
                for j in range(iterations):
                    os.makedirs("./cv{}/{}/data/iter{}".format(i+1, model, j), exist_ok=True)
if __name__ == "__main__":
    main()
