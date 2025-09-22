import pickle as pk
import numpy as np
import torch as tr
from planar_features import FeatureExtractor

def train_regressor(model, optimizer, features, labels, num_train, num_valid, num_updates, checkpoint_path):

    # # resample for label imbalance?
    # max_lab = labels[:num_train].max()    

    min_valid_loss = tr.inf # for early stopping
    lc = {"train":[], "valid":[]}

    for update in range(num_updates):
        errors = model(features[:num_train]) - labels[:num_train]
        loss = tr.mean(errors**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lc["train"].append(loss.item())
        
        with tr.no_grad():
            predictions = model(features)
            loss = tr.mean((predictions[num_train:num_train+num_valid] - labels[num_train:num_train+num_valid])**2)
            lc["valid"].append(loss.item())

            # early-stop checkpoint
            if loss < min_valid_loss:
                min_valid_loss = loss
                tr.save(model.state_dict(), checkpoint_path)

        if update % max(1, num_updates // 100) == 0:
            print(f"{update} of {num_updates}: train|valid loss = {lc['train'][-1]:.5f} | {lc['valid'][-1]:.5f} <= {errors.abs().max()}")

    return lc, predictions


if __name__ == "__main__":

    do_train = False
    num_train = 8000 # number of samples for fitting only
    num_valid = 1000 # number of samples for testing only, leave some for calibration
    num_updates = 20_000 # number of gradient updates

    with open("failrate_data_10k_512r.pkl", "rb") as f:
        (env, init_states, fail_rates) = pk.load(f)

    get_features = FeatureExtractor(env) # linear
    # get_features = FeatureExtractor(env, kernel="quadratic") # 
    # get_features = FeatureExtractor(env, kernel="cubic")

    observations = env.get_observation(init_states)
    features, _ = get_features(observations)
    labels = fail_rates
    print(f"{num_train} training examples ({len(labels)}) total, {features.shape[-1]} features")

    # MLP
    regressor = tr.nn.Sequential(
        tr.nn.Linear(features.shape[1], 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 1),
        tr.nn.Sigmoid(),
        tr.nn.Flatten(0), # effectively squeezes
    )
    print(f"MLP: {sum([p.numel()for p in regressor.parameters()])} parameters")

    # opt = tr.optim.SGD(regressor.parameters(), lr=0.01)
    opt = tr.optim.Adam(regressor.parameters(), lr=0.0001)

    x = tr.tensor(features).float()
    y = tr.tensor(labels).float()

    if do_train:
        lc, predictions = train_regressor(regressor, opt, x, y, num_train, num_valid, num_updates, "tfr_early.pt")
        with open("tfr_results.pkl","wb") as f:
            pk.dump((lc, predictions, labels), f)

    with open("tfr_results.pkl","rb") as f:
        (lc, predictions, labels) = pk.load(f)

    regressor.load_state_dict(tr.load("tfr_early.pt", weights_only=True))
    with tr.no_grad():
        predictions_early = regressor(x)
    stop = np.argmin(lc["valid"])

    errors = sorted(tr.abs(predictions_early - labels)[num_train+num_valid:])
    interval = errors[int(np.ceil(0.95*(len(errors)+1)))]

    import matplotlib.pyplot as pt

    pt.figure(figsize=(16,4))

    pt.subplot(1,4,1)
    pt.plot(lc["train"],"b-")
    pt.plot(lc["valid"],"r-")
    pt.yscale("log")
    pt.legend(["Train","Validation"])
    pt.ylabel("MSE")
    pt.xlabel("Training iteration")
    pt.title("Learning curves")

    pt.subplot(1,4,2)
    pt.scatter(labels[:num_train], predictions[:num_train], marker=".", color='b')
    pt.scatter(labels[num_train:], predictions[num_train:], marker="+", color='r')
    pt.xlim([min(predictions.min(),labels.min()), max(predictions.max(),labels.max())])
    pt.ylim([min(predictions.min(),labels.min()), max(predictions.max(),labels.max())])
    pt.legend(["Train","Validation"])
    pt.xlabel("Label")
    pt.ylabel("Prediction")
    pt.title("Final iteration performance")

    pt.subplot(1,4,3)
    pt.scatter(labels[:num_train], predictions_early[:num_train], marker=".", color="b")
    pt.scatter(labels[num_train:], predictions_early[num_train:], marker="+", color="r")
    pt.xlim([min(predictions_early.min(),labels.min()), max(predictions_early.max(),labels.max())])
    pt.ylim([min(predictions_early.min(),labels.min()), max(predictions_early.max(),labels.max())])
    pt.legend(["Train","Validation"])
    pt.xlabel("Label")
    pt.ylabel("Prediction")
    pt.title(f"Early stop ({stop}) performance")

    pt.subplot(1,4,4)
    sort_idx = np.argsort(-labels)
    pt.plot(np.arange(1,len(labels)+1), predictions_early[sort_idx], 'r.', label="predictions")
    pt.plot(np.arange(1,len(labels)+1), labels[sort_idx], 'k-', label="labels")
    # pt.plot([0,len(errors)],[interval,interval], 'k--', label="95%% confidence interval")
    # pt.ylim([min(predictions.min(),labels.min()), max(predictions.max(),labels.max())])
    pt.legend()
    pt.xlabel("Sorted samples")
    pt.ylabel("Failure rate")
    pt.xscale("log")
    # pt.title("Conformal calibration")

    pt.tight_layout()
    pt.savefig("tfr_results.png")
    pt.show()

