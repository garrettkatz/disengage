import pickle as pk
import numpy as np
import torch as tr
from planar_features import FeatureExtractor

def sigmoid(x): return 1 / (1 + np.exp(-x))
def inv_sigmoid(x): return np.log(x) - np.log(1-x)

if __name__ == "__main__":

    do_train = True
    num_train = 3000 # number of samples for fitting only
    num_updates = 1000 # number of gradient updates

    with open(f"failrate_data.pkl", "rb") as f:
        (env, init_states, fail_rates) = pk.load(f)

    get_features = FeatureExtractor(env) # linear
    # get_features = FeatureExtractor(env, kernel="quadratic") # 
    # get_features = FeatureExtractor(env, kernel="cubic")

    observations = env.get_observation(init_states)
    features, _ = get_features(observations)
    labels = fail_rates
    print(f"{num_train} examples, {features.shape[-1]} features")

    # # linear regression
    # # regressor = np.linalg.lstsq(features, labels, rcond=None)[0]
    # # predictions = features @ regressor

    # # linear regression with sigmoid activation
    # targets = np.where(labels == 0, 1e-3, labels)
    # regressor = np.linalg.lstsq(features[:num_train], inv_sigmoid(targets[:num_train]), rcond=None)[0]
    # predictions = sigmoid(features @ regressor)

    # MLP
    x = tr.tensor(features).float()
    y = tr.tensor(labels).float()
    regressor = tr.nn.Sequential(
        tr.nn.Linear(features.shape[1], 128),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(128, 32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32, 1),
        tr.nn.Sigmoid()
    )
    opt = tr.optim.SGD(regressor.parameters(), lr=0.01)

    if do_train:

        lc = {"train":[], "test":[]}
        for update in range(num_updates):
            errors = regressor(x[:num_train]) - y[:num_train,None]
            loss = tr.mean(errors**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lc["train"].append(loss.item())
            
            with tr.no_grad():
                loss = tr.mean((regressor(x[num_train:]) - y[num_train:,None])**2)
                lc["test"].append(loss.item())
    
            if update % max(1, num_updates // 100) == 0:
                print(f"{update} of {num_updates}: train|test loss = {lc['train'][-1]:.5f} | {lc['test'][-1]:.5f} <= {errors.abs().max()}")
    
        with tr.no_grad():
            predictions = regressor(x).detach().squeeze().numpy()
        # errors = np.fabs(predictions - labels)
    
        with open("tfr_results.pkl","wb") as f:
            pk.dump((lc, predictions, labels),f)

    with open("tfr_results.pkl","rb") as f:
        (lc, predictions, labels)=pk.load(f)

    import matplotlib.pyplot as pt
    pt.subplot(1,2,1)
    pt.plot(lc["train"],"b-")
    pt.plot(lc["test"],"r-")
    pt.yscale("log")
    pt.legend(["Train","Test"])
    pt.subplot(1,2,2)
    pt.scatter(labels[:num_train], predictions[:num_train], marker=".")
    pt.scatter(labels[num_train:], predictions[num_train:], marker="+")
    pt.axis("equal")
    pt.legend(["Train","Test"])
    pt.show()

