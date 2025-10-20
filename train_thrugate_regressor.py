import pickle as pk
import numpy as np
import torch as tr

def train_regressor(model, optimizer, features, labels, num_train, num_valid, num_updates, checkpoint_path):

    # resample for label imbalance
    loss_fn = tr.nn.BCEWithLogitsLoss(pos_weight = (labels[:num_train]==0).sum() / (labels[:num_train]==1).sum())

    min_valid_loss = tr.inf # for early stopping
    lc = {"train":[], "valid":[]}

    for update in range(num_updates):

        logits = model(features[:num_train])
        targets = labels[:num_train]
        loss = loss_fn(logits, targets)
        lc["train"].append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with tr.no_grad():
            all_logits = model(features)
            logits = all_logits[num_train:num_train+num_valid]
            targets = labels[num_train:num_train+num_valid]
            loss = loss_fn(logits, targets)
            lc["valid"].append(loss.item())

            # early-stop checkpoint
            if loss < min_valid_loss:
                min_valid_loss = loss
                tr.save(model.state_dict(), checkpoint_path)

        if update % max(1, num_updates // 100) == 0:
            print(f"{update} of {num_updates}: train|valid loss = {lc['train'][-1]:.5f} | {lc['valid'][-1]:.5f}")

    return lc, all_logits


if __name__ == "__main__":

    do_train = False
    num_train = 1600 # number of samples for fitting only
    num_valid = 400 # number of samples for testing only, leave some for calibration
    num_updates = 40_000 # number of gradient updates

    npz = np.load("gfd_345k.npz")
    labels = npz["failures"]
    features = npz["init_obs"]
    npz.close()

    print(f"{num_train} training examples ({len(labels)}) total, {features.shape[-1]} features")

    # MLP - same architecture as critic
    # num_hidden = 64 # same as critic
    num_hidden = 16
    regressor = tr.nn.Sequential(
        tr.nn.Linear(features.shape[-1], num_hidden),
        tr.nn.Tanh(),
        tr.nn.Linear(num_hidden, num_hidden),
        tr.nn.Tanh(),
        tr.nn.Linear(num_hidden, 1),
    )

    print(f"MLP: {sum([p.numel()for p in regressor.parameters()])} parameters")

    # opt = tr.optim.SGD(regressor.parameters(), lr=0.01)
    opt = tr.optim.Adam(regressor.parameters(), lr=0.0001)

    x = tr.tensor(features).float()
    y = tr.tensor(labels).reshape(-1, 1).float() # reshape for batch dimension

    if do_train:
        lc, predictions = train_regressor(regressor, opt, x, y, num_train, num_valid, num_updates, "ttr_early.pt")
        with open("ttr_results.pkl","wb") as f:
            pk.dump((lc, predictions, labels), f)

    with open("ttr_results.pkl","rb") as f:
        (lc, predictions, labels) = pk.load(f)

    regressor.load_state_dict(tr.load("ttr_early.pt", weights_only=True))
    with tr.no_grad():
        predictions_early = regressor(x)
    stop = np.argmin(lc["valid"])

    # apply sigmoids and flatten
    predictions = tr.sigmoid(predictions).flatten()
    predictions_early = tr.sigmoid(predictions_early).flatten()

    test_preds = predictions_early[num_train+num_valid:].round().numpy()
    test_labs = labels[num_train+num_valid:]

    print(len(test_labs))

    print(f"Test Accuracy = {(test_preds == test_labs).mean()}")
    print(f"False positive rate = {(test_preds[test_labs == 0] == 1).mean()}")
    print(f"False negative rate = {(test_preds[test_labs == 1] == 0).mean()}")

    print("Confusion:")
    print(f"TP = {((test_preds == 1) & (test_labs == 1)).sum()} ({((test_preds == 1) & (test_labs == 1)).mean()})")
    print(f"FP = {(test_preds > test_labs).sum()} ({(test_preds > test_labs).mean()})")
    print(f"TN = {((test_preds == 0) & (test_labs == 0)).sum()} ({((test_preds == 0) & (test_labs == 0)).mean()})")
    print(f"FN = {(test_preds < test_labs).sum()} ({(test_preds < test_labs).mean()})")

    import matplotlib.pyplot as pt

    pt.figure(figsize=(16,4))

    pt.subplot(1,4,1)
    pt.plot(lc["train"],"b-")
    pt.plot(lc["valid"],"r-")
    pt.yscale("log")
    pt.legend(["Train","Validation"])
    pt.ylabel("BCE loss")
    pt.xlabel("Training iteration")
    pt.title("Learning curves")

    pt.subplot(1,4,2)
    pt.scatter(labels[:num_train] + .01*np.random.randn(num_train), predictions[:num_train], marker=".", color='b')
    pt.scatter(labels[num_train:num_train+num_valid] + .01*np.random.randn(num_valid), predictions[num_train:num_train+num_valid], marker="+", color='r')
    pt.xlim([-0.5, 1.5])
    pt.ylim([-.1, 1.1])
    pt.legend(["Train","Validation"])
    pt.xlabel("Label")
    pt.ylabel("Prediction")
    pt.title("Final iteration performance")

    pt.subplot(1,4,3)
    pt.scatter(labels[:num_train] + .01*np.random.randn(num_train), predictions_early[:num_train], marker=".", color="b")
    pt.scatter(labels[num_train:num_train+num_valid] + .01*np.random.randn(num_valid), predictions_early[num_train:num_train+num_valid], marker="+", color="r")
    pt.xlim([-0.5, 1.5])
    pt.ylim([-.1, 1.1])
    pt.legend(["Train","Validation"])
    pt.xlabel("Label")
    pt.ylabel("Prediction")
    pt.title(f"Early stop ({stop}) performance")

    pt.subplot(1,4,4)
    sort_idx = np.argsort(-predictions_early)
    pt.plot(np.arange(1,len(labels)+1), predictions_early[sort_idx], 'r-', label="predictions")
    pt.plot(np.arange(1,len(labels)+1), labels[sort_idx], 'k.', label="labels")
    pt.legend()
    pt.xlabel("Sorted samples")
    pt.ylabel("Failure indicator")
    pt.title("Predictions vs labels")
    pt.xscale("log")

    pt.tight_layout()
    pt.savefig("ttr_results.eps")
    pt.show()





