
def run_repetition(mu):

    tau = calibrate()
    status = deploy()
    
    return status, tau

if __name__ == "__main__":

    mu, confusion = train_mu()
    for rep in range(num_repetitions):
        status, tau = run_repetition(mu)


