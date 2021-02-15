import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!
    use_improved_weight_init = True
    """
    model_improved_weights = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_weights = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weights, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weights, val_history_improved_weights = trainer_improved_weights.train(
        num_epochs)
    
    model_improved_weights_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_weights_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weights_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weights_sigmoid, val_history_improved_weights_sigmoid = trainer_improved_weights_sigmoid.train(
        num_epochs)
    """
    learning_rate = .02
    use_improved_sigmoid = True
    use_momentum = True
    model_improved_weights_sigmoid_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_weights_sigmoid_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weights_sigmoid_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weights_sigmoid_momentum, val_history_improved_weights_sigmoid_momentum = trainer_improved_weights_sigmoid_momentum.train(
        num_epochs)
        


    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved_weights_sigmoid_momentum["loss"], "Task 3 Model - Improved weights and sigmoid", npoints_to_average=10)
    
    
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_improved_weights_sigmoid_momentum["accuracy"], "Task 3 Model - Improved weights and sigmoid")
    
    
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
