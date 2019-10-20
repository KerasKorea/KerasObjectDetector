import sys, os
sys.path.insert(0, os.path.abspath('./'))
from keras.optimizers import SGD
import yolk
from yolk.backend.ssd_backend import setup_training

def main():
    # Make a Model
    model = yolk.detector.load_training_model(num_classes=20)
    loss = yolk.detector.get_loss()
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss=loss)

    # Load Generators
    dataset_path = "./datasets"
    train_generator, callbacks, val_generator, val_steps = setup_training(model, dataset_path)

    # Train a Model
    initial_epoch   = 0
    final_epoch     = 120
    steps_per_epoch = 1000
    model.fit_generator(generator=train_generator,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=val_steps,
                        steps_per_epoch=steps_per_epoch,
                        epochs=final_epoch,
                        initial_epoch=initial_epoch)

    

if __name__ == '__main__':
    main()
