import sys, os
sys.path.insert(0, os.path.abspath('./'))
from keras.optimizers import SGD
import yolk
from yolk.parser import parse_args



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    args = parse_args(args)

    # Make a Model
    model = yolk.detector.load_training_model(20, args)
    loss = yolk.detector.get_losses(args)
    sgd = SGD(learning_rate=0.001, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss=loss)

    # Load Generators
    train_generator = yolk.detector.get_data_generator(args)

    # Train a Model
    initial_epoch   = 0
    final_epoch     = 120
    steps_per_epoch = 1000
    model.fit_generator(generator=train_generator,
                        # callbacks=callbacks,
                        # validation_data=val_generator,
                        # validation_steps=val_steps,
                        steps_per_epoch=steps_per_epoch,
                        epochs=final_epoch,
                        initial_epoch=initial_epoch)

    ###########
    ###########
    ###########
    """
    predict_generator = val_dataset.generate(batch_size=1,
                                            shuffle=True,
                                            transformations=[convert_to_3_channels,
                                                            resize],
                                            label_encoder=None,
                                            returns={'processed_images',
                                                    'filenames',
                                                    'inverse_transform',
                                                    'original_images',
                                                    'original_labels'},
                                            keep_images_without_gt=False)
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)


    y_pred = model.predict(batch_images)

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)
    y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded_inv[i])
    """

    

if __name__ == '__main__':
    main()
