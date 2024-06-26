EPOCHS = 20

history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.samples / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=validation_data_gen,
    validation_steps=int(np.ceil(validation_data_gen.samples / float(BATCH_SIZE)))
)
