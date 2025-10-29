from tensorflow.keras import callbacks

def train_model(model, train_gen, val_gen, y_train_std, y_val_std,
                batch_size=16, epochs=50, save_path="./best_model/best_model.keras"):
    """
    Train the model with generators and callbacks.

    Returns:
        history: training history
        test_metrics: dict with standardized MAE/MSE and optionally raw MAE/MSE
    """

    steps_per_epoch = len(y_train_std) // batch_size
    validation_steps = len(y_val_std) // batch_size

    print(f"Training samples: {len(y_train_std)} | Validation samples: {len(y_val_std)}")
    print(f"Steps per epoch: {steps_per_epoch} | Validation steps: {validation_steps}")
    print(f"Batch size: {batch_size}")

    # --- Callbacks ---
    checkpoint_cb = callbacks.ModelCheckpoint(
        save_path,
        save_best_only=True,
        monitor="val_mae",
        mode="min"
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    # --- Train the model ---
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
        verbose=1
    )

    return history
