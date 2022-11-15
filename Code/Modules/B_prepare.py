from Code.Selector.Selector import Selector


def prepare(args, shape):

    # Load selectors of Loss function and Optimizer
    Loss = Selector("loss").select(args["Prepare"].loss)()
    Optimizer = Selector("optimizer").select(args["Prepare"].optimizer)(
        learning_rate=args["Prepare"].learning_rate,
        **vars(args[args["Prepare"].optimizer])
    )

    if len(shape) == 2:
        new_shape = (shape[0], shape[1], 1)
    else:
        new_shape = shape

    # Load Regression CNN
    Siamese = SiameseNetwork(
        shape=new_shape,
        latent=args["Prepare"].latent,
        kernel_size_x=args["Prepare"].kernel_size_x,
        kernel_size_y=args["Prepare"].kernel_size_y,
        pool_size_x=args["Prepare"].pool_size_x,
        pool_size_y=args["Prepare"].pool_size_y,
        dropout=args["Prepare"].dropout,
        filters=args["Prepare"].filters,
    )

    # Compile and summarise model
    Siamese.compile(optimizer=Optimizer, loss=Loss, metrics=["accuracy"])
    Siamese.summary()

    return Siamese
