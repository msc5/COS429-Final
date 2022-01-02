
def test(
        name,
        model,
        dataloader,
        callback,
        loss_fn,
        device,
):

    # Load Model from state_dict
    dir = os.path.join('saves', model.__name__, name)
    if not os.path.exists(dir):
        print(f'{name} does not exist')
    model_path = os.path.join(dir, 'model')
    log_path = os.path.join(dir, 'log_test')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    batches = len(dataloader)
    print(batches)

    # Initialize Logger
    logger = Logger(1, batches, log_path)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.zero_grad()

    # Print header
    print(logger.header())
    tic = time.perf_counter()

    with torch.no_grad():
        for j, test_ds in enumerate(dataloader):
            results = callback(
                model,
                test_ds,
                None,
                loss_fn, device,
                train=False
            )
            toc = time.perf_counter()
            log = logger.log((0, 0, *results), toc - tic)

    print(log)
