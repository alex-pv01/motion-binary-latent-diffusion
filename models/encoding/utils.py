def load_mbinaryae_from_checkpoint(H, mbinaryae, optim, ema_binaryae):
    try:
        mbinaryae.module = load_model(mbinaryae.module, "mbinaryae", H.load_step, H.load_dir, strict=True, device=binaryae.device).cuda()
    except:
        mbinaryae.module = load_model(mbinaryae.module, "mbinaryae_ema", H.load_step, H.load_dir, strict=True, device=binaryae.device).cuda()
    if H.load_optim:
        optim = load_model(optim, "optim", H.load_step, H.load_dir, strict=True, device=binaryae.device)

    if H.ema:
        try:
            ema_mbinaryae = load_model(ema_mbinaryae, "mbinaryae_ema", H.load_step, H.load_dir, strict=True, device=binaryae.device)
        except FileNotFoundError:
            log("No EMA model found, starting EMA from model load point", level="warning")
            ema_mbinaryae = copy.deepcopy(mbinaryae)

    # return none if no associated saved stats
    try:
        train_stats = load_stats(H, H.load_step)
    except FileNotFoundError:
        log("No stats file found - starting stats from load step.")
        train_stats = None
    return mbinaryae, optim, ema_mbinaryae, train_stats