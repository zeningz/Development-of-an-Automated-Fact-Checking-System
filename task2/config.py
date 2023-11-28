def cls_setting(args):
    # default setting
    args.batch_size = getattr(args, 'batch_size', 16)
    args.epoch = getattr(args, 'epoch', 50)
    args.report_freq = getattr(args, "report_freq", 10)
    args.accumulate_step = getattr(args, "accumulate_step", 2)
    args.model_type = getattr(args, "model_type", "roberta-large")
    args.warmup_steps = getattr(args, "warmup_steps", 150)
    args.grad_norm = getattr(args, "grad_norm", 1)
    args.seed = getattr(args, "seed", 5)
    args.max_lr = getattr(args, "max_lr", 3e-5)
    args.max_length = getattr(args, "max_length", 512)
    args.eval_interval = getattr(args, "eval_interval", 20)
