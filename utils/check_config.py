def check_config(__C):
    assert __C.training['warmup_steps'] < __C.training['max_steps'], "warmup-steps should be smaller than max-steps"