import torch.optim as Optim

def get_optim(__C, params):
    params = params
    std_optim = getattr(Optim, __C.optim['name'])
    eval_str = 'params'
    for key in __C.optim:
        if key == 'name':
            continue
        eval_str += ' ,' + key + '=' + str(__C.optim[key])
    optim = eval('std_optim'+'('+eval_str+')')
    return optim