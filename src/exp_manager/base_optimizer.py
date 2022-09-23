import torch

class BaseOptimizer():
    def __init__(
        self,
        model,optimizer_state,
        PARAM_GROUPS=(),
        freeze_bn=False,
        breed='adam',
        weight_decay=0.0005,
        lr_policy='multistep',
        lr=1e-4, 
        gamma=0.1,
        momentum=0.9,
        betas=(0.9,0.999),
        milestones=[25],
        max_epochs=1000,
        ):    

        # init the optimizer
        if hasattr(model,'_get_param_groups'): # use the model function
            p_groups = model._get_param_groups(lr,wd=weight_decay)
        else:
            allprm = [prm for prm in model.parameters() if prm.requires_grad]
            p_groups = [{ 'params':allprm, 'lr':lr} ]
        
        if breed=='sgd':
            self.optimizer = torch.optim.SGD( p_groups, lr=lr, \
                                momentum=momentum, \
                                weight_decay=weight_decay )

        elif breed=='adagrad':
            self.optimizer = torch.optim.Adagrad( p_groups, lr=lr, \
                                weight_decay=weight_decay )

        elif breed=='adam':
            self.optimizer = torch.optim.Adam( p_groups, lr=lr, \
                                betas=betas, \
                                weight_decay=weight_decay )    
        else:
            raise ValueError("no such solver type %s" % breed)
        print("  -> solver type = %s" % breed)

        if lr_policy=='multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR( \
                        self.optimizer, milestones=milestones, gamma=gamma)
            self.scheduler.max_epochs = max_epochs
        else:
            raise ValueError("no such lr policy %s" % lr_policy)    

        if optimizer_state is not None:
            print("  -> setting loaded optimizer state")        
            self.optimizer.load_state_dict(optimizer_state)

        self.optimizer.zero_grad()
    