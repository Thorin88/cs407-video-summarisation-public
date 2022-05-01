from anchor_based.dsnet import DSNet
from anchor_free.dsnet_af import DSNetAF
from feat_dsnet.feat_dsnet_af import FeatureDSNetAF

def get_anchor_based(base_model, num_feature, num_hidden, anchor_scales,
                     num_head, **kwargs):
    return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head)

def get_anchor_free(base_model, num_feature, num_hidden, num_head, **kwargs):
    return DSNetAF(base_model, num_feature, num_hidden, num_head)

def get_feat_anchor_free(base_model, num_feature, num_hidden, num_head, **kwargs):
    # print("Base Model == " + str(base_model))
    return FeatureDSNetAF(base_model, num_feature, num_hidden, num_head)

def get_model(model_type, **kwargs):
    if model_type == 'anchor-based':
        return get_anchor_based(**kwargs)
    elif model_type == 'anchor-free':
        return get_anchor_free(**kwargs)
    elif model_type == 'feat-anchor-free':
        return get_feat_anchor_free(**kwargs)
    # Just allows you to use basic in the command args
    elif model_type == 'basic':
        return None
    else:
        raise ValueError(f'Invalid model type {model_type}')
