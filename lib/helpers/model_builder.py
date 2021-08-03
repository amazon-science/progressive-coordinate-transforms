from lib.models.patchnet import PatchNet
from lib.models.patchnet_pct import PatchNet_PCT


def build_model(cfg_model, dataset_helper, logger):
    assert cfg_model['name'] in ['patchnet_pct', 'patchnet']
    if cfg_model['name'] == 'patchnet':
        return PatchNet(cfg=cfg_model,
                        num_heading_bin=dataset_helper.num_heading_bin,
                        num_size_cluster=dataset_helper.num_size_cluster,
                        mean_size_arr=dataset_helper.mean_size_arr)
    elif cfg_model['name'] == 'patchnet_pct':
        return PatchNet_PCT(cfg=cfg_model,
                        num_heading_bin=dataset_helper.num_heading_bin,
                        num_size_cluster=dataset_helper.num_size_cluster,
                        mean_size_arr=dataset_helper.mean_size_arr)
    else:
        raise NotImplementedError("%s model is not supported" % cfg_model['name'])
