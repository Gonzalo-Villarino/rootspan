"""Convert the linked public PAT checkpoint to a tensor-only state dictionary.

Only allow explicit Torch/torchvision/SMP model classes, never arbitrary
checkpoint globals. No training or image inference. Source is unchanged.
"""
import hashlib,importlib,json
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
folder=Path(__file__).resolve().parent
source=folder/'pat_original.pth'
names=torch.serialization.get_unsafe_globals_in_checkpoint(source)
if not all(n.startswith(('torch.','torchvision.','segmentation_models_pytorch.')) for n in names):
    raise ValueError('Unexpected checkpoint globals; inspect before loading.')
classes=[getattr(importlib.import_module(n.rsplit('.',1)[0]),n.rsplit('.',1)[1]) for n in names]
with torch.serialization.safe_globals(classes):
    original=torch.load(source,map_location='cpu',weights_only=True)
if isinstance(original,torch.nn.DataParallel):original=original.module
state=original.state_dict()
model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation='sigmoid')
model.load_state_dict(state,strict=True)
destination=folder/'pat_state_dict.pt'
if destination.exists():
    saved=torch.load(destination,map_location='cpu',weights_only=True)
    assert saved.keys()==state.keys() and all(torch.equal(saved[k],v) for k,v in state.items())
else:torch.save(state,destination)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/','repository':'https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm','released_model_url':'https://drive.google.com/file/d/1E5f0Heav8L_k9WxvvEQoBxC_J_zLH-J4/view','released_filename':'best_model_unet_plusplus_resnet34_periderm4classes_Lin_0301_1024_6.pth','source_bytes':source.stat().st_size,'source_sha256':sha(source),'tensor_state_sha256':sha(destination),'original_saved_training_flag':original.training,'strict_state_match':True,'allowed_globals':names,'output_channels':['background','periderm','endodermis','lateral_root'],'preprocessing':'RGB/255, mean[.485,.456,.406], std[.229,.224,.225], matching official ResNet34 ImageNet preprocessing.','source_code_license':'See LICENSE; scripts copied from public repository for inspection, not run. Model redistribution license not separately established.','training_membership_of_MT25_rep3':'Unknown; cannot claim independent evaluation.'}
(folder/'download_and_conversion_record.json').write_text(json.dumps(record,indent=2)+'\n')
print('PAT state dictionary verified; provenance record saved.')
