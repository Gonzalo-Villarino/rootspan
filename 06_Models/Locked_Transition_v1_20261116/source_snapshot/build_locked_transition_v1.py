"""Build a self-contained copy for verification; seal only after successful replay."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'08_Model_Refinement_20261116'
BUNDLE=ROOT/'06_Models/Locked_Transition_v1_20261116'
VERSION='Transition_v1_20261116'

def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def function_source(path,name):
    text=path.read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,node)
def manifest():
    files={str(p.relative_to(BUNDLE)):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name not in ['MANIFEST.json','MANIFEST.sha256']}
    save(BUNDLE/'MANIFEST.json',{'version':VERSION,'files':files,'purpose':'Frozen reproducible transition checkpoint; biological review accepted on listed examples, not validated unattended length measurement.'})
    (BUNDLE/'MANIFEST.sha256').write_text(sha(BUNDLE/'MANIFEST.json')+'\n')

def build():
    BUNDLE.mkdir(exist_ok=False)
    for folder in ['weights','fixtures/profiles','evidence','source_snapshot']:(BUNDLE/folder).mkdir(parents=True,exist_ok=True)
    prior=ROOT/'06_Models/20261116_Two_Appearance_Candidate'
    old=json.loads((prior/'candidate_manifest.json').read_text())
    for key,name in [('head','presence_head.pt'),('encoder','pat_state_dict.pt')]:
        source=Path(old[key]);assert sha(source)==old[key+'_sha256']
        target=BUNDLE/'weights'/name;shutil.copy2(source,target);assert sha(target)==sha(source)
    for source,name in [('locked_checkpoint_runner.py','run_checkpoint.py'),('locked_checkpoint_verify.py','verify_checkpoint.py')]:
        shutil.copy2(SOURCE/source,BUNDLE/name)
    # Extract the exact approved functions, omitting research scripts' external paths.
    core='"""Exact approved image standardization and decoder functions, bundled locally."""\nimport numpy as np\nfrom scipy.ndimage import gaussian_filter1d\nWIDTH, HEIGHT = 320, 256\nWINDOW_WIDTH = 320.0\nREQUIRED_SPACED_E_FIELDS = 3\n\n'
    for file,names in [('appearance_experiment.py',['standardize']),('test_both_appearances.py',['boundary_rule']),('transition_continuity_rule.py',['spaced_indices','continuity_rule','run_checks'])]:
        for name in names:core+=function_source(SOURCE/file,name)+'\n\n'
        shutil.copy2(SOURCE/file,BUNDLE/'source_snapshot'/file)
    (BUNDLE/'frozen_core.py').write_text(core)
    for name in ['check_frozen_212_root1.py','check_frozen_a04_root3.py','extend_root3_reference_route.py','check_transition_continuity.py',Path(__file__).name]:
        shutil.copy2(SOURCE/name,BUNDLE/'source_snapshot'/name)
    settings={'version':VERSION,'input':'RGB image crop plus main-root path in crop coordinates; path must be checked separately.',
        'window_width':320,'window_height':256,'stride_pixels':64,'tangent_half_distance':80,
        'sampling':'Bilinear native-pixel sampling, no geometric resize; constant-zero outside image.',
        'normalization':'Exact frozen_core.standardize: row-energy centre, sigma12, p99.5green to160, then ImageNet RGB mean/std.',
        'encoder':'Frozen PAT ResNet34, depth5; mean-pooled L2-normalized feature stages1–5, concatenated then L2-normalized.',
        'head':'Frozen linear two-output sigmoid presence head','presence_threshold':0.5,
        'mixed_requirement':'Both predicted appearances >=0.5; long rectangular Endoderm alone is not mixed.',
        'decoder':'continuity_guard_v1; original3-window candidate/support rules plus3E-only centres spaced320path pixels as contradiction before supporting P run.',
        'batch_size':8,'cpu_threads':4,'device':'MPS if available, otherwise CPU',
        'boundary_definition':'T = left edge of mixed-cell zone on main root; mixed zone included with Periderm for future regional measurements.',
        'no_training':True,'no_automatic_full_slide_length_claim':True}
    save(BUNDLE/'settings.json',settings)
    packages={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata['Name']}
    save(BUNDLE/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'architecture':platform.machine(),'installed_packages':dict(sorted(packages.items(),key=lambda x:x[0].lower()))})
    (BUNDLE/'requirements.freeze.txt').write_text('\n'.join(f'{n}=={v}' for n,v in sorted(packages.items(),key=lambda x:x[0].lower()))+'\n')
    parent=ROOT/'04_Results/69_Transition_Continuity_Check'
    summary=json.loads((parent/'comparison_results.json').read_text())
    approved=next(r for r in summary['results'] if r['case']=='2_12_root1')
    acceptance={'recorded_utc':datetime.now(timezone.utc).isoformat(),'expert':'Gonzalo Villarino',
        'statement':'solid blue is excellnt, can we lock this model or what?',
        'accepted':True,'accepted_case':'2_12 root1','accepted_T_source_xy':approved['new_T_source_xy'],
        'accepted_T_path_pixels':approved['new_T_path_pixels'],'accepted_decoder':'continuity_guard_v1',
        'scope':'Post-prediction biological acceptance and authorisation to freeze this version. Not independent numerical accuracy or all-slide validation.',
        'preserved_earlier_accepted_cases':['COL0 root3','A04 root3'],'other_case':'20_29 root4 still returns no supported T.',
        'evaluation_limit':'2_12 feedback informed this decoder; it is development data for this version.',
        'default_pixel_segmentation_unet_unchanged':True}
    acceptance_file=parent/'BIOLOGICAL_REVIEW_ACCEPTED_20261116.json'
    if acceptance_file.exists():raise FileExistsError(acceptance_file)
    save(acceptance_file,acceptance);save(BUNDLE/'ACCEPTANCE.json',acceptance)
    for source,name in [(parent/'212_CONTINUITY_CANDIDATE.png','approved_result.png'),
        (parent/'comparison_results.json','comparison_results.json'),(parent/'experiment_plan.json','experiment_plan.json'),
        (parent/'verification.json','development_verification.json'),(parent/'biological_definition_reconfirmed.json','biological_definition.json'),
        (ROOT/'04_Results/65_Mixed_Appearance_Evidence_Check/BIOLOGICAL_REVIEW_ACCEPTED_20261116.json','COL0_acceptance.json'),
        (ROOT/'04_Results/67_Fixed_A04_Root3_Check/BIOLOGICAL_REVIEW_ACCEPTED_20261116.json','A04_acceptance.json')]:
        shutil.copy2(source,BUNDLE/'evidence'/name)
    pred_parent=ROOT/'04_Results/68_Fixed_212_Root1_Check'
    for source,name in [(pred_parent/'image.png','212_image.png'),(pred_parent/'image_only_route.json','212_route.json'),
                        (pred_parent/'prediction_before_comparison.json','212_original_prediction.json'),
                        (parent/'2_12_root1_decoder_result.json','212_approved_decoder.json')]:
        shutil.copy2(source,BUNDLE/'fixtures'/name)
    plan=json.loads((parent/'experiment_plan.json').read_text())
    for entry in plan['inputs']:
        expected=next(r for r in summary['results'] if r['case']==entry['case'])
        save(BUNDLE/'fixtures/profiles'/f'{entry["case"]}.json',{'case':entry['case'],'prediction':json.loads(Path(entry['prediction']).read_text()),'expected_T_path_pixels':expected['new_T_path_pixels']})
    (BUNDLE/'READ_ME_FIRST.txt').write_text('LOCKED TRANSITION CHECKPOINT v1 — 16 November 2026\n\nWhat is saved\nThe exact learned image-recognition weights, image preparation settings, and continuity rule that produced the solid blue T you approved on2_12root1. The earlier accepted COL0/A04positions were retained. Your approval, example picture, source code, software versions and reference checks are included. No training occurs when this checkpoint runs.\n\nWhat locked means\nThe files are made read-only after checking the saved copy. Fingerprints detect accidental edits. Future improvements must go into a new version folder. This is preservation of a working research checkpoint, not proof it works on every root.\n\nWhat it currently does\nGiven a microscopy crop and a checked main-root path, it predicts Endoderm-like and Periderm-like appearances, then selects a supported T or returns no T. Both appearances are needed for a mixed call. It does not yet perform validated unattended measurements from every whole slide.\n\nHow to check the copy\nUse the project ml-env Python. Run verify_checkpoint.py --require-read-only. To replay the approved example, run run_checkpoint.py --self-check --output NEW_FOLDER_OUTSIDE_THIS_CHECKPOINT. The latter reproduces image scores and T, then replays four saved profiles.\n\nUse on a later root\nrun_checkpoint.py --image IMAGE_CROP.png --route MAIN_ROOT_ROUTE.json --output NEW_RESULT_FOLDER\nThe route uses path_xy_crop_pixels and distance_along_path_pixels, with optional source_crop_xyxy for full-slide coordinates. Image and path must match. This program does not find all roots in a whole slide; path preparation is a separate checked step. Original geometry settings/code are archived in source_snapshot.\n\nSoftware\nThe existing00_Software/ml-env environment runs this copy. environment.json and requirements.freeze.txt record versions; no environment or extra models are downloaded by this bundle. Other software/hardware versions may cause numerical differences; rerun self-check.\n\nLearning note\nLocking saves what we have. Testing other images tells us how dependable it is.2_12 helped improve this rule, so it cannot be its independent final test. No micrometre-length accuracy is claimed here. The earlier segmentation U-Net file is retained separately.\n')
    save(BUNDLE/'provenance.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'original_head':old['head'],'original_encoder':old['encoder'],
        'head_sha256':old['head_sha256'],'encoder_sha256':old['encoder_sha256'],
        'source_experiment':str(parent),'training_examples_source':old['labels'],
        'model_lineage':'Published PAT encoder frozen; linear head taught on five reviewed fields from two roots on COL0. New continuity decoder developed with2_12feedback.',
        'working_U_Net_sha256':sha(ROOT/'06_Models/working_model.pt'),
        'source_images_unmodified':True,'requires_installed_python_libraries':True})
    manifest();print(BUNDLE)

if __name__=='__main__':build()
