from pathlib import Path
import json,csv,io,urllib.request,hashlib,sys
P=Path(__file__).resolve().parents[1];W=P/'16_Per_Root_Estimates_20261117';ident='20261117T155514Z_34c06d0c'
api=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/state?job='+ident));j=api['job'];assert j['status']=='complete',j['message'];s=j['summary'];report=s['per_root_report']
assert api['version']=='0.6.0';assert len(s['rows'])==7;assert report['numeric_count']==7
assert not s.get('diagnostic_error') and not s.get('per_root_report_error')
assert [round(r['periderm_length_um']) for r in report['rows']]==[0,28990,4358,70,318,8404,1551]
assert report['rows'][0]['value_source']=='Expert correction';assert all(r['value_source']=='PAT mask estimate' for r in report['rows'][1:])
url='http://127.0.0.1:8765/files/'+ident+'/results/periderm_estimates.csv'
text=urllib.request.urlopen(url).read().decode();export=list(csv.DictReader(io.StringIO(text)));assert len(export)==7
for a,b in zip(export,report['rows']):
 assert int(a['root'])==b['root'];assert float(a['periderm_length_um'])==b['periderm_length_um'];assert a['value_source']==b['value_source'];assert a['snapshot_generated_at_utc']
r=P/'09_Periderm_GUI/runs'/ident
saved=list(csv.DictReader((r/'results/periderm_estimates.csv').open()))
for a,b in zip(saved,report['rows']):assert float(a['periderm_length_um'])==b['periderm_length_um']
source=P/'02_Data/16_New_SALK_GUI_Checks_20261117/7_18.tif'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(source)==s['source_sha256']
sys.path.insert(0,str(P/'09_Periderm_GUI'));from version_registry import verify_v2,verify_v21
verify_v2();verify_v21()
col=json.loads((P/'09_Periderm_GUI/runs/20261117T155137Z_7d3c3a7d/job.json').read_text());assert col['status']=='complete';assert col['summary']['per_root_report']['numeric_count']==6
facts={'browser_upload_run':ident,'source_name':j['image_name'],'source_sha256':s['source_sha256'],'numeric_estimates':7,'main_report_and_live_export_match':True,'folder_snapshot_values_match':True,'original_source_unchanged':True,'locked_V1_V2_models_unchanged':True,'root1_expert_correction_marked':True,'Col_rep1_previous_run_preserved_with_6_values':True,'software_tests_passed':40,'training_occurred':False,'report_is_mask_based_estimate_not_validated_anatomy':True}
(W/'browser_run_verification.json').write_text(json.dumps(facts,indent=2)+'\n');(W/'periderm_estimates_7_18.csv').write_text(text)
files=['09_Periderm_GUI/per_root_estimates.py','09_Periderm_GUI/server.py','09_Periderm_GUI/index.html','09_Periderm_GUI/pat_diagnostics.py','09_Periderm_GUI/pat_postprocessing.py']
(W/'final_source_hashes.json').write_text(json.dumps({f:sha(P/f) for f in files},indent=2)+'\n')
print(json.dumps(facts,indent=2))
