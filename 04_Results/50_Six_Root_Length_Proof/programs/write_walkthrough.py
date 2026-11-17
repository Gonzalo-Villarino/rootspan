from pathlib import Path
import json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'04_Results/50_Six_Root_Length_Proof';r=json.loads((F/'proof_record.json').read_text());d=Document();sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.top_margin=sec.bottom_margin=Inches(.7);sec.left_margin=sec.right_margin=Inches(.75)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
 st=d.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
d.styles['Normal'].font.size=Pt(11.5);d.styles['Normal'].paragraph_format.space_after=Pt(7);d.styles['Title'].font.size=Pt(26);d.styles['Heading 1'].font.size=Pt(16)
for style in d.styles:
 for border in list(style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
p=d.add_paragraph('Periderm length proof of concept','Title');d.add_paragraph('Learning record for Gonzalo Villarino   |   15 November 2026')
d.add_paragraph('We now have a way to measure the curved length of Periderm on all six roots in MT25_rep3. The example below uses your saved outlines. The automatic model results are saved separately because the model still makes mistakes when locating the transition.')
d.add_picture(str(F/'OPEN_ME_SIX_ROOT_LENGTHS.png'),width=Inches(7))
d.add_paragraph('Figure 1. Cyan is the measured path. Magenta marks the start of your Periderm outline; white marks its end. The roots are numbered from top to bottom. These lines use your annotations and are not evidence that the model found the same boundaries.',style='Normal')
d.add_paragraph('What this demonstrates','Heading 1');d.add_paragraph('The computer can follow a curved root, add the distances along its centre, and save a length for each root. We still need to check that the start and end correspond to the biological region we want to measure.')
d.add_page_break();d.add_paragraph('Compare the six roots','Heading 1');d.add_paragraph('All values below are original image pixels. “From your outlines” means a centreline was calculated inside each saved Periderm polygon. “Model only” means the image was analysed without reading your annotations, followed by a separate rule that keeps Periderm connected to the basal end.')
t=d.add_table(rows=1,cols=4);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
headers=['Root','From your outlines\nPixels','Model only\nPixels','Manual measurement\nPixels']
for col,width in zip(t.columns,[.65,2.05,1.8,2.5]):col.width=Inches(width)
for c,txt in zip(t.rows[0].cells,headers):c.text=txt
for rr in r['roots']:
 cells=t.add_row().cells
 for c,txt in zip(cells,[str(rr['root_id']),f"{rr['annotation_assisted_length_px']:,.0f}",f"{rr['automatic_length_px']:,.0f}",'Not entered']):c.text=txt
for row_index,row in enumerate(t.rows):
 for j,c in enumerate(row.cells):
  c.width=Inches([.65,2.05,1.8,2.5][j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
  tcpr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
  for edge in ['top','left','bottom','right']:
   z=OxmlElement('w:'+edge);z.set(qn('w:val'),'single');z.set(qn('w:sz'),'4');z.set(qn('w:color'),'D9D9D9');b.append(z)
  tcpr.append(b);m=OxmlElement('w:tcMar')
  for edge in ['top','left','bottom','right']:
   z=OxmlElement('w:'+edge);z.set(qn('w:w'),'85');z.set(qn('w:type'),'dxa');m.append(z)
  tcpr.append(m)
  if row_index==0:
   shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DCE7F3');tcpr.append(shade)
  for p in c.paragraphs:
   p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(0)
   for run in p.runs:run.font.size=Pt(10.5);run.bold=row_index==0
head=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(head)
p=d.add_paragraph('The differences show that the automatic transition detector is not ready for unattended measurements. Agreement on one root is not proof of accuracy on new slides. Your earlier practice outlines are also not perfect; roots 4 and 5 have the later refinements.');p.paragraph_format.space_before=Pt(8)
d.add_paragraph('Start here when you are ready','Heading 1')
steps=[('Step 1 Open the review page','In the project folder, open 04_Results, then 50_Six_Root_Length_Proof. Double-click OPEN_ME_REVIEW_AND_MEASURE.html. No training is started.'),('Step 2 Look at one root','Choose Root 1. Use the zoom slider and scroll across the image. Check the cell shapes and where Periderm begins. The two comparison lines are hidden at first.'),('Step 3 Compare the lines','Turn on “Show outline-based line” and then “Show model-only line”. A difference between them is something to inspect; the computer has not automatically decided which is biologically correct.'),('Step 4 Make your own measurement','For a manual trace, click at the transition, then add points along the centre of the main root to the basal end. The last point is the end. Use Undo for a misplaced point. Do not follow a lateral root.'),('Step 5 Save before closing','Download the comparison CSV and the trace JSON. Keep both: the CSV stores lengths and the JSON stores the points you clicked. You can also enter a separate Fiji measurement in the comparison table.')]
for title,body in steps:
 p=d.add_paragraph();p.add_run(title+'. ').bold=True;p.add_run(body)
d.add_page_break();d.add_paragraph('What changed and what did not','Heading 1');d.add_paragraph('No new model was trained for this measurement demonstration. The existing three-category PAT transfer model was kept unchanged. I added full-slide prediction, root separation, centreline measurement, a six-root comparison table, and a local review page. Earlier models and results remain saved.')
d.add_paragraph('I also checked the released four-category PAT model and a different transition rule. Neither justified declaring the automatic detector reliable. Those exploratory results remain in folder 49; the six-root review uses the preserved three-category model with the documented basal-connected rule.')
d.add_paragraph('The meaning of the three measurements','Heading 1')
for title,txt in [('Model only','The model sees the image, predicts tissue categories, and a rule turns its Periderm region into a length. The transition can be wrong.'),('From your outlines','The computer uses the Periderm outlines you already drew to define the measured region. This is annotation-assisted measurement.'),('Manual measurement','You choose the transition and basal end and trace the root, for example in Fiji or in the local review page. No independent manual measurements have been supplied yet.')]:
 p=d.add_paragraph();p.add_run(title+'. ').bold=True;p.add_run(txt)
d.add_paragraph('Pixels and micrometres','Heading 1');d.add_paragraph('Fiji is one way to enter the microscope calibration. Python can use the same number. If the scale is 0.5299 pixels per micrometre, divide a pixel length by 0.5299. For example, 5,299 pixels equals 10,000 micrometres, or 10 millimetres. Cropping does not change this scale; resizing does.')
d.add_paragraph('You confirmed that scale for 20_29 and 2_12. It has not been separately confirmed for MT25, so the six-root table remains in pixels. A display zoom does not change the measurements.')
d.add_paragraph('Scientific check before use','Heading 1');d.add_paragraph('The PAT paper measures along the primary root between the hypocotyl region and the Periderm–Endoderm transition, including younger Periderm, and includes quality control. Our code is an adapted proof of concept, not an exact reproduction of the original PAT software. We must agree on the anatomical endpoints and compare with manual measurements before treating lengths as experimental results.')
d.add_paragraph('PAT paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/')
d.add_paragraph('What to learn next','Heading 1');d.add_paragraph('Begin with one root. Check its transition and basal end, make a manual measurement, then compare the numbers. This separates two questions: “Did the model identify the right tissue?” and “Did the computer measure the chosen path correctly?”')
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT;footer.add_run('Periderm learning record  |  ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
path=ROOT/'05_Notes/06_Periderm_Length_Proof_Walkthrough.docx';d.save(path);print(path)
