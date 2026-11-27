import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {constants as fsConstants} from 'node:fs';
const runtime=process.env.CODEX_PRIMARY_RUNTIME_ROOT || 'C:/Users/alexg/.cache/codex-runtimes/codex-primary-runtime';
process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES ??= path.join(runtime,'dependencies/node/node_modules');
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const SKILL='C:/Users/alexg/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const WORK=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const campaign=path.resolve(process.argv[2]);
const d=JSON.parse(await fs.readFile(path.join(campaign,'presentation_data.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(WORK,'assets/reference_template.pptx')));
p.slides.keep([0,3]);
const [cover,base]=p.slides.items;
const slides=[];
const snapshot=await p.inspect({kind:'shape,textbox,image,table,chart',limit:10000,maxChars:2000000});
const records=snapshot.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
const keep=new Set(['footer-date','footer-page','slide-title']);
p.delete(records.filter(r=>r.slideIndex===base.index&&!keep.has(r.name)&&!(r.name==='divider'&&Math.abs((r.position?.top??0)-143)<1)).map(r=>r.id));
const navy='#002B5E',gold='#948151',blue='#6F8FAE',body='#263645';
const models=['pinn','bounded_nn','standard_nn','random_forest','gradient_boosting'];
const labels=['Bound + smoothness','Bound only','Unconstrained NN','Random forest','Gradient boosting'];
const colors=[gold,blue,'#999999',navy,'#507A61'];
const chartOwners=[],tableOwners=[],noteText=[];
function text(s,str,x,y,w,h,size=27,bold=false,color=body){
 const t=s.shapes.add({geometry:'textbox',position:{left:60+(x-65)*840/1150,top:y,width:w*840/1150,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=str;t.text.style={typeface:'Arial',fontSize:size*0.75,bold,color,autoFit:'none',insets:0,wrap:'square'};return t;
}
function slide(title){
 const s=slides.length===0?cover:base.duplicate(); slides.push(s);
 s.shapes.items.find(v=>v.name==='footer-date').text='11 Nov 2026';
 s.shapes.items.find(v=>v.name==='footer-page').text=String(slides.length);
 const h=s.shapes.items.find(v=>v.name==='slide-title');if(h)h.text=title;
 return s;
}
function notes(s,script,source){const value='SPEAKER SCRIPT\n'+script+'\n\nSOURCES\n'+path.join(campaign,source)+'\nCampaign: '+d.campaign;s.speakerNotes.text=value;noteText.push({slide:slides.length,title:s.shapes.items.find(v=>['slide-title','cover-title'].includes(v.name)).text.toString(),notes:value});}
function chart(s,type,cfg){cfg.position={...cfg.position,left:60,top:178,width:840,height:385};function scaleFonts(o){for(const k of Object.keys(o??{})){if(k==='fontSize')o[k]*=0.8;else if(o[k]&&typeof o[k]==='object')scaleFonts(o[k]);}}scaleFonts(cfg);for(const ser of cfg.series){ser.values=ser.values.map(v=>Number(v.toPrecision(12)));if(type==='line')ser.smooth=false;}if(type==='line'){cfg.lineOptions={smooth:false};cfg.xAxis={...cfg.xAxis,tickLabelPosition:'low'};}const c=s.charts.add(type,cfg);applyPresentationChartFont(c,{fontFamily:'Arial'});chartOwners.push(slides.length);return c;}
function table(s,values,y,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:60,top:y,width:840,height:370,values,columnWidths:widths.map(w=>w*840/1150)});t.borders.assign({fill:'#161616',width:0.8});t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:15,right:15,top:8,bottom:8},anchor:'middle'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?navy:(r%2?'#FFFFFF':'#F7F5EF');cell.text.style={typeface:'Arial',fontSize:19,color:r===0?'#FFFFFF':body,bold:r===0,verticalAlignment:'middle',autoFit:'none'};}tableOwners.push(slides.length);return t;}
const sum=models.map(m=>d.summary.find(v=>v.model===m));
const avg=a=>a.reduce((x,y)=>x+y,0)/a.length;
const improvement=d.paired.filter(v=>v.delta_rmse_nm<0).length;
const delta=avg(d.paired.map(v=>v.delta_rmse_nm));
let s=slide('Bonded film thickness: research update');
cover.shapes.items.find(v=>v.name==='cover-title').text='Physics-Informed Neural Network\nResearch Update';
cover.shapes.items.find(v=>v.name==='cover-subtitle').text='Measured-data preparation and validation';
cover.shapes.items.find(v=>v.name==='Rectangle 6').text='Alexander Adams\nLi Lab\nNovember 11, 2026';
notes(s,'Today I will summarize the follow-up to the October update. We assembled the audited measurements into a PINN-ready table, repeated comparisons across held-out coating conditions, and tested what happens when we reduce the training data. There are 120 model fits, but no newly collected laboratory experiments. The main question is whether the physical restrictions improve prediction accuracy, rather than simply keeping predictions in an allowed range. I will also identify exactly what is still needed to activate the remaining physics terms.','RESEARCH_UPDATE.md');
s=slide('Data and comparison design');
table(s,[['Prepared measurements','Controlled comparison'],['58 hexane, 35 toluene, 33 ethyl acetate','Five models share the same held-out conditions'],['Concentration units corrected to g/L','Five folds; three initialization seeds'],['15 summaries and 1 invalid row excluded','Fewer training conditions; fixed test set']],175,[550,600]);
text(s,'Total thickness is a measured input: this is film-state inference.',65,565,1140,65,27,true,navy);
notes(s,'We started with 142 source records. Fifteen toluene summary rows may overlap the individual readings, and another toluene row reports bonded thickness above total thickness. Those 16 rows were excluded under the existing audit, leaving 126. We retained source filenames, row numbers and record IDs. The concentration values are in grams per liter. A condition means one solvent and concentration combination. We hold out all readings at a condition together. Five folds rotate conditions through the test role, while three seeds repeat model initialization on exactly the same splits. Batch IDs are still missing, so condition grouping is a practical proxy, not proof of independent experiments.','data_provenance.json');
s=slide('Prediction error on held-out conditions');
chart(s,'bar',{position:{left:65,top:155,width:1150,height:410},categories:labels,series:[{name:'Condition-weighted RMSE (nm)',values:sum.map(v=>v.group_rmse_mean),fill:blue,points:colors.map((fill,idx)=>({idx,fill}))}],hasLegend:false,barOptions:{direction:'column',gapWidth:85},xAxis:{textStyle:{fontSize:18},majorGridlines:null},yAxis:{min:0,title:'Error (nm), lower is better',numberFormatCode:'0.00',textStyle:{fontSize:18},majorGridlines:{fill:'#DFE5EC',width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:21,bold:true}}});
text(s,'Each coating condition receives equal weight; bars average three seeds.',65,590,1150,55,25);
notes(s,`The vertical axis is root mean squared error, expressed in nanometers. Lower is better. We first average squared errors within each condition, so a condition with many repeated readings does not dominate the score. The bound-plus-smoothness model averages ${sum[0].group_rmse_mean.toFixed(4)} nm, compared with ${sum[1].group_rmse_mean.toFixed(4)} nm for the bound-only model. The other bars are an unconstrained neural network and two tree-based reference models. These are new comparisons on the combined three-solvent dataset, so we should not read a change from the older hexane-only score as a controlled improvement. All figures are exploratory results on existing measurements. Seed variation is tabulated in the report; three seeds are not a confidence interval.`,'cv_summary.csv');
s=slide('Effect of the smoothness constraint');
chart(s,'line',{position:{left:65,top:155,width:1150,height:390},categories:['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'],series:[11,42,73].map((seed,i)=>({name:`Seed ${seed}`,values:d.paired.filter(v=>v.seed===seed).map(v=>v.delta_rmse_nm),line:{fill:colors[i],width:3},marker:{symbol:'circle',size:6}})),hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:20}},xAxis:{textStyle:{fontSize:19}},yAxis:{title:'Change in error (nm)',numberFormatCode:'0.000',textStyle:{fontSize:18},majorGridlines:{fill:'#DFE5EC',width:1}}});
text(s,`Below zero favors smoothness: improvement in ${improvement} of 15 comparisons.`,65,575,1150,70,27,true,navy);
notes(s,`Here we subtract the bound-only error from the error with added smoothness. A negative value means smoothness helped. Each line is one initialization seed, and each point is a different held-out group of conditions. Smoothness improves ${improvement} of 15 comparisons, with a mean paired change of ${delta.toFixed(4)} nanometers. The comparisons reuse data and are not independent experiments. The output bound guarantees nonnegative bonded thickness no greater than measured total; a smoothness penalty only discourages a wavy response. Neither fact guarantees accurate predictions. This is why we keep compliance and error as separate outcomes.`,'paired_curvature_effect.csv');
s=slide('Measured training-data requirements');
chart(s,'line',{position:{left:65,top:155,width:1150,height:390},categories:['25%','50%','75%','100%'],series:[0,1,3,4].map(i=>({name:labels[i],values:[.25,.5,.75,1].map(f=>avg(d.learning.filter(v=>v.model===models[i]&&v.fraction===f).map(v=>v.group_rmse_nm))),line:{fill:colors[i],width:3},marker:{symbol:'circle',size:6}})),hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},xAxis:{title:'Requested share of training conditions',textStyle:{fontSize:19}},yAxis:{min:0,title:'Error (nm), lower is better',numberFormatCode:'0.00',textStyle:{fontSize:18},majorGridlines:{fill:'#DFE5EC',width:1}}});
text(s,'Same validation and test conditions throughout; no synthetic rows added.',65,575,1150,70,27,true,navy);
notes(s,'This experiment removes complete training conditions while leaving validation and test measurements unchanged. The percentages refer to conditions rather than individual repeated readings, with integer rounding. Each point averages the same three initialization seeds. The full-data endpoint reuses the first fold of the main study. A lower error as more data become available supports collecting broader measured coverage, but this is only one fixed test fold and should not be read as a universal sample-size requirement. The full report includes all five models and the variation across seeds. We have not claimed that synthetic augmentation helps; this establishes its measured-only reference.','learning_curve_metrics.csv');
s=slide('Remaining physics inputs');
table(s,[['Component','Current status'],['Thickness bound and empirical smoothness','Implemented and tested on measured records'],['Withdrawal-speed smoothness','Needs matched, varying speed measurements'],['Hydrodynamic mobile-layer penalty','Needs matched fluid properties and dry conversion'],['Sample matching and applicability','126-row keyed intake prepared; verification pending']],170,[555,595]);
text(s,'Available values still need traceable sample matching before activation.',65,565,1150,65,27,true,navy);
notes(s,'The October update says the values needed for the remaining losses are available. I have not changed that statement. In this repository, however, I could not identify a sample-linked table with the required measured speed, solution properties and independently justified wet-to-dry conversion. Older analytical scripts use solvent constants and an assumed withdrawal speed, which cannot silently become measured inputs. I prepared an intake with all 126 record IDs and source rows so those measurements can be matched directly. The hydrodynamic relation would apply to the mobile layer under justified conditions; it is not an adsorption law for bonded thickness. The conversion must not be calculated from the bonded target, because that would reveal the answer to the model.','physics_measurement_intake.csv');
s=slide('Conclusions and next experiment');
text(s,'Completed',65,170,1080,45,30,true,gold);
text(s,'Audited combined data, repeated grouped comparisons,\nwhole-condition learning curves and archived figures.',65,225,1130,110,29);
text(s,'Next experiment',65,370,1080,45,30,true,gold);
text(s,'Match the remaining physics measurements, then test\neach added penalty on the same protected splits.',65,425,1130,105,29);
text(s,'New batches are still needed for independent validation.',65,580,1150,55,27,true,navy);
notes(s,'The main deliverable is a reproducible measured-data comparison, including every split, prediction, checkpoint and figure. We can now distinguish the effect of the bound, the effect of smoothness, and the effect of training-data quantity. The immediate next step is to supply the matched physics table and its applicability evidence, then enable velocity and hydrodynamic penalties separately before combining them. We should also obtain batch and specimen identities and reserve new laboratory measurements for a truly independent assessment. These existing-data comparisons do not identify an adsorption mechanism or establish recipe-only forecasting.','RESEARCH_UPDATE.md');
p.slides.keep(slides.map(s=>s.index));
for(let i=0;i<slides.length;i++)slides[i].moveTo(i);
await fs.mkdir(path.join(WORK,'.build','template-previews'),{recursive:true});await fs.mkdir(path.join(WORK,'output'),{recursive:true});
await fs.writeFile(path.join(WORK,'speaker_notes_template.json'),JSON.stringify(noteText,null,2));
for(let i=0;i<p.slides.items.length;i++){const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(WORK,'.build','template-previews',`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
const candidate=path.join(WORK,'.build','candidate-template.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const stamp=Date.now();
const prepared=path.join(WORK,'.build',`portable-${stamp}.pptx`);
const conversion=spawnSync(path.join(runtime,'dependencies/python/python.exe'),[path.join(WORK,'materialize_charts_windows.py'),path.join(SKILL,'container_tools/materialize_literal_chart_workbooks.py'),candidate,prepared,'--workspace',WORK,'--receipt',path.join(WORK,'.build',`chart-snapshot-${stamp}.json`)],{encoding:'utf8'});
if(conversion.status!==0)throw new Error(conversion.stdout+'\n'+conversion.stderr);
// This Windows workspace does not support hard-link publication. Preserve exclusive
// creation and all package/hash checks while publishing the validated bytes by copy.
const originalLink=fs.link;
fs.link=async(source,destination)=>{for(const q of [source,destination])if(!path.resolve(q).startsWith(path.resolve(WORK)+path.sep))throw new Error('Publication path outside deck workspace');await fs.copyFile(source,destination,fsConstants.COPYFILE_EXCL);};
const finalPath=path.join(WORK,'output','Li_Lab_Research_Update_2026-11-11_Template.pptx');
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:prepared,finalPath,
 pythonExecutable:path.join(runtime,'dependencies/python/python.exe'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit','--validate-bullet-geometry',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'reference',families:['Arial','Georgia'],referencePath:path.join(WORK,'assets/reference_template.pptx'),referenceSha256:createHash('sha256').update(await fs.readFile(path.join(WORK,'assets/reference_template.pptx'))).digest('hex')},explicitTotalSlideCount:7,requiredNativeChartOwnerSlides:chartOwners,requiredNativeTableOwnerSlides:tableOwners,materializeLiteralChartWorkbooks:false,verifyArtifactToolImport:true,receiptPath:path.join(WORK,'.build','validation-template.json')});
fs.link=originalLink;
console.log(JSON.stringify({finalPath:result.finalPath}));process.exit(0);
