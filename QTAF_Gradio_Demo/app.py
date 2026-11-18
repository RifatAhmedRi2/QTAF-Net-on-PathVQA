import json,re,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image,ImageOps
from models import late,image,question,san,qtaf,ablation,resnet

ROOT=Path(__file__).parent
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
NAMES={
 'CNN + BiLSTM':('late','best_late_fusion_finetuned.pth',55.23),
 'QTAF without attention':('no_attention','qtaf_ablation_no_attention_finetuned_best.pth',55.19),
 'Stacked Attention Network':('san','best_san_finetuned.pth',55.02),
 'QTAF without type fusion':('no_type_fusion','qtaf_ablation_no_type_fusion_finetuned_best.pth',54.84),
 'QTAF single shared head':('single_head','qtaf_ablation_single_head_finetuned_best.pth',54.55),
 'ResNet18 + BiLSTM':('resnet','best_resnet18_bilstm_finetuned.pth',54.50),
 'Full QTAF-Net':('qtaf','best_qtaf_net_finetuned.pth',54.40),
 'Question-only BiLSTM':('question','best_question_only_bilstm_finetuned.pth',48.79),
 'Image-only CNN':('image','best_image_only_cnn_finetuned.pth',31.64),
}
METRICS=pd.DataFrame([
 ['CNN + BiLSTM',55.23,86.76,23.65],['QTAF without attention',55.19,86.50,23.83],
 ['Stacked Attention Network',55.02,86.85,23.15],['QTAF without type fusion',54.84,86.59,23.06],
 ['QTAF single shared head',54.55,86.20,22.85],['ResNet18 + BiLSTM',54.50,86.79,22.16],
 ['Full QTAF-Net',54.40,86.26,22.49],['Question-only BiLSTM',48.79,84.77,12.75],
 ['Image-only CNN',31.64,53.69,9.56]],columns=['Model','Overall EM (%)','Yes/No EM (%)','Open EM (%)'])

_vocab=None
_cache={}
def get_vocab():
 global _vocab
 if _vocab is None:
  p=ROOT/'vocab.json'
  if not p.exists():raise RuntimeError('Missing vocab.json. Run `python prepare_vocab.py` first, using the original PathVQA training split.')
  _vocab=json.loads(p.read_text())
  assert len(_vocab['word2idx'])==3972 and len(_vocab['open_answer2idx'])==1000
 return _vocab

def build_model(kind):
 vocab=get_vocab();v=len(vocab['word2idx']);a=len(vocab['open_answer2idx'])
 constructors={
 'late':lambda:late.LateFusionVQA(v,a), 'image':lambda:image.ImageOnlyCNN(a),
 'question':lambda:question.QuestionOnlyBiLSTM(v,a),
 'san':lambda:san.StackedAttentionNetwork(v,a,num_hops=2),
 'qtaf':lambda:qtaf.QTAFNet(v,a),
 'no_attention':lambda:ablation.AblatedQTAFNet(v,a,'no_attention'),
 'no_type_fusion':lambda:ablation.AblatedQTAFNet(v,a,'no_type_fusion'),
 'single_head':lambda:ablation.AblatedQTAFNet(v,a,'single_head'),
 'resnet':lambda:resnet.ResNet18BiLSTMBaseline(v,a)}
 return constructors[kind]()

def load_model(name):
 if name in _cache:return _cache[name]
 kind,filename,_=NAMES[name]
 m=build_model(kind)
 state=torch.load(ROOT/'models'/filename,map_location='cpu',weights_only=True)
 m.load_state_dict(state,strict=True)
 m.to(DEVICE).eval()
 _cache[name]=m
 return m

def encode_inputs(pil,question_text):
 vocab=get_vocab()['word2idx']
 tokens=re.findall(r'\w+|[^\w\s]',question_text.lower().strip())[:32]
 ids=[vocab.get(t,1) for t in tokens]
 ids=ids+[0]*(32-len(ids))
 img=ImageOps.contain(pil.convert('RGB'),(224,224),method=Image.Resampling.BILINEAR)
 padded=Image.new('RGB',(224,224),(255,255,255))
 padded.paste(img,((224-img.width)//2,(224-img.height)//2))
 pixels=torch.from_numpy(np.asarray(padded,dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
 return pixels,torch.tensor([ids],dtype=torch.long,device=DEVICE),padded

def predict_one(name,im,q,answer_type):
 kind=NAMES[name][0]
 model=load_model(name)
 with torch.inference_mode():
  if kind=='image':out=model(im)
  elif kind=='question':out=model(q)
  elif kind in ('qtaf','san'):out=model(im,q,return_attention=True)
  else:out=model(im,q)
  if kind=='single_head':
   all_logits=out[0]
   logits=all_logits[:,:2] if answer_type=='Yes/No' else all_logits[:,2:]
  else:logits=out[0] if answer_type=='Yes/No' else out[1]
  probs=F.softmax(logits,dim=-1)[0]
  confidence,index=probs.max(dim=0)
  idx=int(index.item())
  if answer_type=='Yes/No':answer=['no','yes'][idx]
  else:answer={v:k for k,v in get_vocab()['open_answer2idx'].items()}[idx]
  attention=None
  if kind=='qtaf':attention=out[3][0]
  elif kind=='san':
   # Last attention hop is returned in the SAN implementation.
   attention=out[2]
 return answer,round(float(confidence)*100,2),attention

def visualize_attention(att,padded):
 if att is None:return None
 if isinstance(att,(list,tuple)):att=att[-1]
 if isinstance(att,torch.Tensor):att=att.detach().float().cpu().squeeze()
 else:return None
 if att.numel()!=784:return None
 a=att.reshape(28,28).numpy()
 a=(a-a.min())/(a.max()-a.min()+1e-8)
 heat=Image.fromarray(np.uint8(a*255)).resize((224,224),Image.Resampling.BILINEAR)
 import matplotlib.cm as cm
 overlay=Image.fromarray(np.uint8(cm.inferno(np.asarray(heat)/255)[:,:,:3]*255))
 return Image.blend(padded,overlay,0.42)

def normalize_answer(value):
 return re.sub(r'\s+', ' ', str(value).strip().lower())

def run(image_input,question_text,answer_type,selection,ground_truth):
 if image_input is None or not question_text.strip():raise gr.Error('Please upload an image and enter a question.')
 try:im,q,padded=encode_inputs(image_input,question_text)
 except Exception as exc:raise gr.Error(str(exc))
 names=list(NAMES) if selection=='Compare all 9 models' else [selection]
 truth=normalize_answer(ground_truth) if ground_truth and ground_truth.strip() else ''
 rows=[];san_heat=None;qtaf_heat=None
 for name in names:
  try:
   answer,confidence,att=predict_one(name,im,q,answer_type)
   correctness=('Correct' if normalize_answer(answer)==truth else 'Incorrect') if truth else 'Not provided'
   rows.append([name,answer,ground_truth.strip() if truth else '—',correctness,confidence,'OK'])
   if att is not None:
    if NAMES[name][0]=='san':san_heat=visualize_attention(att,padded)
    elif NAMES[name][0]=='qtaf':qtaf_heat=visualize_attention(att,padded)
  except Exception as exc:
   rows.append([name,'—',ground_truth.strip() if truth else '—','Not evaluated',None,f'Error: {str(exc)[:110]}'])
 return pd.DataFrame(rows,columns=['Model','Predicted answer','Ground truth','Result','Confidence (%)','Status']),san_heat,qtaf_heat

# Wide, responsive layout: controls beside image, full-width results below.
CUSTOM_CSS='''
.gradio-container {max-width: 1600px !important; margin: auto !important;}
#live-results .wrap {overflow-x: auto;}
#live-results table {font-size: 14px;}
#live-results th, #live-results td {white-space: normal !important; overflow-wrap: anywhere;}
'''
with gr.Blocks(title='QTAF-Net | Pathology VQA') as demo:
 gr.Markdown('# QTAF-Net | Pathology Visual Question Answering')
 gr.Markdown('⚠️ **Research prototype only!** Not for diagnosis or clinical decision-making. Predictions may be incorrect. The reported test accuracies are fixed research results, not measurements on your uploaded image.')
 with gr.Tab('Live inference & model comparison'):
  with gr.Row(equal_height=False):
   with gr.Column(scale=1,min_width=350):
    img=gr.Image(type='pil',label='Pathology image',height=330)
   with gr.Column(scale=1,min_width=350):
    question_box=gr.Textbox(label='Question',placeholder='What is shown in the pathology image?',lines=3)
    type_box=gr.Radio(['Yes/No','Open-ended'],value='Yes/No',label='Answer category (select manually)')
    ground_truth_box=gr.Textbox(label='Ground truth (correct answer from PathVQA CSV)',placeholder='e.g. yes (optional)')
    select=gr.Dropdown(['Compare all 9 models']+list(NAMES),value='Compare all 9 models',label='Model')
    go=gr.Button('Run inference',variant='primary')
  gr.Markdown('### Live predictions — all selected models')
  output=gr.Dataframe(label='Live predictions',interactive=False,wrap=True,
                      max_height=430,elem_id='live-results',
                      column_widths=['27%','18%','17%','14%','12%','12%'])
  gr.Markdown('### Attention visualizations')
  with gr.Row(equal_height=True):
   with gr.Column(scale=1,min_width=300):
    san_attention=gr.Image(label='SAN attention overlay',interactive=False,height=330)
   with gr.Column(scale=1,min_width=300):
    qtaf_attention=gr.Image(label='Full QTAF-Net attention overlay',interactive=False,height=330)
  go.click(run,[img,question_box,type_box,select,ground_truth_box],[output,san_attention,qtaf_attention])
  gr.Markdown('**Note:** Enter the original correct answer from your PathVQA CSV to display Correct/Incorrect. Leave it blank to show predictions without grading. Select the correct answer category manually. The held-out evaluation used oracle answer-type routing; automatic routing is a separate QTAF-Net experiment. Confidence is the maximum softmax probability, not calibrated clinical certainty.')
 with gr.Tab('Held-out research results'):
  gr.Markdown('### Final exact-match accuracy on the PathVQA test split')
  gr.Dataframe(METRICS,interactive=False)
  gr.Markdown('**Best overall:** CNN + BiLSTM (55.23%). **Best Yes/No:** SAN (86.85%). **Best open-ended:** QTAF without attention (23.83%).')
  gr.Markdown('The top-1,000 open-answer vocabulary excludes **1,128 / 3,357 (33.6%)** held-out open-ended answers. The exploratory 2,000-answer CNN + BiLSTM experiment reached **52.63% validation overall EM**, versus **54.42%** with 1,000 answers; it was not part of the nine-model final test comparison.')
 with gr.Tab('Methods & limitations'):
  gr.Markdown('''### Inference protocol
- Images: RGB, aspect-preserving resize, white padding, 224 × 224, pixels scaled to [0,1].
- Questions: lowercased, punctuation-aware tokens, 32-token length, training-only vocabulary.
- Open-ended predictions: top 1,000 training answers. Yes/No: separate binary output head, except the single-head ablation.
- All nine architectures are loaded from the original notebook definitions with **strict** checkpoint loading.
- For a fair demonstration, select the true question type. Model answers may be wrong even when confidence is high.
- SAN and QTAF attention overlays are descriptive visualizations, not medical explanations.
- A single training seed was used; small accuracy differences should not be overinterpreted.''')
if __name__=='__main__':
 demo.launch(
     server_name='127.0.0.1',
     server_port=int(os.getenv('PORT','7860')),
     theme=gr.themes.Soft(primary_hue='cyan'),
     css=CUSTOM_CSS
 )