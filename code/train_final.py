import torch
from torch import nn
from torch.utils.data import DataLoader
from image import CSRImageEncoder
from text import CSRTextEncoder
import os
from tqdm import tqdm
import json
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
import csw1 as eval_w
import eval_coco

use_cuda=torch.cuda.is_available()
print("torch.cuda.is_available",torch.cuda.is_available())
eval_threshold = 1
use_cuda=1
temperature=1.0
saving_threshold=10
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4096, help='batch size')                    
parser.add_argument('--lr', dest='lr', default = 0.00001 ,help='learning rate')                                        
parser.add_argument('--size', dest='size', default = "mid" ,help='dataset size')
parser.add_argument('--gpu_no', dest='gpu_no', default = "0" ,help='gpu_no') 
parser.add_argument('--k_top', dest='k_top', type=int, default = 256 ,help='k_top') 
parser.add_argument('--sent_path', dest='sent_path', type=str, default = 'train_only_mscoco_normalized.json' ,help='sent_path') 
parser.add_argument('--get_emb', dest='get_emb', type=int, default = 0 ,help='k_top') 

params = vars(parser.parse_args())
dev="cuda:"+params['gpu_no']
device = torch.device(dev if torch.cuda.is_available() else "cpu")


def load_json(file):
    with open(file,'r') as f:
        data=json.load(f)
    return data

def read_txt_file(file):
    text_dict={}
    with open(file,'r') as f:
        f1=f.readlines()
        for i in f1:
            k=i.strip().split()
            arr=np.array([float(j) for j in k[1:]])
            text_dict[k[0]]=arr
    return text_dict

get_emb = params['get_emb']
sent_path = params['sent_path']
num_epochs = int(params['num_epochs'])
bs = int(params['batch_size'])
lr = params['lr']
size = params['size']
top_k=params['k_top']
k_top=params['k_top']
if get_emb==0:
    pretrained_text_embeddings =  load_json("mscoco_train/clip_text_embs.json")
    print("1")
    pretrained_image_embeddings =  load_json("mscoco_train/clip_image_embs.json")
    print("2")
    sentences_path = os.path.join("mscoco_train",sent_path)
    sentences = load_json(sentences_path)
mappings = load_json("mscoco_train/mappings_rev.json")
val_text_embeddings = load_json("mscoco_val/mscoco_val_clip_text_id.json")
val_image_embeddings = load_json("mscoco_val/mscoco_val_clip_image_id.json")

def build_sparse_mask(texts):
    N = len(texts)
    V = 1000
    data=[]
    mask = torch.zeros([N,V])
    #mask = mask.to(device)
    for i in texts:
        #print(sent[i])
        dat=sentences[i]
        data.append(dat)
    data1=np.array(data)
    return torch.tensor(data1).to(device)

# desired_gpus = "0,2,3"  # Comma-separated GPU indices
# os.environ["TORCH_VISIBLE_DEVICES"] = desired_gpus
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device_ids = [0, 1,2,3]

def dump_vectors(X, outfile, words):
	print ("shape", X.shape)
	fw = open(outfile, 'w')
	for i in range(len(words)):
		fw.write(words[i] + " ")
		for j in X[i]:
			fw.write(str(j) + " ")
		fw.write("\n")
	fw.close()

def dump_image_vectors(X, outfile, words):
	print ("shape", X.shape)
	fw = open(outfile, 'w')
	unique_words=[]
	for i in range(len(words)):
		if words[i] not in unique_words:
			unique_words.append(words[i])
		
			fw.write(words[i] + " ")
			for j in X[i]:
				fw.write(str(j) + " ")
			fw.write("\n")
	fw.close()

def get_image_dict(X,words):
	unique_words=[]
	dic={}
	for i in range(len(words)):
		if words[i] not in unique_words:
			arr=[]
			unique_words.append(words[i])
			for j in X[i]:
				arr.append(j)
			dic[words[i]]=arr
	return dic

def get_text_dict(X,words):
	dic={}
	for i in range(len(words)):
		arr=[]
		for j in X[i]:
			arr.append(j)
		dic[words[i]]=arr
	return dic

def getEmbeddings(text_dict_path, mappings_path,model,text_filename,image_filename,rank):
	data_loader = prepare(rank, world_size=4,batch_size=bs)
	image_emb,text_emb,image_n,text_n=[],[],[],[]
	for images, texts,image_name,text_name in data_loader:
		mask = bow_mask = build_sparse_mask(list(texts),rank)
		image_embeddings, text_embeddings, _, _,_,_ = model(images, list(texts),image_name,text_name,mask)
		image_emb.extend(image_embeddings.cpu().data.numpy())
		text_emb.extend(text_embeddings.cpu().data.numpy())
		image_n.extend(image_name)
		text_n.extend(text_name)
	 
	dump_image_vectors(np.array(image_emb),image_filename,image_n)	
	dump_vectors(np.array(text_emb),text_filename,text_n)
	print("saveddd...........")

def cosine_similarity(tensor1, tensor2):
  cosine_sim = torch.einsum('i,i->', tensor1, tensor2)
  return cosine_sim

def cos(t1,t2):
	cos1 = torch.nn.CosineSimilarity(dim=0)
	cos2=cos1(t1,t2)
	return 1-cos2


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_loss_new1(image_embeddings, text_embeddings,image_names,text_names,mask, margin=0.001):
  if mask== None:
      image_embs=image_embeddings#.to('cpu')*mask.to('cpu')
      text_embs=text_embeddings#.to('cpu')*mask.to('cpu')
  else:
      image_embs=image_embeddings*mask#.to('cpu')
      text_embs=text_embeddings*mask#.to('cpu')  
  image_name=image_names.tolist()
  text_name=text_names.tolist()
  image_names=[str(f"{int(i):06}") for i in image_name]
  text_names=[str(f"{int(i):012}") for i in text_name]
  text_dict,image_dict,text_img_map={},{},{}
  text_em,image_em=[],[]

  for i in range(len(image_names)):
  	image_dict[image_names[i]]=image_embs[i]

  for i in range(len(text_names)):
  	text_dict[text_names[i]]=text_embs[i]
  	#text_dict_init[text_names[i]]=text_init_emb[i]
    
  for i in text_dict.keys():
    text_em.append(text_dict[i])
    image_em.append(image_dict[i[0:6]])
  	
  text_embeddings1=torch.stack(text_em,dim=0)#np.array(text_em)
  image_embeddings1=torch.stack(image_em,dim=0)#np.array(image_em)

  logits = (text_embeddings1 @ image_embeddings1.T) / temperature
  images_similarity = image_embeddings1 @ image_embeddings1.T
  texts_similarity = text_embeddings1 @ text_embeddings1.T
  targets = F.softmax((images_similarity + texts_similarity) / 2 * temperature, dim=-1)
  texts_loss = cross_entropy(logits, targets, reduction='none')
  images_loss = cross_entropy(logits.T, targets.T, reduction='none')
  #print("ttt",targets)
  #print("iii,tttt",images_loss,texts_loss)
  loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
  return loss.mean()

class ContrastiveModel(nn.Module):
  def __init__(self):
    super(ContrastiveModel, self).__init__()
    self.text_encoder = CSRTextEncoder()
    self.image_encoder = CSRImageEncoder()#self.image_config)

  def forward(self, images, texts,image_names,text_names,bow_mask,train):    
    text_embeddings,text_recon,text_names1 = self.text_encoder(texts,text_names,bow_mask,train,device,top_k)
    image_embeddings,image_recon,image_names1 = self.image_encoder(images,image_names,bow_mask,train,device,top_k)
    return image_embeddings, text_embeddings,image_recon,text_recon,image_names1,text_names1

def evaluate(model, data_loader, device,epoch,get_emb):
    model.eval()  # Set the model to evaluation mode
    total_loss,co,ir,tr = 0.0,0.0,0.0,0.0
    local_image_emb_s, local_text_emb_s, local_image_n_s, local_text_n_s = [], [], [], []
    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, texts, image_names, text_names in data_loader:
            image_name = torch.tensor([int(i) for i in image_names], dtype=torch.int64).to(device)
            text_name = torch.tensor([int(i) for i in text_names], dtype=torch.int64).to(device)
            #bow_mask = build_sparse_mask(list(text_names))
            image_embeddings, text_embeddings, image_recon, text_recon, _, _ = model(images.to(device), texts.to(device), image_name, text_name, None, train=False)
            contrastive_loss = contrastive_loss_new1(image_embeddings, text_embeddings, image_name, text_name, None)
            total_l = contrastive_loss.item() + image_recon.item() + text_recon.item()
            total_loss+=total_l
            co+=contrastive_loss.item()
            ir+=image_recon.item()
            tr+=text_recon.item()
            if epoch%1==0:
                local_image_emb_s.extend(image_embeddings.cpu().data.numpy())
                local_text_emb_s.extend(text_embeddings.cpu().data.numpy())
                local_image_n_s.extend(image_names)
                local_text_n_s.extend(text_names)
    avg_loss = total_loss / len(data_loader)
    avg_cl = co/len(data_loader)
    avg_ir = ir/len(data_loader)
    avg_tr = tr/len(data_loader)
    print(f"Evaluation Loss: {avg_loss:.4f}",avg_cl,avg_ir, avg_tr)
    dirname = "all_"+str(sent_path.split('.')[0])+str(bs)+"_" + str(lr) +"_"+str(k_top)
    if not os.path.exists(dirname):
    	os.makedirs(dirname)
    output_file = dirname+"/"+"results.txt"
    text_filename =  dirname+"/"+str(epoch)+"_output_text.txt"
    image_filename = dirname+"/"+str(epoch)+"_output_image.txt"
    model_filename = dirname+"/"+str(epoch)+"model.pt"
    print(text_filename,image_filename,epoch,num_epochs)
    if get_emb==1:
    	dirname = "eval_all_"+str(sent_path.split('.')[0])+str(bs)+"_" + str(lr) +"_"+str(k_top)
    	if not os.path.exists(dirname):
    		os.makedirs(dirname)
    	output_file = dirname+"/"+"results.txt"
    	text_filename =  dirname+"/"+str(epoch)+"_output_text.txt"
    	image_filename = dirname+"/"+str(epoch)+"_output_image.txt"
    	dump_image_vectors(np.array(local_image_emb_s), image_filename, local_image_n_s)
    	dump_vectors(np.array(local_text_emb_s), text_filename, local_text_n_s)
    else:
    	if epoch%5==0:
    	    	img_dict=get_image_dict(np.array(local_image_emb_s), local_image_n_s)
    	    	text_dict=get_text_dict(np.array(local_text_emb_s), local_text_n_s)
    	    	torch.save(model.state_dict(), model_filename)
		#print(np.array(local_image_emb_s))
    	    	p1,p5,p10 =eval_coco.acc_i2t(img_dict,text_dict,device)
    	    	mrr1,mrr3,mrr10,n1,n3,n10,pr1,pr3,pr10 = eval_w.acc(img_dict,model_filename,top_k,device)
    	    	with open(output_file, 'a') as f:
    	    	    	f.write(f"Epoch: {epoch:.4f}\n")
    	    	    	f.write(f"I2T: {p1:.4f}, {p5:.4f}, {p10:.4f}\n")
    	    	    	f.write(f"Exc: {mrr1:.4f}, {mrr3:.4f}, {mrr10:.4f}, {n1:.4f}, {n3:.4f}, {n10:.4f},{pr1:.4f}, {pr3:.4f}, {pr10:.4f}\n")
    	    	    	f.write("\n")


def train(model, optimizer,scheduler, data_loader,device,dataloader_eval):
  # Track training statistics
  epoch_loss = 0.0  # Cumulative loss for each epoch
  total_loss = 0.0  # Cumulative loss across all epochs
  cl,ir,tr = 0.0,0.0,0.0
  for epoch in tqdm(range(num_epochs), desc="Training"):
    K=0
    #data_loader.sampler.set_epoch(epoch)
    image_emb_s,text_emb_s,image_n_s,text_n_s=[],[],[],[]
    local_image_emb_s, local_text_emb_s, local_image_n_s, local_text_n_s = [], [], [], []
    for images, texts,image_names,text_names in data_loader:
      optimizer.zero_grad()
      image_name = torch.tensor([int(i) for i in image_names], dtype=torch.int64).to(device)
      text_name = torch.tensor([int(i) for i in text_names], dtype=torch.int64).to(device)
      #print(image_name,text_name)
      bow_mask = build_sparse_mask(list(text_names))
      image_embeddings, text_embeddings, image_recon, text_recon,image_names1,text_names1 = model(images.to(device), texts.to(device),image_name,text_name,bow_mask,train=True)
      contrastive_loss= contrastive_loss_new1(image_embeddings, text_embeddings,image_names1,text_names1,None)  
      if epoch % saving_threshold == 0:
        local_image_emb_s.extend(image_embeddings.cpu().data.numpy())
        local_text_emb_s.extend(text_embeddings.cpu().data.numpy())
        local_image_n_s.extend(image_names)
        local_text_n_s.extend(text_names)
      total_loss = contrastive_loss + image_recon + text_recon
      if K%70==0:
        print("losses",total_loss.item(),contrastive_loss.item(),image_recon.item(), text_recon.item(),K,epoch)
      K+=1
      cl+=contrastive_loss.item()
      ir+=image_recon.item()
      tr+=text_recon.item()
      # Backward pass and update weights
      total_loss.backward()
      optimizer.step()
     
    epoch_loss += total_loss.item()
      
    avg_cl = cl /len(data_loader)
    avg_ir = ir /len(data_loader)
    avg_tr = tr /len(data_loader)
    avg_epoch_loss = epoch_loss / len(data_loader)
    print(f"Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
    print("cl,ir,tr",avg_cl,avg_ir,avg_tr)
    if epoch %eval_threshold ==0:
        evaluate(model,dataloader_eval,device,epoch,get_emb)
    # Reset epoch loss for the next epoch
    
    epoch_loss = 0.0

  # Print final training statistics (optional)
  scheduler.step()
  print(f"Total Training Loss: {total_loss:.4f}")

class ImageTextPairDataset():
  """Dataset class for image-text pairs."""

  def __init__(self,train1):
    if train1:
        self.pretrained_text_embeddings = pretrained_text_embeddings
        self.pretrained_text_embeddings = pretrained_text_embeddings        	
        self.pretrained_image_embeddings = pretrained_image_embeddings
        self.mappings = mappings
    else:
        self.pretrained_text_embeddings = val_text_embeddings
        self.pretrained_image_embeddings = val_image_embeddings
        #self.mappings = val_mappings
    self.image_text_pairs = []

    for text_id, text_content in self.pretrained_text_embeddings.items():
      text_name = text_id#self.mappings['text_mappings'][text_id]
      image_emb = self.pretrained_image_embeddings[text_name[0:6]]
      self.image_text_pairs.append((torch.tensor(image_emb), torch.tensor(text_content), text_name[0:6], text_name))

  def __len__(self):
    return len(self.image_text_pairs)

  def __getitem__(self, idx):
    image_path, text, text_id, full_text_id = self.image_text_pairs[idx]
    # You might need to perform additional data loading or preprocessing here (e.g., loading images)
    return self.image_text_pairs[idx]#image_path, text, text_id, full_text_id

def main():
    model = ContrastiveModel().to(device)
    print("d111")
    if get_emb==1:
    	model_filename = "all_train_only_mscoco_normalized4096_0.0001_256/30model.pt"
    	model.load_state_dict(torch.load(model_filename))
    	print("d2")
    	model.eval()
    	data_list_eval = ImageTextPairDataset(train1=False)
    	data_loader_eval = DataLoader(data_list_eval, batch_size=bs, shuffle=False)
    	evaluate(model, data_loader_eval, device,555,get_emb)
    else:
    	optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    	data_list = ImageTextPairDataset(train1=True)
    	data_loader = DataLoader(data_list, batch_size=bs, shuffle=True)
    	scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    	margin = 1.0  # Hyperparameter for contrastive loss (consider margin mining techniques)
     
    	data_list_eval = ImageTextPairDataset(train1=False)
    	data_loader_eval = DataLoader(data_list_eval, batch_size=bs, shuffle=False)
    	train(model, optimizer,scheduler,data_loader,device,data_loader_eval)#,rank,world_size)

if __name__ == '__main__':
      main()
