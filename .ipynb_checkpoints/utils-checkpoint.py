#En este archivo declaramos funciones muy utilizadas y constantes
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoConfig

#Declaramos como constantes los diccionarios con los tokens y las conversiones
PROD_TOK = {"OTHER": 0, "1200": 1, "1300": 2, "1400": 3, "1600": 4, "1700": 5, "1800": 6, "1900": 7, "2100": 8, "2400": 9, "2405": 9, "2600": 10, "2800": 11, "2805": 11, "AR": 12}

PIEZ_TOK = {"OTHER":0,"CAPUCHON": 1,"CUERPO": 2,"CPO": 2,"BONETE": 3,"ANILLO": 4,"DISCO": 5,"CPO.TOB": 6,"CPO TOB": 6,"CONTRATUERCA": 7,"TOBERA": 8,"BUJE": 5,"GUIA": 10, "GUÍA": 10, "RETEN": 11,"VASTAGO": 12,"VAST": 12,"TORN": 13,"PORTADISCO": 14,"TUERCA": 15,"CABEZA": 16,"APOYO": 17,"CAJA": 18,"OBTURADOR": 19}

#Este diccionario es un auxiliar en el que no hay piezas repetidas
AUX_TOK = {"OTHER":0,"CAPUCHON": 1,"CUERPO": 2,"BONETE": 3,"ANILLO": 4,"DISCO": 5,"CPO TOB": 6,"CONTRATUERCA": 7,"TOBERA": 8,"BUJE": 5,"GUIA": 10, "RETEN": 11, "VASTAGO": 12, "TORN": 13, "PORTADISCO": 14, "TUERCA": 15, "CABEZA": 16, "APOYO": 17, "CAJA": 18, "OBTURADOR": 19}

#La función preprocess pasa de secuencia de texto a tokens
def preprocess(input:list[str], tokens:dict):
    keys = [i for i in tokens.keys()]
    processed = np.zeros(len(input), dtype=int)
    for i in range(len(input)):
        for k in keys:
            if k in input[i]:
                processed[i] = tokens[k]
                break
            else:
                processed[i] = tokens['OTHER']
    return torch.tensor(np.asarray([processed]))

#La función sentence distance lo que hace es calcular la L2 norm entre 2 vectores que son la salida intermedia de un modelo
def sentence_distance(sent1:torch.tensor, sent2:torch.tensor, model):
    global features
    features = {}
    model(input_ids=sent1)
    feat1 = features['feats'][0][0]
    features = {}
    model(input_ids=sent2)
    feat2 = features['feats'][0][0]
    return np.linalg.norm(feat1.cpu()-feat2.cpu())

#La función sequence similarity retorna la  similitud L2 entre dos secuencias de piezas o productos no preprocesadas
def sequence_similarity(inp1:list[str], inp2:list[str], model:str, tokens:dict, device:torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    model.bert.encoder.layer[11].output.dense.register_forward_hook(get_features('feats'))
    sent1 = preprocess(inp1, tokens).to(device)
    sent2 = preprocess(inp2, tokens).to(device)
    return sentence_distance(sent1, sent2, model)

#This is the class corresponding to the custom reward model
class RewardModel (torch.nn.Module):
    def __init__(self, model_name:str, hidden_size:int, classes:int,*model_args, **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.vocab_size = classes+1
        self.config.pad_token_id = classes
        self.config.hidden_size = hidden_size
        #print(self.config)
        self.base = AutoModelForSequenceClassification.from_config(self.config)

        #print(self.base.classifier)
        #Last layer must be changed for it to be a regression problem
        self.base.classifier =  torch.nn.Linear(in_features=hidden_size, out_features=1, bias=True)

        #Only keep this line if the model was not pre-trained (it's a fresh BERT model)
        #PAD = max(list(AUX_TOK.values()))+1 #Change this line according to the model (pieces/products)
        #self.base.bert.embeddings.word_embeddings = torch.nn.Embedding(PAD+1, hidden_size, PAD)
        #self = self.base

    #Our custom model should take as input a pair of right/wrong answers with a fixed sequence
    def forward(self, input_ids, attention_mask=None,
                token_type_ids=None, train:bool=False):
        #Whether the model is being trained or tested is important when inferencing
        if train:
            #The input_ids are structured in a way in which the first chunk corresponds to right examples and the second one to wrong ones
            input_right, input_wrong  = input_ids[0][0], input_ids[0][1]
            out_right = self.base(input_ids=input_right, attention_mask=attention_mask, token_type_ids=token_type_ids)
            out_wrong = self.base(input_ids=input_wrong, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return [out_right, out_wrong]
        else:
            return self.base(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    #Setup a custom .train() function because the base model will not train
    def train(self, mode:bool):
        self.base.train(mode=mode)
        return self.base


