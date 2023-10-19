#Essential importations
import gradio as gr
import numpy as np
import pandas as pd
from utils import PROD_TOK, AUX_TOK, preprocess

#Longitud máxima de una secuencia a evaluar
MAX_LEN = 30

with gr.Blocks() as my_demo:
    gr.Markdown("# Generador de datasets supervisados por humano con Gradio")
    choice = gr.Radio(['Piezas', 'Productos'], label="Elija que tipo de dataset se generará:")

    #Esta función genera una secuencia de piezas o productos aleatoriamente
    def generate(option:str):
        if option == "Piezas":
            keys = list(AUX_TOK.keys())
        
        elif option == "Productos":
            keys = list(PROD_TOK.keys())

        #Definimos la longitud de la secuencia a generar de forma aleatoria
        seq_len = np.random.randint(2, MAX_LEN)
        out_seq = []
        
        for i in range(seq_len):
            out_seq.append(np.random.choice(keys))
            
        return out_seq #Se devuelve una lista con las/los piezas/productos generados aleatoriamente
    
    demo = gr.Interface(generate, choice, 'text')


if __name__ == "__main__":
    my_demo.launch()