#Essential importations
import gradio as gr
import numpy as np
import pandas as pd
from utils import PROD_TOK, AUX_TOK, preprocess

#Longitud máxima de una secuencia a evaluar
MAX_LEN = 30

with gr.Blocks() as my_demo:
    gr.Markdown("# Generador de datasets supervisados por humano con Gradio")
    choice = gr.Radio(['Piezas', 'Productos'], value='Piezas', label="Elija que tipo de dataset se generará:")
    options = gr.Radio([i for i in list(AUX_TOK.keys())], label="Elija la pieza a producir:", interactive=True)
    
    #Se crea un dataframe para guardar los ejemplos de entrenamiento
    df = pd.DataFrame(data={'Input_seq':[None],'Output':[None],'Type':[None]})
    
    #Esta función genera una secuencia de piezas o productos aleatoriamente
    def generate(option:str):
        global out_seq
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

    #Esta función cambia la opción para que se puedan elegir entre piezas o productos
    def change_options(option:str):
        if option == "Piezas":
            options = gr.Radio([i for i in list(AUX_TOK.keys())], label="Elija la pieza a producir:", interactive=True)
        elif option == "Productos":
            options = gr.Radio([i for i in list(PROD_TOK.keys())], label="Elija el producto a producir:", interactive=True)
        return options

    def save(option:str, selected:str):
        global out_seq
        global df
        row = pd.Series([out_seq, selected, option], index=df.columns)
        df = df.append(row)
        return df
        
    #El generator es simplemente para que cada vez que el usuario le de click se genere la secuencia
    generator = gr.Interface(generate, choice, 'text')
    #Con el choice.change logro que se actualice cada vez que el usuario cambia de piezas a productos
    choice.change(change_options, choice, options)
    #El saver guarda la secuencia y lo que el usuario eligió como salida
    saver = gr.Interface(save, [choice, options], 'text')
    


if __name__ == "__main__":
    my_demo.launch()


