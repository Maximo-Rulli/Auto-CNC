#Essential importations
import gradio as gr
import numpy as np
import pandas as pd
from utils import PROD_TOK, AUX_TOK, preprocess

#Longitud máxima de una secuencia a evaluar
MAX_LEN = 30

with gr.Blocks() as main:
    gr.Markdown("# Generador de datasets supervisados por humano con Gradio")
    choice = gr.Radio(['Piezas', 'Productos'], value='Piezas', label="Elija que tipo de dataset se generará:")
    options = gr.Radio([i for i in list(AUX_TOK.keys())], label="Elija la pieza a producir:", interactive=True)
    num = 0 #num lleva la cuenta de ejemplos que se generaron
    
    #Se crea un dataframe para guardar los ejemplos de entrenamiento
    df = pd.DataFrame(data={'Input_seq':[None],'Output':[None],'Type':[None]})
    df.drop([0], inplace=True)
    
    #Esta función genera una secuencia de piezas o productos aleatoriamente
    def generate(option:str):
        global input_seq
        if option == "Piezas":
            keys = list(AUX_TOK.keys())
            
        elif option == "Productos":
            keys = list(PROD_TOK.keys())
            
        #Definimos la longitud de la secuencia a generar de forma aleatoria
        seq_len = np.random.randint(2, MAX_LEN)
        input_seq = []
        
        for i in range(seq_len):
            input_seq.append(np.random.choice(keys))
            
        return input_seq #Se devuelve una lista con las/los piezas/productos generados aleatoriamente

    #Esta función cambia la opción para que se puedan elegir entre piezas o productos
    def change_options(option:str):
        if option == "Piezas":
            options = gr.Radio([i for i in list(AUX_TOK.keys())], label="Elija la pieza a producir:", interactive=True)
        elif option == "Productos":
            options = gr.Radio([i for i in list(PROD_TOK.keys())], label="Elija el producto a producir:", interactive=True)
        return options

    def save(option:str, selected:str):
        global input_seq
        global df
        global num
        if option == "Piezas":
            tokens = AUX_TOK
        elif option == "Productos":
            tokens = PROD_TOK
        #El formato de salida de preprocess es de tensor por lo que lo pasamos a una lista
        input_seq = preprocess(input_seq, tokens).tolist()[0]
        selected =  preprocess([selected], tokens).tolist()[0]
        
        #Creamos una fila que contenga los nuevos datos
        row = pd.DataFrame(data={'Input_seq':[input_seq], 'Output':[selected],'Type':[option]})
        df = pd.concat([df, row], axis=0, ignore_index=True)

        #Guardamos en cada iteración el documento, nunca sabemos cuando el usuario decide cerrar la aplicación
        df.to_hdf('human_data.h5', key='df', index=False)
        num += 1
        return f'Datos guardados con éxito! Iteración n°: {num}'
        
    #El generator es simplemente para que cada vez que el usuario le de click se genere la secuencia
    generator = gr.Interface(generate, choice, 'text')
    #Con el choice.change logro que se actualice cada vez que el usuario cambia de piezas a productos
    choice.change(change_options, choice, options)
    #El saver guarda la secuencia y lo que el usuario eligió como salida
    saver = gr.Interface(save, [choice, options], 'text')
    


if __name__ == "__main__":
    main.launch()


