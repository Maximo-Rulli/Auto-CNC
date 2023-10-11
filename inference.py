#!pip install pymssql
#!pip install transformers

import pymssql
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np

SERVER = '192.168.0.247\SOLUTIIONWEB'
DATABASE = 'Solutiion'
USERNAME = 'sa'
PASSWORD = 'M0r3n02800!'
USER = 'sa'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pr = AutoModelForSequenceClassification.from_pretrained("VCNC/bert_3").to(device)
model_pi = AutoModelForSequenceClassification.from_pretrained("VCNC/bert_piezas").to(device)

#La función preprocess una vez entrada una lista con la descripción de los productos logra preprocesar la entrada para BERT
def preprocess(input:list[str], tokens):
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

def get_producto(machine):
################################### SQL CONNECTION #########################
    conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    sql_select =  f'''select DISTINCT
CASE
        WHEN PATINDEX('%[0-9]%', c.DES_PROD) > 0
            THEN SUBSTRING(c.DES_PROD, PATINDEX('%[0-9]%', c.DES_PROD), 4)
        ELSE
            NULL
    END AS DES_PROD,
P.FE_EMIT
from cargamaq2 c
join programa p on p.CG_ORDF = c.CG_ORDF

where c.cg_celda like '%{machine}%' and C.ESTADO = 0
ORDER BY FE_EMIT ASC'''

    tokens = {"OTHER": 0, "1200": 1, "1300": 2, "1400": 3, "1600": 4, "1700": 5, "1800": 6, "1900": 7, "2100": 8, "2400": 9, "2405": 9, "2600": 10, "2800": 11, "2805": 11, "AR": 12}
    cursor.execute(sql_select)
    product_list= cursor.fetchall()

    lista = []
    for row in product_list:
        if row[0] == '2805' or  row[0] == '2804':
            lista.append('2800')
        if row[0] == '1805' :
            lista.append('1800')


        else:
            if row[0] != None :
                if row[0] != '500' :
                    lista.append(row[0])
        if row[0] in tokens:
            lista.append(row[0])
        else:
            lista.append("OTHER")
    # Commit the changes to the database
    conn.commit()

    # Close the connection
    conn.close()
    return lista


def inferencia_producto(lista):
    tokens = {"OTHER": 0, "1200": 1, "1300": 2, "1400": 3, "1600": 4, "1700": 5, "1800": 6, "1900": 7, "2100": 8, "2400": 9, "2405": 9, "2600": 10, "2800": 11, "2805": 11, "AR": 12}
    input_list = lista.copy()  # Create a copy of the input list

    input_ids = preprocess(input_list, tokens).to(device)

    output = model_pr(input_ids)['logits']

    mask = torch.full_like(output, -40)  # Initialize the mask with -5

    for idx, element in enumerate(input_list):
        if element in tokens:
            token_id = tokens[element]
            mask[:, token_id] = output[:, token_id]

    # Find the element with the highest probability among the remaining elements
    max_index = torch.argmax(mask)

    # If you want to get the actual label associated with the max_index in your 'tokens' dictionary
    for label, index in tokens.items():
        if index == max_index:
            max_label = label
            break

    # Create a new list to store removed elements
    output_list = []
    removed_count = 0  # Counter to track the number of removed elements
    max_removal_limit = 15  # Maximum number of elements to remove

    for element in input_list:
        if element == max_label and removed_count < max_removal_limit:
            if element in lista:
                lista.remove(element)
                output_list.append(element)
                removed_count += 1
        elif (element == '2805' or element == '2804') and max_label == '2800':
            if element in lista:
                lista.remove(element)
                output_list.append(element)
        elif (element == '1805') and max_label == '1800':
            if element in lista:
                lista.remove(element)
                output_list.append(element)

    print("Max Index:", max_index)
    print("Max Label:", max_label)
    print(f'lista: {lista}')
    print(f'mask: {mask}')
    print(f'output: {output}')

    return output_list


def get_piezas(machine):
################################### SQL CONNECTION #########################
    """conn = pymssql.connect(server=SERVER, database=DATABASE, user=USER, password=PASSWORD)
    cursor = conn.cursor()
    sql_select =  f'''
   SELECT
    CASE
        WHEN C.DES_PROD LIKE '%TUERCA RETEN%' THEN 'TUERCA'
        WHEN C.DES_PROD LIKE '%CAPUCHON%' THEN 'CAPUCHON'
        WHEN C.DES_PROD LIKE '%CUERPO%' THEN 'CUERPO'
        WHEN C.DES_PROD LIKE '%BONETE%' THEN 'BONETE'
        WHEN C.DES_PROD LIKE '%ANILLO%' THEN 'ANILLO'
        WHEN C.DES_PROD LIKE '%DISCO%' THEN 'DISCO'
        WHEN C.DES_PROD LIKE '%CPO.TOB%' OR C.DES_PROD LIKE '%CPO TOB%' OR C.DES_PROD LIKE '%CUERPO TOBERA%' THEN 'CPO.TOB'
        WHEN C.DES_PROD LIKE '%CONTRATUERCA%' THEN 'CONTRATUERCA'
        WHEN C.DES_PROD LIKE '%TOBERA%' THEN 'TOBERA'
        WHEN C.DES_PROD LIKE '%BUJE%' THEN 'BUJE'
        WHEN C.DES_PROD LIKE '%GUIA%' OR C.DES_PROD LIKE '%GUÍA%' THEN 'GUIA'
        WHEN C.DES_PROD LIKE '%RETEN%' THEN 'RETEN'
        WHEN C.DES_PROD LIKE '%VASTAGO%' OR C.DES_PROD LIKE '%VAST%' THEN 'VASTAGO'
        WHEN C.DES_PROD LIKE '%TORN%' THEN 'TORN'
        WHEN C.DES_PROD LIKE '%PORTADISCO%' THEN 'PORTADISCO'
        WHEN C.DES_PROD LIKE '%TUERCA%' THEN 'TUERCA'
        WHEN C.DES_PROD LIKE '%CABEZA%' THEN 'CABEZA'
        WHEN C.DES_PROD LIKE '%APOYO%' THEN 'APOYO'
        WHEN C.DES_PROD LIKE '%CAJA%' THEN 'CAJA'
        WHEN C.DES_PROD LIKE '%OBTURADOR%' THEN 'OBTURADOR'
        ELSE 'OTHER'
    END AS DES_PROD_CATEGORY,

	C.CG_ORDF,
	P.FE_EMIT
FROM
    cargamaq2 C
JOIN PROGRAMA P ON C.CG_ORDF = P.CG_ORDF
WHERE
    C.CG_CELDA LIKE '%{machine}%' and estado = 0
ORDER BY
    P.FE_EMIT ASC;
    '''
    cursor.execute(sql_select)
    product_list= cursor.fetchall()

    lista = []
    for row in product_list:
        lista.append(row[0])
    # Commit the changes to the database
    conn.commit()

    # Close the connection
    conn.close()"""
    lista = ['CAPUCHON','CAPUCHON','BONETE','CUERPO','CABEZA','BONETE','TORN','CUERPO','BUJE','TORN','BUJE','CABEZA','BONETE','OTHER']
    return lista

def inferencia_piezas(lista):
    tokens = {
    "OTHER":0,
    "CAPUCHON": 1,
    "CUERPO": 2,
    "CPO": 2,
    "BONETE": 3,
    "ANILLO": 4,
    "DISCO": 5,
    "CPO.TOB": 6,
    "CPO TOB": 6,
    "CONTRATUERCA": 7,
    "TOBERA": 8,
    "BUJE": 5,
    "GUIA": 10,
    "GUÍA": 10,  # Add both forms of GUIA
    "RETEN": 11,
    "VASTAGO": 12,
    "VAST": 12,
    "TORN": 13,
    "PORTADISCO": 14,
    "TUERCA": 15,
    "CABEZA": 16,
    "APOYO": 17,
    "CAJA": 18,
    "OBTURADOR": 19,
}
    input_list = np.copy(lista)  # Create a copy of the input list
    input_list = input_list.tolist()

    input_ids = preprocess(lista, tokens).to(device)
    print(input_ids)

    output = model_pi(input_ids)['logits']
    print(f'lista: {lista}')

    mask = torch.full_like(output, -40)  # Initialize the mask with -5

    for idx, element in enumerate(lista):
        if element in tokens:
            token_id = tokens[element]
            mask[:, token_id] = output[:, token_id]

    # Find the element with the highest probability among the remaining elements
    max_index = torch.argmax(mask)

    # If you want to get the actual label associated with the max_index in your 'tokens' dictionary
    for label, index in tokens.items():
        if index == max_index:
            max_label = label
            break

    # Create a new list to store removed elements

    removed_count = 0  # Counter to track the number of removed elements
    max_removal_limit = 15  # Maximum number of elements to remove

    output_list = []

    for i in range(len(input_ids[0])):
        if input_ids[0, i] == max_index and removed_count < max_removal_limit:  # Check if we haven't reached the removal limit
            if input_ids[0, i] in input_ids:  # Check if the element is still in the original list
                output_list.append(lista[i])  # Keep the rest of the elements
                input_list.pop(i-removed_count)  # Remove the element from the original list
                removed_count += 1  # Increment the removed_count

    print("Max Index:", max_index)
    print("Max Label:", max_label)
    print(f'Mask: {mask}')
    print(f'Output: {output}')
    print('Lista output: ', output_list)
    print('Lista actualizada: ', input_list)


    return output_list, input_list

def main():
    global machine

# Assign a specific start date and time to the first element
    machine = 'CN6'

    if machine == 'CN5' or machine == 'CM1':
        prod_inicial = get_producto(machine)
        print(prod_inicial)
        output_prod_accumulated = []
        salida_sql = []

        while prod_inicial:
             prod_final = inferencia_producto(prod_inicial)
             output_prod_accumulated.extend(prod_final)
    # Insert each element from lista_final into the PROD_PLAN table
             for producto in prod_final:
                 #insert_sql(producto,machine)
                 salida_sql.append((producto, machine))

        print("output_list:", output_prod_accumulated)

    else:
        lista_inicial = get_piezas(machine)
        print(lista_inicial)
        output_list_accumulated = []
        salida_sql = []
        lista_piezas = lista_inicial

        while len(lista_piezas):
             lista_salida, lista_piezas = inferencia_piezas(lista_piezas)
             output_list_accumulated.extend(lista_salida)
     # Insert each element from lista_final into the PROD_PLAN table
             for pieza in lista_salida:
                #insert_sql(pieza,machine)
                salida_sql.append((pieza, machine))

        print("output_list:", output_list_accumulated)


if __name__ == "__main__":
    main()

